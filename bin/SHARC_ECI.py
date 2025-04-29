#!/usr/bin/env python3

# ******************************************
#
#    SHARC Program Suite
#
#    Copyright (c) 2025 University of Vienna
#
#    This file is part of SHARC.
#
#    SHARC is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    SHARC is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    inside the SHARC manual.  If not, see <http://www.gnu.org/licenses/>.
#
# ******************************************

import os
import time
import shutil
import datetime
import copy
from io import TextIOWrapper
from typing import Optional
from qmout import QMout
from constants import NUMBERS

import itertools
from ast import literal_eval as make_tuple
import numpy as np
import yaml
from SHARC_HYBRID import SHARC_HYBRID
from utils import InDir, mkdir, itnmstates, expand_path, question
from pyscf import gto
import EHF
import ECI
merge_moles = gto.mole.conc_mol


__all__ = ["SHARC_ECI"]

AUTHORS = "Tomislav Piteša & Sascha Mausenberger"
VERSION = "4.0"
VERSIONDATE = datetime.datetime(2025, 5, 1)
NAME = "ECI"
DESCRIPTION = "   HYBRID interface for EHF/ECI calculations of multichromophoric systems"

CHANGELOGSTRING = """
"""

all_features = set(  # TODO: Depends on child
    [
        "h",
        "dm",
        "mol",
        "point_charges"
    ]
)


class SHARC_ECI(SHARC_HYBRID):
    """
    Excitonic Configuration Interaction interface
    """

    _version = VERSION
    _versiondate = VERSIONDATE
    _authors = AUTHORS
    _changelogstring = CHANGELOGSTRING
    _name = NAME
    _description = DESCRIPTION

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #  self.QMin.template.types.update({"fragments": dict, "calculation": dict})
        self._template_types = {
                "paddingstates" : list,
                "fragments": { "key": str, "value": { "atoms": str,
                               #  "refcharge": { "key": int, "value": int },
                              "aufbau_site_states": [ { "Z": int, "M": int, "N": int } ],
                               "EHF": { "interface": str,
                                        "guess": (str,bool),
                                        "write": (str,bool),
                                        "max_cycles": int,
                                        "forced": bool, 
                                        "tQ": float,
                                       "embedding_site_state": { "key": int, "value": { "Z": int, "M": int, "N": int } }
                                 },
                               "SSC": { "interface": str,
                                        "data": str,
                                       "states": { "key": int, "value": [ int ] }
                                 }
                                }
                              },   
                "calculation": { "tO": float,
                                 "RI": { "active": bool,
                                         "Jauxbasis": str,
                                         "Kauxbasis": str,
                                         "tS": float,
                                         "tC": float,
                                         "chunksize": int
                                        },
                                "excitonic_basis": { "ECI": { "key": int, "value": (bool,str,list) }#,
                                                     #  "CT": { "key": int, "value": (bool,str,list) } 
                                                    },
                                 "active_integrals": (dict,str) 
                                },
        }

        self._template_defaults = {
                "paddingstates" : [],
                "fragments": { "atoms": None,
                               #  "refcharge": None,
                               "aufbau_site_states": None,
                               "EHF": { "interface": None,
                                        "guess": True, # To be changed to child.QMin.resources['pwd']/QM.out
                                        "write": True, # To be changed to child.QMin.resources['pwd']/QM.out 
                                        "max_cycles": 20,
                                        "forced": True, 
                                        "tQ": 0.001,
                                        "embedding_site_state": None
                                 },
                               "SSC": { "interface": None,
                                        "data": "w",  # To be changed to child.QMin.resources['pwd']/QM.out 
                                        "states": dict
                                 }
                             },
                "calculation": { "tO": 0.95,
                                "RI": { "active": True,
                                       "Jauxbasis": "augccpvdzri",
                                         "Kauxbasis": "augccpvdzri",
                                         "tS": 0.0001,
                                         "tC": 0.001,
                                         "chunksize": -1 
                                        },
                                 "excitonic_basis": { "ECI": {0: True, 1: "all" }},#, "CT": {0: True} },  # Default is ECIS
                                 "active_integrals": "all"
                                },
        }

        self._resources_types = self.QMin.resources.types.copy()
        self._resources_types.update({"sitejobs": [ [str, int] ]})
        self.QMin.resources.data.update({"sitejobs": None, "ngpu": 0})
        self.QMin.template.data.update(
            {
                "paddingstates" : [],
                "fragments": {}, # This will be later set up for each fragment
                "calculation" : self._template_defaults["calculation"]
            }
        )

        self.EHFjobs: dict = {}
        self.ECIjobs: dict = {}


    @staticmethod
    def version() -> str:
        return SHARC_ECI._version

    @staticmethod
    def versiondate() -> datetime.datetime:
        return SHARC_ECI._versiondate

    @staticmethod
    def changelogstring() -> str:
        return SHARC_ECI._changelogstring

    @staticmethod
    def authors() -> str:
        return SHARC_ECI._authors

    @staticmethod
    def name() -> str:
        return SHARC_ECI._name

    @staticmethod
    def description() -> str:
        return SHARC_ECI._description

    @staticmethod
    def about() -> str:
        return f"{SHARC_ECI._name}\n{SHARC_ECI._description}"

    def get_features(self, KEYSTROKES: Optional[TextIOWrapper] = None) -> set[str]:
        """return availble features

        ---
        Parameters:
        KEYSTROKES: object as returned by open() to be used with question()
        """
        if self._kindergarden: return all_features

        file =  question(
            "Please specify the path to your ECI.template file", str, KEYSTROKES=KEYSTROKES, default="ECI.template"
        )

        self.setupINFOS["template_file"] = expand_path(file)
        self.read_template(self.setupINFOS["template_file"])
        QMin = self.QMin

        self.charges_to_do = set([ Z for fdict in self.QMin.template["fragments"].values() for Z in fdict["EHF"]["embedding_site_state"].keys() ])
        # Instatiate all children
        child_dict = {}
        for Z in self.charges_to_do: # Full-system charge
            for label, fragment in QMin.template["fragments"].items():
                interface = fragment["EHF"]["interface"] 
                #child_dict[(label,'embedding',Z)] = (interface, [], {"logfile": os.path.join(label+"_embedding_Z"+str(Z), "QM.log"),"logname": label+"_embedding_Z"+str(Z) })
                child_dict[(label,'embedding',Z)] = interface
                interface = fragment["SSC"]["interface"] 
                for z in fragment["SSC"]["states"].keys(): # Fragment's charge
                    child_dict[(label,z,Z)] = interface


        num_to_key = {}
        print("Based on the template file, the following EHF and SSC children of ECI interface will be instantiated:")
        for i, (key, value) in enumerate(child_dict.items()):
            print(" ["+str(i+1)+"] "+str(key)+" "+value)
            num_to_key[i+1] = key
        answer = question("Would you like to delete (i.e., not instantiate) some of them? If so, type the comma-separated list of children's numbers in one line:", 
                          str, KEYSTROKES=KEYSTROKES, autocomplete=False, default="")
        if not answer == "":
            nums = [int(num) for num in answer.split(',')]
            for num in nums:
                del child_dict[num_to_key[num]]
        print("Instantiating the following EHF and SSC children of ECI interface:")
        for i, (key, value) in enumerate(child_dict.items()):
            print(str(key)+" "+str(value[0]))
            #  if key[1] == 'embedding':
                #  mkdir(key[0]+"_"+key[1]+"_Z"+str(key[2]))
            #  else:
                #  mkdir(key[0]+"_z"+str(key[1])+"_Z"+str(key[2]))
        self.instantiate_children(child_dict)
        for (label,z,Z), child in self._kindergarden.items():
            child_features = child.get_features(KEYSTROKES)
            if not "point_charges" in child_features:
                print("Child "+str(label,z,Z)+" does not have point_charges feature and thus cannot be a child of ECI interface. Aborting.")
                raise ValueError
            if z == "embedding" and not "multipolar_fit" in child_features:
                print("Child "+str(label,z,Z)+" does not have multipolar_fit feature and thus cannot be a child of ECI interface. Aborting.")
                raise ValueError
            if isinstance(z,int) and not "density_matrices" in child_features:
                print("Child "+str(label,z,Z)+" does not have density_matrices feature and thus cannot be a child of ECI interface. Aborting.")
                raise ValueError
        return all_features

    def get_infos(self, INFOS: dict, KEYSTROKES: Optional[TextIOWrapper] = None) -> dict:
        """prepare INFOS obj

        ---
        Parameters:
        INFOS: dictionary with all previously collected infos during setup
        KEYSTROKES: object as returned by open() to be used with question()
        """

        file = question("Please specify path to the resource file of the ECI interface:", str, default="ECI.resrouces", KEYSTROKES=KEYSTROKES)
        self.setupINFOS["resources_file"] = expand_path(file)
        #  INFOS["children_infos"] = {label:{} for label in self._kindergarden.keys()}
        for label, child in self._kindergarden.items():
            print("Getting infos of the child "+str(label))
            child.get_infos(INFOS,KEYSTROKES=KEYSTROKES)
            #child.get_infos(INFOS["children_infos"][label],KEYSTROKES=KEYSTROKES)
        return INFOS

    def prepare(self, INFOS: dict, dir_path: str):
        cwd = os.getcwd()
        os.chdir(dir_path)
        shutil.copy(self.setupINFOS['template_file'], os.getcwd())
        shutil.copy(self.setupINFOS['resources_file'], os.getcwd())
        for (label,z,Z), child in self._kindergarden.items():
            if z == 'embedding':
                child_dir = os.path.join(os.getcwd(),label+'_embedding_Z'+str(Z)) 
                mkdir(child_dir)
                child.prepare(INFOS,child_dir)
            else:
                child_dir = os.path.join(os.getcwd(),label+'_z'+str(z)+'_Z'+str(Z)) 
                mkdir(child_dir)
                child.prepare(INFOS,child_dir)
        os.chdir(cwd)
        pass

    @staticmethod
    def _format_header(sentence):
        if sentence != "": sentence = " " + sentence + " "
        total_length = 75
        eq_length = (total_length - len(sentence)) // 2
        return " " + "=" * eq_length + sentence + "=" * (total_length - len(sentence) - eq_length)

    def _check_type_recursively(self, v, t):
        if isinstance( t, type ) or isinstance( t, tuple ):
            if not isinstance( v, t ):
                self.log.error(f"Entry "+str(v)+" is of a type "+str(type(v))+" and should be of the type "+t.__name__+"!")
                raise ValueError()
        elif isinstance( t, dict ):
            if not isinstance( v, dict ):
                self.log.error(f"Entry "+str(v)+" is of a type "+str(type(v))+" and should be dictionary!")
                raise ValueError()
            elif "key" in t and "value" in t:
                for key, value in v.items():
                    self._check_type_recursively( key, t["key"] )
                    self._check_type_recursively( value, t["value"] )
            else:
                for key in v.keys():
                    if not key in t:
                        self.log.error(f"Key "+str(key)+" is found on the place where it should not be! Supported keys on that place are "+str(t.keys()))
                        raise ValueError()
                    self._check_type_recursively( v[key], t[key] )
        elif isinstance( t, list ):
            if not isinstance( v, list ):
                self.log.error(f"Entry "+str(v)+" is of a type "+str(type(v))+" and should be list!")
                raise ValueError()
            else:
                if len(t) == 1:
                    for o in v:
                        self._check_type_recursively( o, t[0] )
                else:
                    if len(t) != len(v):
                        self.log.error(f"List "+str(v)+" is of len "+str(len(v))+" and should be of len "+str(len(t))+"!")
                        raise ValueError()
                    else:
                        for ov, ot in zip(v,t):
                            self._check_type_recursively( ov, ot )


    def read_template(self, template_file: str = "ECI.template") -> None:
        """
        Parser for ECI template in yaml format

        template_file:  Path to template file
        """

        self._read_template = True
        # TODO: validate *_site_state values
        self.log.debug(f"Parsing template file {template_file}")

        # Open template file and parse yaml
        with open(template_file, "r", encoding="utf-8") as tmpl_file:
            tmpl_dict = yaml.safe_load(tmpl_file)
            self.log.debug(f"Parsing yaml file:\n{tmpl_dict}")

        # Load fragments
        if len(tmpl_dict.get("fragments",{})) < 1:
            self.log.error(f"No fragments found in {template_file}!")
            raise ValueError()
        self.log.print("Found "+str(len(tmpl_dict["fragments"]))+" fragments in template file.")
        for label, fragment in tmpl_dict["fragments"].items():
            self.QMin.template["fragments"][label] = copy.deepcopy(self._template_defaults["fragments"])
            self.QMin.template["fragments"][label].update(copy.deepcopy(fragment))
            self.QMin.template["fragments"][label]["EHF"].update(copy.deepcopy(self._template_defaults["fragments"]["EHF"]))
            self.QMin.template["fragments"][label]["EHF"].update(copy.deepcopy(fragment.get("EHF",{})))
            self.QMin.template["fragments"][label]["SSC"].update(copy.deepcopy(self._template_defaults["fragments"]["SSC"]))
            self.QMin.template["fragments"][label]["SSC"].update(copy.deepcopy(fragment.get("SSC",{})))
        # Load calculation
        self.QMin.template["calculation"] = copy.deepcopy(self._template_defaults["calculation"])
        self.QMin.template["calculation"].update(copy.deepcopy(tmpl_dict.get("calculation",{})))
        self.QMin.template["calculation"]["RI"].update(copy.deepcopy(self._template_defaults["calculation"]["RI"]))
        self.QMin.template["calculation"]["RI"].update(copy.deepcopy(tmpl_dict.get("calculation",{}).get("RI",{})))
        self.QMin.template["calculation"]["excitonic_basis"].update(copy.deepcopy(self._template_defaults["calculation"]["excitonic_basis"]))
        self.QMin.template["calculation"]["excitonic_basis"].update(copy.deepcopy(tmpl_dict.get("calculation",{}).get("excitonic_basis",{})))

        self.log.debug("Template file before checking types:")
        self.log.debug(str(self.QMin.template))
        # Verify types of entire template
        self._check_type_recursively( self.QMin.template.data, self._template_types )

        return
        
    def read_resources(self, resources_file: str = "ECI.resources") -> None:
        """
        Parser for ECI resources in yaml format

        resources_file:  Path to resource file
        """

        self._read_resources = True
        self.log.debug(f"Parsing resources file {resources_file}")

        # Open resources_file file and parse yaml
        with open(resources_file, "r", encoding="utf-8") as res_file:
          res_dict = yaml.safe_load(res_file)
          self.log.debug(f"Parsing yaml file:\n{res_dict}")

        self.QMin.resources.data.update(res_dict)
        self.QMin.resources["scratchdir"] = expand_path(self.QMin.resources["scratchdir"])
        self._check_type_recursively( self.QMin.resources.data, self._resources_types )

        self.QMin.resources['sitejobs'] = [ tuple(job) for job  in self.QMin.resources['sitejobs'] ]
        self.QMin.resources["scratchdir"] = expand_path(self.QMin.resources["scratchdir"])

        # TODO sanity checks

    def format_kindergarden(self,format):
        if "EHF" in self._kindergarden:
            if format == "divided":
                return
            elif format == "expanded":
                self._kindergarden = { (label,"embedding",Z):interface for (label,Z),interface in self._kindergarden["EHF"].items() } | { (label,z,Z):interface for (label,z,Z),interface in self._kindergarden["SSC"].items() }
                return
            else:
                raise ValueError
        else:
            if format == "divided":
                self._kindergarden = { "EHF": { (label,Z):interface for (label,z,Z),interface in self._kindergarden.items() if z == "embedding" },
                                       "SSC": { (label,z,Z):interface for (label,z,Z),interface in self._kindergarden.items() if z != "embedding" } 
                                      }
                return
            elif format == "expanded":
                return
            else:
                raise ValueError


    def setup_interface(self) -> None:
        """
        Load and initialize all child interfaces
        """
        QMin = self.QMin

        mkdir(QMin.resources['scratchdir'])

        self.charges_to_do = set([ Z for i, Z in enumerate(QMin.molecule['charge']) if QMin.molecule['states'][i] > 0 ])
        self.mults_per_charges = { Z: [ i+1 for i,s in enumerate(QMin.molecule["states"]) if s > 0 and QMin.molecule["charge"][i] == Z ] for Z in self.charges_to_do }        

        # Instatiate all children
        child_dict = {}
        for Z in self.charges_to_do: # Full-system charge
            for label, fragment in QMin.template["fragments"].items():
                interface = fragment["EHF"]["interface"] 
                child_dict[(label,'embedding',Z)] = (interface, [], {"logfile": os.path.join(label+"_embedding_Z"+str(Z), "QM.log"),"logname": label+"_embedding_Z"+str(Z) })
                interface = fragment["SSC"]["interface"] 
                # Uncomment this when CT is added
                for z in fragment["SSC"]["states"].keys(): # Fragment's charge
                    if os.path.isdir(label+"_z"+str(z)+"_Z"+str(Z)):
                        child_dict[(label,z,Z)] = (interface, [], {"logfile": os.path.join(label+"_z"+str(z)+"_Z"+str(Z), "QM.log"), "logname": label+"_z"+str(z)+"_Z"+str(Z)}) 
                #  z = fragment["refcharge"][Z]
                #  child_dict[(label,z,Z)] = (interface, [], {"logfile": os.path.join(label+"_z"+str(z)+"_Z"+str(Z), "QM.log"), "logname": label+"_z"+str(z)+"_Z"+str(Z)}) 


        self.instantiate_children(child_dict)
        self.format_kindergarden("divided")

        # Convert atoms to lists of indices
        atoms = {label:[] for label in QMin.template['fragments']}
        for label, fragment in QMin.template['fragments'].items():
            strings = fragment['atoms'].split(',')
            for group in strings:
                group2 = group.split('-')
                if len(group2) == 1:
                    atoms[label].append(int(group2[0]))
                elif len(group2) == 2:
                    for i in range( int(group2[0]), int(group2[1]) + 1 ):
                        atoms[label].append(i)
        for label, a in atoms.items():
            QMin.template['fragments'][label]['atoms'] = a

        # Call setup_mol read_resources, read_template for each EHF child, and do sanity checks
        for (label,Z), child in self._kindergarden["EHF"].items():
            qmin = {}
            atoms = self.QMin.template["fragments"][label]["atoms"] 
            embedding_site_state = self.QMin.template["fragments"][label]["EHF"]["embedding_site_state"][Z]
            z = embedding_site_state["Z"]
            m = embedding_site_state["M"]
            n = embedding_site_state["N"]
            #check whether embedding_site_state is realistic
            if n < 1: 
                self.log.error("Ordinal number of embedding site state of fragment "+label+" for full-system charge "+str(Z)+" is < 1!")
                raise ValueError
            if m < 1: 
                self.log.error("Multiplicity of embedding site state of fragment "+label+" for full-system charge "+str(Z)+" is < 1!")
                raise ValueError
            qmin["states"] = [ 0 for i in range(m-1) ] + [ n ]
            qmin["charge"] = [ 0 for i in range(m-1) ] + [ z ]
            qmin["NAtoms"] = len(atoms)
            qmin["IAn"] = [ NUMBERS[self.QMin.molecule['elements'][a]] for a in atoms ]
            qmin["retain"] = "retain 1"
            qmin["savedir"] = expand_path( os.path.join( self.QMin.save["savedir"], label+"_embedding_Z"+str(Z) ) )
            qmin["point_charges"] = True
            child.setup_mol( qmin )

            # Save the reference to the embedding site state in the template object of ECI interface
            self.QMin.template["fragments"][label]["EHF"]["embedding_site_state"][Z] = child.states[n*m-1]
            
            # Read template and resources
            child.read_resources( os.path.join( label+'_embedding_Z'+str(Z), child.name()+'.resources' ) )
            child.read_template( os.path.join( label+'_embedding_Z'+str(Z), child.name()+'.template' ) )
            scratchdir = os.path.join( expand_path(self.QMin.resources['scratchdir']), label+'_embedding_Z'+str(Z)) 
            mkdir(scratchdir)
            child.QMin.resources['scratchdir'] = scratchdir
            child.QMin.resources["pwd"] = os.path.join( self.QMin.resources["pwd"], label+"_embedding_Z"+str(Z) )
            child.QMin.resources["cwd"] = os.path.join( self.QMin.resources["cwd"], label+"_embedding_Z"+str(Z) )
            child.setup_interface()
            #  if self.QMin.template["fragments"][label]["EHF"]["guess_file"] == True:
                #  self.QMin.template["fragments"][label]["EHF"]["guess_file"] = os.path.join( child.QMin.resources["cwd"], 'QM.out' )
            #  if self.QMin.template["fragments"][label]["EHF"]["write"] == True:
                #  self.QMin.template["fragments"][label]["EHF"]["write"] = os.path.join( child.QMin.resources["cwd"], 'QM.out' )

        # Call setup_mol read_resources, read_template for each SSC child, and do sanity checks
        astates = { label: [] for label in self.QMin.template["fragments"].keys() }
        for (label,z,Z), child in self._kindergarden["SSC"].items():
            qmin = {}
            atoms = self.QMin.template["fragments"][label]["atoms"] 
            states = self.QMin.template["fragments"][label]["SSC"]["states"][z]
            aufbau_site_states = self.QMin.template["fragments"][label]["aufbau_site_states"]
            for s in aufbau_site_states:
                if s["Z"] == z and states[s["M"]-1] < s["N"]:
                    self.log.error("Aufbau site state "+str(s)+" of fragment "+label+" is not included in the site states ("+str(states)+")")
                    raise ValueError
            qmin["states"] = states.copy()
            qmin["charge"] = [ z if states[i] > 0 else 0 for i in range(len(states)) ]
            qmin["NAtoms"] = len(atoms)
            qmin["IAn"] = [ NUMBERS[self.QMin.molecule['elements'][a]] for a in atoms ]
            qmin["retain"] = "retain 1"
            qmin["savedir"] = expand_path( os.path.join( self.QMin.save["savedir"], label+"_z"+str(z)+"_Z"+str(Z) ) )
            qmin["point_charges"] = True
            child.setup_mol( qmin )

            for s in child.states:
                if any( [ astate["Z"] == s.Z and astate["M"] == s.S + 1 and astate["N"] == s.N for astate in aufbau_site_states ] ):
                    astates[label].append(s)
            
            # Read template and resources
            child.read_resources( os.path.join( label+'_z'+str(z)+'_Z'+str(Z), child.name()+'.resources' ) )
            child.read_template( os.path.join( label+'_z'+str(z)+'_Z'+str(Z), child.name()+'.template' ) )
            scratchdir = os.path.join( expand_path(self.QMin.resources['scratchdir']), label+'_z'+str(z)+'_Z'+str(Z)) 
            mkdir(scratchdir)
            child.QMin.resources['scratchdir'] = scratchdir
            child.QMin.resources["pwd"] = os.path.join( self.QMin.resources["pwd"], label+"_z"+str(z)+"_Z"+str(Z) )
            child.QMin.resources["cwd"] = os.path.join( self.QMin.resources["cwd"], label+"_z"+str(z)+"_Z"+str(Z) )
            child.setup_interface()

        # Save the reference to the aufbau site states in the template object of ECI interface
        for label, fdict in self.QMin.template["fragments"].items():
            fdict["aufbau_site_states"] = astates[label]

        # Constructing active_integrals from the template
        #  active_integrals = {"J": { (0,0): [], (0,1): [], (0,2): [],
                                        #  (1,0): [], (1,1): [] },
                                  #  "K": { (0,0): [], (0,1): [], (0,2): [],
                                        #  (1,0): [], (1,1): [] }
                                  #  }
        active_integrals = {"J": { (0,0): [], (0,1): [], (0,2): [] },
                                  "K": { (0,0): [], (0,1): [], (0,2): [] }
                                  }
        if QMin.template['calculation']['active_integrals'] == 'all':
            for JK in ['J','K']:
                #  for int_type in [ (0,0), (0,1), (0,2), (1,1) ]:
                for int_type in [ (0,0), (0,1), (0,2) ]:
                    for fpair in itertools.combinations( QMin.template['fragments'], 2 ):
                        active_integrals[JK][int_type].append(fpair)
                #  for fpair in itertools.combinations( QMin.template['fragments'], 2 ):
                    #  for spectator in QMin.template['fragments']:
                        #  ftriple = (fpair[0], fpair[1], spectator)
                        #  active_integrals[JK][(1,0)].append(ftriple)
        else: 
            for JK, value in QMin.template['calculation']['active_integrals'].items(): 
                if value == 'all':
                    #  for int_type in [ (0,0), (0,1), (0,2), (1,1) ]:
                    for int_type in [ (0,0), (0,1), (0,2) ]:
                        for fpair in itertools.combinations( QMin.template['fragments'], 2 ):
                            active_integrals[JK][int_type].append(fpair)
                        #  for fpair in itertools.combinations( QMin.template['fragments'], 2 ):
                            #  for spectator in QMin.template['fragments']:
                                #  ftriple = (fpair[0], fpair[1], spectator)
                                #  active_integrals[JK][(1,0)].append(ftriple)
                else:
                    for int_type, multiples in value.items():
                        if multiples == 'all':
                            #  if int_type == '(1,0)':
                                #  for fpair in itertools.combinations( QMin.template['fragments'], 2 ):
                                    #  for spectator in QMin.template['fragments']:
                                        #  ftriple = (fpair[0],fpair[1],spectator)
                                        #  active_integrals[JK][(1,0)].append(ftriple)
                            #  else:
                                #  for fpair in itertools.combinations( QMin.template['fragments'], 2 ):
                                    #  active_integrals[JK][make_tuple(int_type)].append(fpair)
                            for fpair in itertools.combinations( QMin.template['fragments'], 2 ):
                                active_integrals[JK][make_tuple(int_type)].append(fpair)
                        else:
                            for multiple in multiples:
                                active_integrals[JK][make_tuple(int_type)].append(tuple(multiple))

        QMin.template['calculation']['active_integrals'] = active_integrals

        # Build actual excitations and charge_transfer items from the template
        basis = self.QMin.template['calculation']['excitonic_basis']
        # excitations
        ECI = {}
        ECI[0] = basis['ECI'].get(0,False)
        del basis['ECI'][0] 
        for rank, value in basis['ECI'].items():
            ECI[rank] = []
            if value == 'all':
                for subset in itertools.combinations( QMin.template['fragments'], rank ):
                    ECI[rank].append(list(subset))
            else:
                for subset in value:
                    ECI[rank].append(list(subset))
        basis['ECI'] = ECI
        # charge-transfers
        #  CT = {}
        #  CT[0] = basis['CT'].get(0,False)
        #  if CT[0]: CT[0] = [ ((),()) ]
        #  if not CT[0]: CT[0] = []
        #  del basis['CT'][0] 
        #  for rank, value in basis['CT'].items():
            #  CT[rank] = []
            #  if value == 'all':
                #  for donors in itertools.combinations_with_replacement(QMin.template['fragments'], rank):
                    #  for acceptors in itertools.combinations_with_replacement(QMin.template['fragments'], rank): 
                        #  mutual = [ d == a for d in donors for a in acceptors ]
                        #  #  print('donors = ', donors)
                        #  #  print('acceptors = ', acceptors)
                        #  #  print('mutual = ', mutual)
                        #  if not any(mutual):
                            #  CT[rank].append((donors,acceptors))
            #  else: # ToDo
                #  for subset in value:
                    #  CT[rank].append(list(subset))
        #  basis['CT'] = CT

        # Set inevitable requests to the children
        # Set point-charge request to all embedding children
        mkdir(self.QMin.save["savedir"], force=False)
        for (label,Z), child in self._kindergarden["EHF"].items():
            s = QMin.template['fragments'][label]["EHF"]['embedding_site_state'][Z] 
            child.read_requests({"h": True, "multipolar_fit": [ (s.S+1, s.N, s.S+1, s.N) ]})
            #  child.QMin.requests['mol'] = True 

        # Set density requests 
        for (label,z,Z), child in self._kindergarden["SSC"].items():
            #  child.read_requests({"h": True, "density_matrices": ["all"], "mol": True})
            child.QMin.requests['h'] = True
            child.QMin.requests['density_matrices'] = ["all"]
            child.QMin.requests['mol'] = True

        #  Setup children interfaces
        #  for (label,C), child in self.embedding_kindergarden.items():
            #  child.setup_interface()
            #  child.QMin.resources['pwd'] = os.path.join( os.getcwd(), label+'_embedding_C'+str(C) ) 
            #  child.QMin.save['savedir'] = os.path.join( QMin.save['savedir'], label+'_embedding_C'+str(C), 'SAVE' ) 
        #  for (label,c,C), child in self._kindergarden.items():
            #  child.setup_interface()
            #  child.QMin.resources['pwd'] = os.path.join( os.getcwd(), label+'_c'+str(c)+'_C'+str(C) ) 
            #  child.QMin.save['savedir'] = os.path.join( QMin.save['savedir'], label+'_c'+str(c)+'_C'+str(C), 'SAVE' ) 

        self.EHFjobs = {Z: None for Z in self.charges_to_do}
        self.SSCjobs = {Z: None for Z in self.charges_to_do}

        return

    def read_requests(self, requests_file: str= "QM.in") -> None:
        super().read_requests(requests_file)

        # Make requests for all children, depending on which requests ECI interface got
        # Inevitable requests (energies and densities) are already given in setup_interface
        QMin = self.QMin

        for (label,Z), child in self._kindergarden["EHF"].items():
            with InDir(QMin.save['savedir']):
                mkdir( label+'_embedding_Z'+str(Z), force=False)
        for (label,z,Z), child in self._kindergarden["SSC"].items():
            with InDir(QMin.save['savedir']):
                mkdir( label+'_z'+str(z)+'_Z'+str(Z), force=False )

        # Sets all logical requests
        for request in ['dm']:
            for child in self._kindergarden["SSC"].values():
                child.QMin.requests[request] = self.QMin.requests[request] # True/False

        return

    def set_coords(self, xyz: str='QM.in') -> None:
        super().set_coords(xyz)

        QMin = self.QMin

        for Z in self.charges_to_do:
            # Set coords
            for label1 in self.QMin.template["fragments"].keys():
                atoms = self.QMin.template['fragments'][label1]['atoms']
                child1 = self._kindergarden["EHF"][(label1,Z)]
                child1.set_coords( self.QMin.coords['coords'][ atoms ] ) 
                for z1 in self.QMin.template["fragments"][label1]["SSC"]["states"].keys(): 
                    # Remove try when CT is added
                    try:
                        child1 = self._kindergarden["SSC"][(label1,z1,Z)]
                        child1.set_coords( self.QMin.coords['coords'][ atoms ] ) 
                    except:
                        pass

            # Set pccoords
            for label1 in self.QMin.template["fragments"].keys():
                # EHF children
                child1 = self._kindergarden["EHF"][(label1,Z)]
                pccoords, pccharge = [], []
                for label2 in self.QMin.template["fragments"].keys():
                    if label2 != label1:
                        child2 = self._kindergarden["EHF"][(label2,Z)]
                        pccoords = pccoords + [ child2.QMin.coords['coords'] ]
                        pccharge = pccharge + [0.]*child2.QMin.coords['coords'].shape[0] # True values are gonna be set in EHF
                if QMin.coords['pccoords'] != None: 
                    pccoords = pccoords + [ self.QMin.coords['pccoords'] ]
                    pccharge.extend(self.QMin.coords['pccharge']) # These are the true values already
                pccoords = np.concatenate( pccoords, axis=0 )
                child1.set_coords( pccoords, pc=True)
                child1.QMin.coords['pccharge'] = pccharge[:]
                # SSC children
                for z1 in self.QMin.template["fragments"][label1]["SSC"]["states"].keys(): 
                    # Remove try after CT is added
                    try:
                        child1 = self._kindergarden["SSC"][(label1,z1,Z)]
                        pccoords, pccharge = [], []
                        for label2 in self.QMin.template["fragments"].keys():
                            if label2 != label1:
                                child2 = self._kindergarden["EHF"][(label2,Z)]
                                pccoords = pccoords + [ child2.QMin.coords['coords'] ]
                                pccharge = pccharge + [0.]*child2.QMin.coords['coords'].shape[0] # True values are gonna be set in EHF
                        if QMin.coords['pccoords'] != None: 
                            pccoords = pccoords + [ self.QMin.coords['pccoords'] ]
                            pccharge.extend(self.QMin.coords['pccharge']) # These are the true values already
                        pccoords = np.concatenate( pccoords, axis=0 )
                        child1.set_coords( pccoords, pc=True)
                        child1.QMin.coords['pccharge'] = pccharge[:]
                    except:
                        pass


        return

    def run(self):
        QMin = self.QMin
         
        #  DOs = { Z:{ label:{} for label in QMin.template['fragments']} for Z in self.charges_to_do}
        self.log.print('Full-system charges to do: '+str(self.charges_to_do))
        self.log.print("")
        for Z in self.charges_to_do:
            self.log.print(self._format_header(""))
            self.log.print(self._format_header("ECI CALCULATION FOR THE FULL-SYSTEM CHARGE "+str(Z)))
            self.log.print(self._format_header(""))
            self.log.print("")
            ehf_garden = { key[0]:interface for key,interface in self._kindergarden["EHF"].items() if key[1] == Z }
            ssc_garden = { (key[0],key[1]):interface for key,interface in self._kindergarden["SSC"].items() if key[2] == Z }

            self.log.print("")
            self.log.print(self._format_header("EHF calculation"))
            self.log.print("")
            # Make guesses for embedding charges
            echarges = { label: None for label in QMin.template['fragments'] }
            estates = { label:QMin.template['fragments'][label]['EHF']['embedding_site_state'][Z] for label in QMin.template['fragments']}
            maxcycles = { label:QMin.template['fragments'][label]['EHF']['max_cycles'] for label in QMin.template['fragments'] }
            forced = { label:QMin.template['fragments'][label]['EHF']['forced'] for label in QMin.template['fragments'] }
            tQ = { label:QMin.template['fragments'][label]['EHF']['tQ'] for label in QMin.template['fragments'] }
            for label, child in ehf_garden.items():
                zero = True
                if self.QMin.template["fragments"][label]["EHF"]["guess"]:
                    file = os.path.join( child.QMin.resources["cwd"], 'QM.out' ) 
                    self.log.print('    Trying to read guess embedding charges for fragment '+label+'...')
                    charges = child.QMin.molecule["charge"] 
                    zero = False
                    try:
                        child.QMout = QMout( filepath=file, charges=charges )
                        self.log.print('       QM.out file successfully read!')
                        try:
                            echarges[label] = child.QMout['multipolar_fit'][(estates[label],estates[label])][:,0]
                            self.log.print('       RESP charges found in it and assigned as the guess charges for EHF!')
                        except:
                            self.log.print('       RESP charges of the embedding site state are probably not present in the QM.out file! Defaulting to zero embedding charges...')
                            zero = True
                    except:
                        self.log.print('       Failed! Defaulting guess charges to zeros...')
                        zero = True
                if zero:
                    echarges[label] = np.zeros(child.QMin.molecule['natom'])
            self.log.print("")

            # Do EHF
            EHFjob = self.EHFjobs[Z]
            EHFjob = EHF.EHF(nproc=self.QMin.resources['ncpu'],
                             echarges=echarges,
                             estates=estates,
                             maxcycles=maxcycles,
                             forced=forced,
                             egarden=ehf_garden,
                             tQ=tQ,
                             log=self.log)
            EHFjob.run()
            self.EHFjobs[Z] = EHFjob
            # Write QM.out files of EHF children if needed
            for label, child in ehf_garden.items():
                if self.QMin.template["fragments"][label]["EHF"]["write"] and self.QMin.template["fragments"][label]["EHF"]["max_cycles"] > 0:
                    file = os.path.join( child.QMin.resources["cwd"], 'QM.out' ) 
                    self.log.print('    Writing QMout object of EHF fragment '+label+' to '+file)
                    child.writeQMout( filename=file )
            self.log.print("")
            self.log.print(self._format_header("End of EHF calculation"))
            self.log.print("")

            # Do SSC if needed fragment-wise
            self.log.print("")
            self.log.print(self._format_header("Site-state calculations"))
            self.log.print("")
            ssc_rungarden = {}
            for (label,z), child in ssc_garden.items():
                if "r" in self.QMin.template["fragments"][label]["SSC"]["data"]:
                    file = os.path.join( child.QMin.resources["cwd"], 'QM.out' ) 
                    charges = child.QMin.molecule['charge'] 
                    self.log.print('    Trying to read site-state data for fragment '+label+' with charge '+str(z)+'...')
                    try:
                        child.QMout = QMout( filepath=file, charges=charges )
                        self.log.print('       QM.out file successfully read!')
                    except:
                        self.log.print('       Failed! Aborting.')
                        raise ValueError
                else:
                    ssc_rungarden[(label,z)] = child
            if len(ssc_rungarden) > 0:
                self.log.print('    SSC children to be runned: '+', '.join([ str(c) for c in ssc_rungarden.keys() ] ))
            else:
                self.log.print('    No SSC children need to be runned.')

            for (label,z), child in ssc_rungarden.items():
                # Writting final embedding charges to the SSC children
                PCs = np.concatenate( [ charges for elabel,charges in self.EHFjobs[Z].echarges.items()  if elabel != label ] )
                child.QMin.coords['pccharge'][0:PCs.shape[0]] = PCs

            # Running actual children
            t1 = time.time()
            ssc_rungarden = { label:ssc_rungarden[label] for label in QMin.resources['sitejobs'] if label in ssc_rungarden }
            self.run_queue(self.log, ssc_rungarden, QMin.resources['ncpu'],indent=' '*4)
            t2 = time.time()
            self.log.print("")
            self.log.print('    Time elapsed in site-state calculations = '+str(round(t2-t1,3))+' sec.')
            for (label,z), child in ssc_garden.items():
                if "w" in self.QMin.template["fragments"][label]["SSC"]["data"]:
                    file = os.path.join( child.QMin.resources["cwd"], 'QM.out' ) 
                    self.log.print('    Writing QMout object of SSC fragment '+str((label,z))+' to '+file) 
                    child.writeQMout( filename=file )
            self.log.print("")
            self.log.print(self._format_header("End of site-state calculations"))
            self.log.print("")

#            # Calculate or read site-specific Dyson orbitals if needed
#            # Dyson orbitals cannot be read from children's QM.out because even if read_children = true, CT level is changed
#            for label in QMin.template['fragments']:
#                dyson_garden = {key[1]:value for key,value in garden.items() if key[0] == label}
#                for c1, child1 in dyson_garden.items():
#                    for c2, child2 in dyson_garden.items():
#                        if c1 ==  c2 - 1:
#                            if not 'r' in QMin.template['calculation']['manage_children']:
#                                self.log.print(' Calculating the Dyson orbitals of fragment '+label+' between charges '+str(c1)+' and '+str(c2))
#                                DOs[C][label][(c1,c2)] = child1.dyson_orbitals_with_other(child2,QMin.resources['scratchdir'],QMin.resources['ncpu'],"64000")
#                            else:
#                                f = open( os.path.join( QMin.resources['cwd'], label+'_C'+str(C)+'_c'+str(c1)+'_c'+str(c2)+'.dyson'), 'r')
#                                lines = f.readlines()
#                                DOs[C][label][(c1,c2)] = {}
#                                for line in lines:
#                                    thes1, thes2, spin, coeffs = line.split('|')
#                                    for s1 in child1.states:
#                                        if s1.symbol(True,True) == thes1:
#                                            break
#                                    for s2 in child2.states:
#                                        if s2.symbol(True,True) == thes2:
#                                            break
#                                    DOs[C][label][(c1,c2)][(s1,s2,spin)] = np.array( [ float(x) for x in coeffs.split() ] )
#                                f.close()
#                            if 'w' in QMin.template['calculation']['manage_children']: 
#                                f = open( os.path.join( QMin.resources['cwd'], label+'_C'+str(C)+'_c'+str(c1)+'_c'+str(c2)+'.dyson'), 'w' )
#                                for (s1,s2,spin), phi in DOs[C][label][(c1,c2)].items():
#                                    f.write(s1.symbol(True,True)+'|'+s2.symbol(True,True)+'|'+spin+'|'+' '.join([ str(round(phi[i],10)) for i in range(len(phi))] )+'\n' )
#                                f.close()
#                #  involved, active = False, False
#                #  for ct, value in QMin.template['calculation']['excitonic_basis']['CT'].items():
#                    #  if ct > 0:
#                        #  if label in set([ item for pair in value for item in pair ]):
#                            #  involved = True
#                            #  break
#                #  actives = []
#                #  for JK in ["J","K"]:
#                    #  for int_types in [(1,0),(1,1)]:
#                        #  actives += QMin.template['calculation']['active_integrals'][JK][int_types]  
#                #  actives = set([ item for subset in actives for item in subset ])
#                #  if label in actives: active = True
#                #  if active and involved:
#                    #  dyson_garden = {key[1]:value for key,value in garden.items() if key[0] == label}
#                    #  for c1, child1 in dyson_garden.items():
#                        #  for c2, child2 in dyson_garden.items():
#                            #  if c1 ==  c2 - 1:
#                                #  DOs[C][label][(c1,c2)] = child1.dyson_orbitals_with_other(child2)
#
            # Calculate dipole-moment matrices of children, should be done in children
            if self.QMin.requests['dm']:
                self.log.print('')
                self.log.print(' Calculating site-specific dipole-moment matrices from the densities...')
                self.log.print('')
                for (label,z), child in ssc_garden.items():
                    iterator = list(enumerate(itnmstates(child.QMin.molecule['states'])))
                    child.QMout['dm'] = {}
                    mol = child.QMout['mol']
                    nuclear_moment = np.sum(np.array([mol.atom_charge(j) * mol.atom_coord(j) for j in range(mol.natm)]), axis=0)
                    mu = mol.intor("int1e_r")
                    dm = np.zeros((3,child.QMin.molecule['nmstates'], child.QMin.molecule['nmstates']),dtype=complex)
                    for i1, s1 in enumerate(child.states):
                        for i2, s2 in enumerate(child.states):
                            if (s1,s2,'tot') in child.QMout['density_matrices']:
                                rho = child.QMout['density_matrices'][(s1,s2,'tot')]
                                x = -np.einsum("xij,ij->x", mu, rho)
                                if s1 is s2: x += nuclear_moment
                                dm[:,i1,i2] = x.astype(complex)
                    child.QMout['dm'] = dm.copy() 

            # Start filling the site-specific data and build fragment instances
            sites = []
            for flabel, fdict in QMin.template['fragments'].items():
                H = {}
                rho = {}
                mu = {}
                grad = {}
                states = {}
                #  phi = DOs[C][flabel]
                #  charge = fdict['refcharge'][Z]
                mol = [ child.QMout['mol'] for (label,z), child in ssc_garden.items() if label == flabel ][0]
                aufbau_states = fdict['aufbau_site_states']

                #  for z, nstates in fdict['SSC']['states'].items():
                for (label,z), child in ssc_garden.items():
                    if label == flabel:
                        #  child = ssc_garden[(flabel,z)]
                        states[z] = child.states
                        H[z] = child.QMout['h']
                        rho[z] = child.QMout['density_matrices']
                        if 'dm' in QMin.requests:
                            mu[z] = {}
                            for i1, s1 in enumerate(states[z]):
                                for i2, s2 in enumerate(states[z]):
                                    if (s1,s2,'tot') in rho[z]:
                                        mu[z][(s1,s2)] = child.QMout['dm'][:,i1,i2].astype(float)

                site = ECI.fragment(label=flabel,
                                    #  Z=charge,
                                    mol=mol,
                                    states=states,
                                    aufbau_states=aufbau_states,
                                    H=H,
                                    rho=rho,
                                    mu=mu,
                                    #  phi=DOs[C][flabel],
                                    Q=echarges[flabel],
                                    index=len(sites))
                sites.append( site )

            # Make job instance
            properties = [ prop for prop in ['dm'] if QMin.requests[prop]]
            job = ECI.calculation( ncpu=QMin.resources['ncpu'],
                                   mem=QMin.resources['memory'],
                                   charge=Z, 
                                   multiplicities=[m+1 for m, nstates in enumerate(QMin.molecule['states']) if nstates > 0 and QMin.molecule['charge'][m] == Z ],
                                   tO=QMin.template['calculation']['tO'],
                                   eci_level=QMin.template['calculation']['excitonic_basis']['ECI'],
                                   #  ct_level=QMin.template['calculation']['excitonic_basis']['CT'],
                                   active_integrals=copy.deepcopy(QMin.template['calculation']['active_integrals']), 
                                   ri=QMin.template['calculation']['RI'],
                                   properties=properties
                                  )
            # Initialize ECI instance
            self.ECIjobs[Z] = ECI.ECI( job=job, sites=sites, log=self.log )
            # Run ECI calculation
            self.log.print("")
            self.log.print(self._format_header("Excitonic part of the ECI calculation"))
            self.log.print("")
            t1 = time.time()
            self.ECIjobs[Z].run()
            t2 = time.time()
            self.log.print('    Time elapsed in the excitonic part = '+str(round(t2-t1,3))+' sec.')
            self.log.print(self._format_header("End of excitonic part of the ECI calculation"))
            self.log.print("")
            self.log.print(self._format_header(""))
            self.log.print(self._format_header("END OF ECI CALCULATION FOR THE FULL-SYSTEM CHARGE "+str(Z)))
            self.log.print(self._format_header(""))
            self.log.print("")
            self.log.print("")
            self.log.print("")
        return

    def getQMout(self):
        QMin = self.QMin
        QMout = self.QMout
        ECIjobs = self.ECIjobs

        states = QMin.molecule["states"]
        nmstates = QMin.molecule["nmstates"]
        natom = QMin.molecule["natom"]
        self.QMout.allocate(
            states, natom, QMin.molecule["npc"], {r for r in QMin.requests.keys() if QMin.requests[r]}
        )


        iterator = list(enumerate(itnmstates(QMin.molecule['states'])))
        for request, value in QMin.requests.items():
            if value:
                if request == 'h':
                    for i, s in enumerate(self.states):
                        self.QMout['h'][i,i] = ECIjobs[s.Z].E[s.S+1][s.N-1].astype(complex)
                if request == 'dm':
                    for i1, s1 in enumerate(self.states):
                        for i2, s2 in enumerate(self.states):
                            if s1.Z == s2.Z and s1.S == s2.S:
                                self.QMout['dm'][:,i1,i2] = ECIjobs[s1.Z].mu[s1.S+1][:,s1.N-1,s2.N-1]
                if request == 'mol':
                    Z = list(self.charges_to_do)[0]
                    for i, flabel in enumerate(QMin.template['fragments']):
                        child = [ child for (label,z,Z), child in self._kindergarden["SSC"].items() if label == flabel ][0]
                        if i == 0:
                            self.QMout['mol'] = child.QMout['mol']
                        else:
                            self.QMout['mol'] = merge_moles( self.QMout['mol'], child.QMout['mol'] )

        return self.QMout 

    def clean_savedir(self):
        self.format_kindergarden("expanded")
        super().clean_savedir()
        self.format_kindergarden("divided")

    def write_step_file(self):
        self.format_kindergarden("expanded")
        super().write_step_file()
        self.format_kindergarden("divided")

    def create_restart_files(self) -> None:
        """
        Create restart files
        """

    def dyson_orbitals_with_other(self, other):
        pass


if __name__ == "__main__":
    SHARC_ECI().main()

