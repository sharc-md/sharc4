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
import datetime
from io import TextIOWrapper
from typing import Optional
from qmin import QMinRequests
from qmout import QMout
from constants import ATOMCHARGE ,NUMBERS

import itertools
import numpy as np
from multiprocessing import Pool
import yaml
from SHARC_HYBRID import SHARC_HYBRID
from utils import InDir, electronic_state, mkdir, itnmstates
import ECI
import EHF
from pyscf import gto
from ast import literal_eval as make_tuple
merge_moles = gto.mole.conc_mol


__all__ = ["SHARC_ECI"]

AUTHORS = "Tomislav Piteša & Sascha Mausenberger"
VERSION = "1.0"
VERSIONDATE = datetime.datetime(2024, 2, 20)
NAME = "ECI"
DESCRIPTION = "   HYBRID interface for excitonic HF/CI with multiple fragments"

CHANGELOGSTRING = """
"""

all_features = set(  # TODO: Depends on child
    [
        "h",
        "dm",
        "mol"
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

        self.QMin.template.data.update(
            {
                "fragments": {},
                "charge": [],
                "calculation": {
                    "EHF_maxcycle": 20,
                    "tQ": 1e-4,
                    "tO": 0.95,
                    "ri": False,
                    "Jauxbasis": 'def2svpjfit',
                    "Kauxbasis": 'def2svpjkfit',
                    "tS": 1e-12,
                    "tL": 1e-5,
                    "manage_children": 'w',
                    "excitonic_basis": None,
                    "active_integrals": None
                },
            }
        )
        self.QMin.template.types.update({"fragments": dict, "charge": list, "calculation": dict})

        self._calculation_types = {
            "EHF_maxcycle": int,
            "tQ": (float, int),
            "tO": (float, int),
            "ri": bool,
            "Jauxbasis": str,
            "Kauxbasis": str,
            "excitonic_basis": dict,
            "manage_children": str,
            "tS": (float, int),
            "tL": (float, int),
            "active_integrals": (dict,str)
        }

        self.fragmentation: dict = {}
        self.embedding_kindergarden: dict = {}
        self.embedding_states: dict = {}
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
        return all_features

    def get_infos(self, INFOS: dict, KEYSTROKES: Optional[TextIOWrapper] = None) -> dict:
        """prepare INFOS obj

        ---
        Parameters:
        INFOS: dictionary with all previously collected infos during setup
        KEYSTROKES: object as returned by open() to be used with question()
        """
        return INFOS

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

        self.QMin.template['charge'] = tmpl_dict['charge']
        self.QMin.template['fragments'].update(tmpl_dict['fragments'])
        self.QMin.template['calculation'].update(tmpl_dict['calculation'])
        #self.QMin.template.update(tmpl_dict)

        # Validate charge
        if self.QMin.template["charge"] is None:
            self.log.error(f"No charges specified in {template_file}!")
            raise ValueError()
        if not all(isinstance(val, int) for val in self.QMin.template["charge"]):
            self.log.error("Charge list must contain integers!")
            raise ValueError()

        # Validate calculation
        for k, v in self.QMin.template["calculation"].items():
            if not isinstance(v, self._calculation_types[k]):
                expected_type = (
                    self._calculation_types[k]
                    if not isinstance(self._calculation_types[k], tuple)
                    else self._calculation_types[k][0]
                )
                self.log.error(f"Calculation parameter {k} must be {expected_type.__name__}, but is {type(v).__name__}!")
                raise ValueError()

        # Validate fragments
        if self.QMin.template["fragments"] is None:
            self.log.error(f"No fragments found in {template_file}!")
            raise ValueError()

        if len(self.QMin.template["fragments"]) < 1:
            self.log.error(f"No fragments found in {template_file}!")
            raise ValueError()
        self.log.debug(f"Found {len(self.QMin.template['fragments'])} fragments in {template_file}.")


        # Validate fragment keys
        CHARGES = {}
        for frag, key in self.QMin.template["fragments"].items():
            CHARGES[frag] = []
            if "atoms" not in key:
                self.log.error(f"No atoms specified for fragment {frag}!")
                raise ValueError()
            else:
                key["atoms"] = str(key["atoms"])

            if "interface" not in key:
                self.log.error(f"No interface specified for fragment {frag}!")
                raise ValueError()

            if "densrep" not in key:
                self.log.debug(f"Setting default densrep (AO) to fragment {frag}")
                self.QMin.template["fragments"][frag]["densrep"] = "AO"

            match key["densrep"].upper():
                case "AO" | "DME":
                    pass
                case _:
                    self.log.error(f"Invalid densrep key {key['densrep']} in fragment {frag}!")
                    raise ValueError()

            if not "refcharge" in key:
                self.log.error(f"No refcharge for fragment {frag} specified!")
                raise ValueError()
            if not isinstance(key["refcharge"], dict):
                self.log.error(f"refcharges in fragment {frag} must be a dictionary!")
                raise ValueError()
            for k, v in key["refcharge"].items():
                if not isinstance(k, int) or not isinstance(v, int):
                    self.log.error(f"refcharge in fragment {frag} must be integers!")
                    raise ValueError()

            if "aufbau_site_states" in key:
                if not isinstance(key["aufbau_site_states"], list):
                    self.log.error(f"aufbau_site_states in fragment {frag} must be a dictionary!")
                    raise ValueError()
                for i in key["aufbau_site_states"]:
                    if not self._check_zmn(i):
                        self.log.error(
                            f"aufbau_site_states dicts in fragment {frag} must contain keys Z, M, N with integer values!"
                        )
                        raise ValueError()

            if "site_states" not in key:
                self.log.error(f"site_states not defined in fragment {frag}!")
                raise ValueError()

            for k, v in key["site_states"].items():
                if not isinstance(k, int) or not isinstance(v, list):
                    self.log.error(f"site_state dictionary in fragment {frag} must contain integer keys and list values!")
                    raise ValueError()
                if not all(isinstance(val, int) for val in v):
                    self.log.error(f"Values of site_state values lists in fragment {frag} must be integer!")
                    raise ValueError()

            if "embedding_site_state" in key:
                if not isinstance(key["embedding_site_state"], dict):
                    self.log.error(f"embedding_site_state in fragment {frag} must be a dictionary!")
                    raise ValueError()
                if not all(isinstance(val, int) for val in key["embedding_site_state"].keys()):
                    self.log.error(f"Keys in embedding_site_state dictionary in fragment {frag} must be integer!")
                    raise ValueError()
                for embeddings in key["embedding_site_state"].values():
                    if not self._check_zmn(embeddings):
                        self.log.error(
                            f"embedding_site_state dicts in fragment {frag} must contain keys Z, M, N with integer values!"
                        )
                        raise ValueError()

            # Check whether the fragment has refcharge and embedding_site_state 
            rc = set(key["refcharge"])
            ess = set(key["embedding_site_state"])
            if rc != ess:
                self.log.error(
                        f"Full-system charges in refcharge and embedding_site_state dictionaries are not matching!"
                        )
                raise ValueError()
            CHARGES[frag] = set(rc)

            # Check whether all aufbau_site_states are in the site_states:
            for s in key["aufbau_site_states"]:
                message = f"Aufbau site state {s} of fragmnet {frag} is not included in the site states of the fragment!"
                if not s["Z"] in key["site_states"]:
                    self.log.error(message)
                    raise ValueError()
                if len(key["site_states"][s["Z"]]) < s["M"]:
                    self.log.error(message)
                    raise ValueError()
                if key["site_states"][s["Z"]][s["M"]-1] < s["N"]:
                    self.log.error(message)
                    raise ValueError()

        # Check whether all fragments have needed specifications for all full-system charges
        for f1 in self.QMin.template['fragments']:
            for f2 in self.QMin.template['fragments']:
                if set(CHARGES[f1]) != set(CHARGES[f2]):
                    self.log.error(
                            f"Not all fragments have refcharge and embedding_site_state dictionaries specified for the all full-system charges!"
                            )
                    raise ValueError()

        # Create charges atribute and check whether all requested charges are there
        # The later part is gonna be moved to read_requests function once charges are upgraded to the master level
        self.charges = CHARGES[next(iter(self.QMin.template['fragments']))]
        self.charges_to_do = set()
        for M, N in enumerate(self.QMin.molecule['states']):
            if N > 0:
                print(M)
                if not self.QMin.template['charge'][M] in self.charges:
                    self.log.error(
                            f"Requested full-system charge {C} cannot be calculated because refcharge and embedding_site_state of any fragmnet are not define for it!"
                            )
                    raise ValueError()
                else:
                    self.charges_to_do.add(self.QMin.template['charge'][M])

        for C in self.charges: 
            self.EHFjobs[C] = None
            self.ECIjobs[C] = None

        # Copied from SHARC_INTERFACE.read_template, will have to be done properly
        for s, nstates in enumerate(self.QMin.molecule["states"]):
            c = self.QMin.template["charge"][s]
            for n in range(nstates):
                for m in range(-s, s + 1, 2):
                    self.states.append(
                        electronic_state(Z=c, S=s, M=m, N=n + 1, C={})
                    )  # This is the moment in which states get their pointers
        return
        
    def read_resources(self, resources_file: str = "ECI.resources") -> None:
        """
        Parser for ECI resources in yaml format

        resources_file:  Path to template file
        """

        # TODO: validate *_site_state values
        self.log.debug(f"Parsing resources file {resources_file}")

        # Open resources_file file and parse yaml
        with open(resources_file, "r", encoding="utf-8") as res_file:
          res_dict = yaml.safe_load(res_file)
          self.log.debug(f"Parsing yaml file:\n{res_dict}")

        self.QMin.resources.update(res_dict)

        newlist = []
        for job in self.QMin.resources['ECI_sitejobs']:
            newlist.append(tuple(job))
        self.QMin.resources['ECI_sitejobs'] = newlist
        self._read_resources = True

        # TODO sanity checks

    def _check_zmn(self, zmn_dict: dict[str, int]) -> bool:
        """
        Check if dictionary contains Z, M, N keys and validate if values are int
        """
        if not isinstance(zmn_dict, dict) or zmn_dict.keys() != {"Z", "M", "N"}:
            return False
        if not all(isinstance(val, int) for val in zmn_dict.values()):
            return False
        return True

    def _setup_children_mol(self,child,atoms,states,charges) -> None:
        basic_infos = {
            "NAtoms": len(atoms),
            "states": states,
            "charge": charges,
            "IAn": [  NUMBERS[self.QMin.molecule['elements'][a]] for a in atoms ],
            "retain": "retain 1"
        }
        child.setup_mol(basic_infos)
        # QMin = self.QMin
        # #  child.QMin.save['step'] = QMin.save['step']
        # child.QMin.molecule['natom'] = len(atoms)
        # child.QMin.molecule['elements'] = [ QMin.molecule['elements'][a] for a in atoms ]
        # child.QMin.molecule["Atomcharge"] = sum(map(lambda x: ATOMCHARGE[x], child.QMin.molecule["elements"]))
        # child.QMin.molecule['frozcore'] = 0
        # states_dict = self.parseStates(states)
        # if len(states_dict["states"]) < 1:
        #     self.log.error("Number of states must be > 0!")
        #     raise ValueError()
        # child.QMin.maps["statemap"] = states_dict["statemap"]
        # child.QMin.molecule["nstates"] = states_dict["nstates"]
        # child.QMin.molecule["nmstates"] = states_dict["nmstates"]
        # child.QMin.molecule["states"] = states_dict["states"]
        # child.QMin.molecule['unit'] = 'bohr'
        # child.QMin.molecule['factor'] = 1. 
        # child.QMin.molecule['point_charges'] = True
        # child.QMin.molecule["charge"] = charges
        # #  child.QMin.save['savedir'] = os.path.join( QMin.save['savedir'], 'SAVEDIR')
        # child._setup_mol = True
        return

    def setup_interface(self) -> None:
        """
        Load and initialize all child interfaces
        """
        QMin = self.QMin

        mkdir(QMin.resources['scratchdir'])

        # Instatiate all children
        child_dict = {}
        for C in self.charges: # Full-system charge
            for label, fragment in QMin.template['fragments'].items():
                for c in fragment['site_states'].keys(): # Fragment's charge
                    interface = fragment['interface'] 
                    #  if 'r' in QMin.template['calculation']['manage_children']: interface = 'QMOUT'
                    child_dict[(label,c,C)] = (interface, [], {"logfile": os.path.join(label+"_c"+str(c)+"_C"+str(C), "QM.log"), "logname": label+"_c"+str(c)+"_C"+str(C)}) 
                interface = fragment['embedding_interface'] 
                #  if 'r' in QMin.template['calculation']['manage_children']: interface = 'QMOUT'
                child_dict[(label,'embedding',C)] = (interface, [], {"logfile": os.path.join(label+"_embedding_C"+str(C), "QM.log"),"logname": label+"_embedding_C"+str(C) })
        self.instantiate_children(child_dict)

        # Exctract embedding_kindergarden and make electronic_state instances for embedding_site_state and aufbau_site_states
        estates = {}
        astates = { label: [] for label in QMin.template['fragments'] }
        for label, fragment in QMin.template['fragments'].items():
            # Exctract aufbau_site_states for the fragment
            for st in fragment['aufbau_site_states']:
                for M in range(-st['M']+1,st['M'],2):
                    astates[label].append( electronic_state( Z=st['Z'], S=st['M']-1, M=M, N=st['N'], C={} ))
            for C in self.charges: 
                # Exctract embedding_kindergarden
                self.embedding_kindergarden[(label,C)] = self._kindergarden[(label,'embedding',C)]
                del self._kindergarden[(label,'embedding',C)]
                # Exctract embedding_site_state for the fragment and C
                s = fragment['embedding_site_state'][C]
                s = electronic_state( Z=s['Z'], S=s['M']-1, M=s['M']-1, N=s['N'], C={} )
                estates[(label,C)] = s
                #  QMin.template['fragments'][label]['embedding_site_state'][C] = s

        # Write electronic_state classes to the template dictionary
        for (label,C), s in estates.items():
            QMin.template['fragments'][label]['embedding_site_state'][C] = s
        for label, alist in astates.items():
            QMin.template['fragments'][label]['aufbau_site_states'] = alist

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

        # Mimic setup_mol for each child 
        for (label,C), child in self.embedding_kindergarden.items():    
            atoms = QMin.template['fragments'][label]['atoms']
            #  print(atoms)
            s = QMin.template['fragments'][label]['embedding_site_state'][C]
            states = ' '.join([ '0' for i in range(s.M-1) ])+' '+str(s.N)
            charges = ' '.join([ '0' for i in range(s.M-1) ])+' '+str(C)
            self._setup_children_mol(child, atoms, states, charges)
        for (label,c,C), child in self._kindergarden.items():    
            atoms = QMin.template['fragments'][label]['atoms']
            for charge, site_states in QMin.template['fragments'][label]['site_states'].items():
                if charge == c:
                    states = ' '.join([str(i) for i in site_states])
                    charges = ' '.join([str(c) for i in site_states]) # TODO: HäH?
                    self._setup_children_mol(child, atoms, states, charges)

        # Read children's template and resource files
        for (label,C), child in self.embedding_kindergarden.items():
            child.read_resources( os.path.join( label+'_embedding_C'+str(C), child.name()+'.resources' ) )
            child.read_template( os.path.join( label+'_embedding_C'+str(C), child.name()+'.template' ) )
            scratchdir = os.path.join( QMin.resources['scratchdir'], label+'_embedding_C'+str(C)) 
            mkdir(scratchdir)
            child.QMin.resources['scratchdir'] = scratchdir 
        for (label,c,C), child in self._kindergarden.items():
            child.read_resources( os.path.join( label+'_c'+str(c)+'_C'+str(C), child.name()+'.resources' ) )
            child.read_template( os.path.join( label+'_c'+str(c)+'_C'+str(C), child.name()+'.template' ) )
            scratchdir = os.path.join( QMin.resources['scratchdir'], label+'_c'+str(c)+'_C'+str(C) ) 
            mkdir(scratchdir)
            child.QMin.resources['scratchdir'] = scratchdir 


        # Constructing active_integrals from the template
        active_integrals = {"J": { (0,0): [], (0,1): [], (0,2): [],
                                        (1,0): [], (1,1): [] },
                                  "K": { (0,0): [], (0,1): [], (0,2): [],
                                        (1,0): [], (1,1): [] }
                                  }
        if QMin.template['calculation']['active_integrals'] == 'all':
            for JK in ['J','K']:
                for int_type in [ (0,0), (0,1), (0,2), (1,1) ]:
                    for fpair in itertools.combinations( QMin.template['fragments'], 2 ):
                        active_integrals[JK][int_type].append(fpair)
                for fpair in itertools.combinations( QMin.template['fragments'], 2 ):
                    for spectator in QMin.template['fragments']:
                        ftriple = (fpair[0], fpair[1], spectator)
                        active_integrals[JK][(1,0)].append(ftriple)
        else: 
            for JK, value in QMin.template['calculation']['active_integrals'].items(): 
                if value == 'all':
                    for int_type in [ (0,0), (0,1), (0,2), (1,1) ]:
                        for fpair in itertools.combinations( QMin.template['fragments'], 2 ):
                            active_integrals[JK][int_type].append(fpair)
                        for fpair in itertools.combinations( QMin.template['fragments'], 2 ):
                            for spectator in QMin.template['fragments']:
                                ftriple = (fpair[0], fpair[1], spectator)
                                active_integrals[JK][(1,0)].append(ftriple)
                else:
                    for int_type, multiples in value.items():
                        if multiples == 'all':
                            if int_type == '(1,0)':
                                for fpair in itertools.combinations( QMin.template['fragments'], 2 ):
                                    for spectator in QMin.template['fragments']:
                                        ftriple = (fpair[0],fpair[1],spectator)
                                        active_integrals[JK][(1,0)].append(ftriple)
                            else:
                                for fpair in itertools.combinations( QMin.template['fragments'], 2 ):
                                    active_integrals[JK][make_tuple(int_type)].append(fpair)
                        else:
                            for multiple in multiples:
                                active_integrals[JK][make_tuple(int_type)].append(make_tuple(multiple))
        QMin.template['calculation']['active_integrals'] = active_integrals

        # Build actual excitations and charge_transfer items from the template
        basis = QMin.template['calculation']['excitonic_basis']
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
        CT = {}
        CT[0] = basis['CT'].get(0,False)
        if CT[0]: CT[0] = [ ((),()) ]
        if not CT[0]: CT[0] = []
        del basis['CT'][0] 
        for rank, value in basis['CT'].items():
            CT[rank] = []
            if value == 'all':
                for donors in itertools.combinations_with_replacement(QMin.template['fragments'], rank):
                    for acceptors in itertools.combinations_with_replacement(QMin.template['fragments'], rank): 
                        mutual = [ d == a for d in donors for a in acceptors ]
                        #  print('donors = ', donors)
                        #  print('acceptors = ', acceptors)
                        #  print('mutual = ', mutual)
                        if not any(mutual):
                            CT[rank].append((donors,acceptors))
            else: # ToDo
                for subset in value:
                    CT[rank].append(list(subset))
        basis['CT'] = CT
        #  print('CT = ', basis['CT'])

        # Set inevitable requests to the children
        # Set point-charge request to all embedding children
        for (label,C), child in self.embedding_kindergarden.items():
            s = QMin.template['fragments'][label]['embedding_site_state'][C] 
            child.QMin.requests['multipolar_fit'] = [ (s.S+1, s.N, s.S+1, s.N) ]
            child.QMin.requests['mol'] = True 

        # Set density requests 
        for label in QMin.template['fragments']:
            #  switches = {'tot': False, 'aa': False, 'bb': False, 'ab': False, 'ba': False }
            tot = False
            partial = False

            # See whether tot densities are needed for this fragment
            breaking = False
            int_types = { 'J': [(0,0), (0,1), (0,2)], 'K': [] }
            for JK, types in int_types.items():
                for t in types:
                    pairs = QMin.template['calculation']['active_integrals'][JK][t]
                    frags = [ p for pair in pairs for p in pair ]
                    if label in frags:
                        tot = True
                        breaking = True
                        break
                if breaking: break

            # See whether aa and bb densities are needed for this fragment
            breaking = False
            int_types = { 'J': [(1,0), (1,1)], 'K': [(0,0), (0,1), (0,2), (1,0), (1,1)] }
            for JK, types in int_types.items():
                for t in types:
                    pairs = QMin.template['calculation']['active_integrals'][JK][t]
                    frags = [ p for pair in pairs for p in pair ]
                    if label in frags:
                        partial = True
                        breaking = True
                        break
                if breaking: break

            # Find all children
            children = []
            for key, child in self._kindergarden.items():
                if label in key:
                    children.append(child)

            # Write density requests
            value = False
            if tot and partial:
                value = 'all'
            elif tot:
                value = 'tot'
            elif partial:
                value = 'partial'
            value = 'all'
            if value:
                for child in children:
                    child.QMin.requests['density_matrices'] = [ value ]

            # Set H
            for child in children:
                child.QMin.requests['h'] = True
                child.QMin.requests['mol'] = True 

        # Setup children interfaces
        for (label,C), child in self.embedding_kindergarden.items():
            child.setup_interface()
            child.QMin.resources['pwd'] = os.path.join( os.getcwd(), label+'_embedding_C'+str(C) ) 
            child.QMin.save['savedir'] = os.path.join( QMin.save['savedir'], label+'_embedding_C'+str(C), 'SAVE' ) 
        for (label,c,C), child in self._kindergarden.items():
            child.setup_interface()
            child.QMin.resources['pwd'] = os.path.join( os.getcwd(), label+'_c'+str(c)+'_C'+str(C) ) 
            child.QMin.save['savedir'] = os.path.join( QMin.save['savedir'], label+'_c'+str(c)+'_C'+str(C), 'SAVE' ) 



        return

    def read_requests(self, requests_file: str= "QM.in") -> None:
        super().read_requests(requests_file)

        # Make requests for all children, depending on which requests ECI interface got
        # Inevitable requests (energies and densities) are already given in setup_interface
        QMin = self.QMin
        QMout = self.QMout

        for (label,C), child in self.embedding_kindergarden.items():
            with InDir(QMin.save['savedir']):
                mkdir( label+'_embedding_C'+str(C), force=False)
        for (label,c,C), child in self._kindergarden.items():
            with InDir(QMin.save['savedir']):
                mkdir( label+'_c'+str(c)+'_C'+str(C), force=False )


        # Sets all logical requests
        for request in ['dm']:
            for child in self._kindergarden.values():
                child.QMin.requests[request] = QMin.requests[request] # True/False

        #  Call step_logic and request_logic for each true child  # -> all done in run_children()
        # for child in self._kindergarden.values():
            #  child._step_logic()
            # child._request_logic()
        # for child in self.embedding_kindergarden.values():
            #  child._step_logic()
            # child._request_logic()

        return

    def set_coords(self, xyz: str='QM.in') -> None:
        super().set_coords(xyz)

        QMin = self.QMin
        for C in self.charges_to_do:
            garden = { (key[0],key[1]):interface for key,interface in self._kindergarden.items() if key[2] == C }
            egarden = { key[0]:interface for key,interface in self.embedding_kindergarden.items() if key[1] == C }

            # write coords
            for label, child in egarden.items():
                atoms = QMin.template['fragments'][label]['atoms']
                child.set_coords( QMin.coords['coords'][ atoms ] ) 
            for (label,c), child in garden.items():
                atoms = QMin.template['fragments'][label]['atoms']
                child.set_coords( QMin.coords['coords'][ atoms ] ) 

            # write external pccoords and pccharges
            for label1, child1 in egarden.items():
                if QMin.coords['pccoords'] != None: 
                    child1.set_coords( QMin.coords['pccoords'], pc=True )
                    child1.coords['pccharge'] = QMin.coords['pccharge']
                else:
                    if QMin.template['fragments'][label1]['frozen']:
                        child1.QMin.molecule['point_charges'] = False
            for (label1,c1), child1 in garden.items():
                if QMin.coords['pccoords'] != None: 
                    child1.set_coords( QMin.coords['pccoords'], pc=True )
                    child1.coords['pccharge'] = QMin.coords['pccharge']
        return

    def run(self):
        QMin = self.QMin
         
        DOs = { C:{ label:{} for label in QMin.template['fragments']} for C in self.charges_to_do}
        for C in self.charges_to_do:
            APCs = { label: None for label in QMin.template['fragments'] }
            garden = { (key[0],key[1]):interface for key,interface in self._kindergarden.items() if key[2] == C }
            egarden = { key[0]:interface for key,interface in self.embedding_kindergarden.items() if key[1] == C }
            estates = {label:QMin.template['fragments'][label]['embedding_site_state'][C] for label in QMin.template['fragments']}

            # Check whether site-state calcuations neeed to be done or the site-state data can already be read from the QM.out files of children
            if 'r' in QMin.template['calculation']['manage_children']:
                for label, child in egarden.items():
                    try:
                        charges = child.QMin.template['charge'] 
                        child.QMout = QMout( filepath=os.path.join( child.QMin.resources['pwd'], 'QM.out' ), charges=charges )
                        APCs[label] = child.QMout['multipolar_fit'][(estates[label],estates[label])][:,0]
                    except:
                        APCs[label] = np.zeros(child.QMin.molecule['natom'])
                for (label,c), child in garden.items():
                    charges = child.QMin.template['charge']
                    #  print('label = ', label, ', charge = ', c)
                    child.QMout = QMout( filepath=os.path.join( child.QMin.resources['pwd'], 'QM.out' ), charges=charges )
                    #  print('densities = ', child.QMout['density_matrices'].keys())
                #  exit()
            else:
                # Read guesses for embedding charges
                for label, child in egarden.items():
                    try:
                        charges = child.QMin.template['charge'] 
                        estate = QMin.template['fragments'][label]['embedding_site_state'][C] # This should already be electronic_state instance
                        print(os.path.join( child.QMin.resources['cwd'], 'QM.out'))
                        child.QMout = QMout( filepath=os.path.join( child.QMin.resources['pwd'], 'QM.out' ), charges=charges )
                        APCs[label] = child.QMout['multipolar_fit'][(estates[label],estates[label])][:,0]
                    except: 
                        APCs[label] = np.zeros(child.QMin.molecule["natom"])
                frozen = { label:child for label, child in egarden.items() if QMin.template['fragments'][label]['frozen'] }
                relaxed = { label:child for label, child in egarden.items() if not QMin.template['fragments'][label]['frozen'] }
                EHFjob = self.EHFjobs[C]
                EHFjob = EHF.EHF(nproc=self.QMin.resources['ncpu'],
                                 APCs=APCs,
                                 estates=estates,
                                 frozen=frozen,
                                 relaxed=relaxed,
                                 maxcycle=QMin.template['calculation']['EHF_maxcycle'],
                                 tQ=QMin.template['calculation']['tQ'],
                                 output=os.path.join( QMin.resources['scratchdir'], 'EHF_C'+str(C) ))
                t1 = time.time()
                EHFjob.run(external_coords=QMin.coords['pccoords'], external_charges=QMin.coords['pccharge'])
                t2 = time.time()
                self.log.print(' Time elapsed in EHF.run = '+str(round(t2-t1,3))+' sec.')
                APCs = EHFjob.APCs
                self.EHFjobs[C] = EHFjob

                # Writting final APCs to the actual children
                for (label1,c1), child1 in garden.items():
                    pccoords = []
                    pccharge = []
                    for label2, child2 in egarden.items():
                        if label1 != label2:
                            pccoords = pccoords + [ child2.QMin.coords['coords'] ]
                            pccharge = pccharge + [ APCs[label2] ]
                    if QMin.coords['pccoords'] != None: 
                        pccoords = pccoords + [ QMin.coords['pccoords'] ]
                        pccharge = pccharge + [ QMin.coords['pccharge'] ]
                    pccoords = np.concatenate( pccoords, axis=0 )
                    pccharge = np.concatenate( pccharge )
                    child1.set_coords( pccoords, pc=True)
                    child1.QMin.coords['pccharge'] = pccharge

                # Running actual children
                t1 = time.time()
                self.run_children(self.log, {label:garden[label] for label in QMin.resources['ECI_sitejobs']}, QMin.resources['ncpu'])
                t2 = time.time()
                self.log.print(' Time elapsed in site-state calculations = '+str(round(t2-t1,3))+' sec.')

            # Calculate or read site-specific Dyson orbitals if needed
            # Dyson orbitals cannot be read from children's QM.out because even if read_children = true, CT level is changed
            for label in QMin.template['fragments']:
                dyson_garden = {key[1]:value for key,value in garden.items() if key[0] == label}
                for c1, child1 in dyson_garden.items():
                    for c2, child2 in dyson_garden.items():
                        if c1 ==  c2 - 1:
                            if not 'r' in QMin.template['calculation']['manage_children']:
                                self.log.print(' Calculating the Dyson orbitals of fragment '+label+' between charges '+str(c1)+' and '+str(c2))
                                DOs[C][label][(c1,c2)] = child1.dyson_orbitals_with_other(child2,QMin.resources['scratchdir'],QMin.resources['ncpu'],"64000")
                            else:
                                f = open( os.path.join( QMin.resources['cwd'], label+'_C'+str(C)+'_c'+str(c1)+'_c'+str(c2)+'.dyson'), 'r')
                                lines = f.readlines()
                                DOs[C][label][(c1,c2)] = {}
                                for line in lines:
                                    thes1, thes2, spin, coeffs = line.split('|')
                                    for s1 in child1.states:
                                        if s1.symbol(True,True) == thes1:
                                            break
                                    for s2 in child2.states:
                                        if s2.symbol(True,True) == thes2:
                                            break
                                    DOs[C][label][(c1,c2)][(s1,s2,spin)] = np.array( [ float(x) for x in coeffs.split() ] )
                                f.close()
                            if 'w' in QMin.template['calculation']['manage_children']: 
                                f = open( os.path.join( QMin.resources['cwd'], label+'_C'+str(C)+'_c'+str(c1)+'_c'+str(c2)+'.dyson'), 'w' )
                                for (s1,s2,spin), phi in DOs[C][label][(c1,c2)].items():
                                    f.write(s1.symbol(True,True)+'|'+s2.symbol(True,True)+'|'+spin+'|'+' '.join([ str(round(phi[i],10)) for i in range(len(phi))] )+'\n' )
                                f.close()
                #  involved, active = False, False
                #  for ct, value in QMin.template['calculation']['excitonic_basis']['CT'].items():
                    #  if ct > 0:
                        #  if label in set([ item for pair in value for item in pair ]):
                            #  involved = True
                            #  break
                #  actives = []
                #  for JK in ["J","K"]:
                    #  for int_types in [(1,0),(1,1)]:
                        #  actives += QMin.template['calculation']['active_integrals'][JK][int_types]  
                #  actives = set([ item for subset in actives for item in subset ])
                #  if label in actives: active = True
                #  if active and involved:
                    #  dyson_garden = {key[1]:value for key,value in garden.items() if key[0] == label}
                    #  for c1, child1 in dyson_garden.items():
                        #  for c2, child2 in dyson_garden.items():
                            #  if c1 ==  c2 - 1:
                                #  DOs[C][label][(c1,c2)] = child1.dyson_orbitals_with_other(child2)

                # Calculate dipole-moment matrices of children, should be done in children
                for (label,c), child in garden.items():
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

            if 'w' in QMin.template['calculation']['manage_children']:
                for label, child in egarden.items():
                    if not QMin.template['fragments'][label]['frozen'] and QMin.template['calculation']['EHF_maxcycle'] > 0:
                        child.writeQMout( filename=os.path.join( child.QMin.resources['pwd'],'QM.out' ) )
                for (label,c), child in garden.items():
                    child.writeQMout( filename=os.path.join( child.QMin.resources['pwd'],'QM.out' ) )

            ECIjob = self.ECIjobs[C]

            # Start filling the site-specific data and build fragment instances
            sites = []
            for flabel, fdict in QMin.template['fragments'].items():
                H = {}
                rho = {}
                mu = {}
                grad = {}
                states = {}
                phi = DOs[C][flabel]
                Z = fdict['refcharge'][C]
                mol = garden[(flabel,Z)].QMout['mol'] 
                aufbau_states = fdict['aufbau_site_states']

                for c, nstates in fdict['site_states'].items():
                    child = garden[(flabel,c)]
                    #states[c] = [ s for s in child.states if nstates[s.S] >= s.N ]
                    states[c] = child.states
                    H[c] = child.QMout['h']
                    rho[c] = child.QMout['density_matrices']
                    if 'dm' in QMin.requests:
                        mu[c] = {}
                        for i1, s1 in enumerate(states[c]):
                            for i2, s2 in enumerate(states[c]):
                                if (s1,s2,'tot') in rho[c]:
                                    mu[c][(s1,s2)] = child.QMout['dm'][:,i1,i2].astype(float)
                    if 'grad' in QMin.requests:
                        grad[c] = child.QMout['grad']

                site = ECI.fragment(label=flabel,
                                    Z=Z,
                                    mol=mol,
                                    states=states,
                                    aufbau_states=aufbau_states,
                                    H=H,
                                    rho=rho,
                                    mu=mu,
                                    phi=DOs[C][flabel],
                                    Q=APCs[flabel],
                                    index=len(sites))
                sites.append( site )

            # Make job instance
            properties = [ prop for prop in ['dm'] if QMin.requests[prop]]
            job = ECI.calculation( ncpu=QMin.resources['ncpu'],
                                   charge=C, 
                                   multiplicities=[m+1 for m, nstates in enumerate(QMin.molecule['states']) if nstates > 0],
                                   tO=QMin.template['calculation']['tO'],
                                   eci_level=QMin.template['calculation']['excitonic_basis']['ECI'],
                                   ct_level=QMin.template['calculation']['excitonic_basis']['CT'],
                                   active_integrals=QMin.template['calculation']['active_integrals'], 
                                   ri=QMin.template['calculation']['ri'],
                                   Jauxbasis=QMin.template['calculation']['Jauxbasis'],
                                   Kauxbasis=QMin.template['calculation']['Kauxbasis'],
                                   tS=QMin.template['calculation']['tS'],
                                   tL=QMin.template['calculation']['tL'],
                                   properties=properties
                                  )
            # Initialize ECI instance
            ECIjob = ECI.ECI( job=job, sites=sites, output=os.path.join( QMin.resources['scratchdir'], 'ECI_C'+str(C) ) )
            # Run ECI-CT calculation
            self.log.print(' Calling ECI.run function for charge '+str(C)+', for details check <SCRATCHDIR>/ECI_C'+str(C)+'.log')
            t1 = time.time()
            ECIjob.run()
            t2 = time.time()
            self.log.print(' Time elapsed in ECI.run = '+str(round(t2-t1,3))+' sec.')
            self.ECIjobs[C] = ECIjob
        return

    def getQMout(self):
        QMin = self.QMin
        QMout = self.QMout
        ECIjobs = self.ECIjobs

        states = QMin.molecule["states"]
        nmstates = QMin.molecule["nmstates"]
        natom = QMin.molecule["natom"]
        QMout.allocate(
            states, natom, QMin.molecule["npc"], {r for r in QMin.requests.keys() if QMin.requests[r]}
        )

        for i, label in enumerate(QMin.template['fragments']):
            child = self._kindergarden[(label,list(QMin.template['fragments'][label]['site_states'].keys())[0], list(self.charges_to_do)[0])]
            if i == 0:
                QMout['mol'] = child.QMout['mol']
            else:
                QMout['mol'] = merge_moles( QMout['mol'], child.QMout['mol'] )

        iterator = list(enumerate(itnmstates(QMin.molecule['states'])))
        for request, value in QMin.requests.items():
            if value:
                if request == 'h':
                    for i, s in enumerate(self.states):
                        QMout['h'][i,i] = ECIjobs[s.Z].E[s.S+1][s.N-1].astype(complex)
                    #  for i, (S,N,M) in iterator: 
                        #  C = QMin.template['charge'][S-1] 
                        #  QMout['h'][i,i] = ECIjobs[C].E[S][N-1].astype(complex)
                if request == 'dm':
                    for i1, s1 in enumerate(self.states):
                        for i2, s2 in enumerate(self.states):
                            if s1.Z == s2.Z and s1.S == s2.S:
                                QMout['dm'][:,i1,i2] = ECIjobs[s1.Z].mu[s1.S+1][:,s1.N-1,s2.N-1]
                    #  for i1, (S1,N1,M1) in iterator: 
                        #  for i2, (S2,N2,M2) in iterator:
                            #  if S1 == S2:
                                #  C = QMin.template['charge'][S1-1] 
                                #  QMout['dm'][:,i1,i2] = ECIjobs[C].mu[S1][:,N1-1,N2-1].astype(complex)

        return 

    def create_restart_files(self) -> None:
        """
        Create restart files
        """

    def dyson_orbitals_with_other(self, other):
        pass

    #  def getQMout(self) -> dict[str, np.ndarray]:
        #  pass

    def prepare(self, INFOS: dict, dir_path: str):
        pass

if __name__ == "__main__":
    SHARC_ECI().main()

