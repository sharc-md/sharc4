#!/usr/bin/env python3

import os
import datetime
from io import TextIOWrapper
from typing import Optional
from qmin import QMinRequests
from qmout import QMout
from constants import ATOMCHARGE

import itertools
import numpy as np
from multiprocessing import Pool
import yaml
from SHARC_HYBRID import SHARC_HYBRID
from utils import InDir, electronic_state
#import ECI

__all__ = ["SHARC_ECI"]

AUTHORS = "Tomislav Piteša & Sascha Mausenberger"
VERSION = "1.0"
VERSIONDATE = datetime.datetime(2024, 2, 20)
NAME = "ECI"
DESCRIPTION = ""

CHANGELOGSTRING = """
"""

all_features = set(  # TODO: Depends on child
    [
        "h",
        "dm",
        "soc",
        "theodore",
        "grad",
        "ion",
        "overlap",
        "phases",
        "molden",
        # raw data request
        "basis_set",
        "wave_functions",
        "density_matrices",
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
                "fragments": None,
                "charge": None,
                "calculation": {
                    "EHF_maxcycle": 20,
                    "tQ": 1e-4,
                    "tO": 0.95,
                    "read_children": False,
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
            "excitonic_basis": dict,
            "read_children": bool,
            "active_integrals": (dict,str)
        }

        self.fragmentation: dict = {}
        self.embedding_kindergarden: dict = {}
        self.embedding_states: dict = {}


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

    def read_template(self, template_file: str = "ECI_template.yaml") -> None:
        """
        Parser for ECI template in yaml format

        template_file:  Path to template file
        """
        #  super().read_template(template_file)

        self._read_template = True
        # TODO: validate *_site_state values
        self.log.debug(f"Parsing template file {template_file}")

        # Open template file and parse yaml
        with open(template_file, "r", encoding="utf-8") as tmpl_file:
            tmpl_dict = yaml.safe_load(tmpl_file)
            self.log.debug(f"Parsing yaml file:\n{tmpl_dict}")

        self.QMin.template.update(tmpl_dict)

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
        for C in self.QMin.template['charge']:
            if not C in self.charges:
                self.log.error(
                        f"Requested full-system charge {C} cannot be calculated because refcharge and embedding_site_state of any fragmnet are not define for it!"
                        )
                raise ValueError()
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

    def _setup_children_mol(self,child,atoms,states) -> None:
        QMin = self.QMin
        #  child.QMin.save['step'] = QMin.save['step']
        child.QMin.molecule['natom'] = len(atoms)
        child.QMin.molecule['elements'] = [ QMin.molecule['elements'][a] for a in atoms ]
        child.QMin.molecule["Atomcharge"] = sum(map(lambda x: ATOMCHARGE[x], child.QMin.molecule["elements"]))
        child.QMin.molecule['frozcore'] = 0
        states_dict = self.parseStates(states)
        if len(states_dict["states"]) < 1:
            self.log.error("Number of states must be > 0!")
            raise ValueError()
        child.QMin.maps["statemap"] = states_dict["statemap"]
        child.QMin.molecule["nstates"] = states_dict["nstates"]
        child.QMin.molecule["nmstates"] = states_dict["nmstates"]
        child.QMin.molecule["states"] = states_dict["states"]
        child.QMin.molecule['unit'] = QMin.molecule['unit']
        child.QMin.molecule['factor'] = QMin.molecule['factor']
        #  child.QMin.save['savedir'] = os.path.join( QMin.save['savedir'], 'SAVEDIR')
        child._setup_mol = True
        return

    def setup_interface(self) -> None:
        """
        Load and initialize all child interfaces
        """
        QMin = self.QMin

        # Instatiate all children
        child_dict = {}
        for C in self.charges: # Full-system charge
            for label, fragment in QMin.template['fragments'].items():
                for c in fragment['site_states'].keys(): # Fragment's charge
                    child_dict[(label,c,C)] = fragment['interface']
                child_dict[(label,'embedding',C)] = fragment['embedding_interface']
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
                group.split('-')
                if len(group) == 1:
                    atoms[label].append(int(group[0]))
                elif len(group) == 2:
                    for i in range( int(group[0]), int(group[1]) + 1 ):
                        atoms[label].append(i)
        for label, a in atoms.items():
            QMin.template['fragments'][label]['atoms'] = a

        # Mimic setup_mol for each child 
        for (label,C), child in self.embedding_kindergarden.items():    
            atoms = QMin.template['fragments'][label]['atoms']
            #  print(atoms)
            s = QMin.template['fragments'][label]['embedding_site_state'][C]
            states = ' '.join([ '0' for i in range(s.M-1) ])+' '+str(s.N)
            self._setup_children_mol(child, atoms, states)
        for (label,c,C), child in self._kindergarden.items():    
            atoms = QMin.template['fragments'][label]['atoms']
            for charge, site_states in QMin.template['fragments'][label]['site_states'].items():
                if charge == c:
                    states = ' '.join([str(i) for i in site_states])
                    self._setup_children_mol(child, atoms, states)

        # Read children's template and resource files
        for (label,C), child in self.embedding_kindergarden.items():
            child.read_resources( os.path.join( label+'_embedding_C'+str(C), child.name()+'.resources' ) )
            child.read_template( os.path.join( label+'_embedding_C'+str(C), child.name()+'.template' ) )
            child.QMin.resources['scratchdir'] = os.path.join( label+'_embedding_C'+str(C), child.QMin.resources['scratchdir']) 
        for (label,c,C), child in self._kindergarden.items():
            child.read_resources( os.path.join( label+'_c'+str(c)+'_C'+str(C), child.name()+'.resources' ) )
            child.read_template( os.path.join( label+'_c'+str(c)+'_C'+str(C), child.name()+'.template' ) )


        # Constructing active_integrals from the template
        active_integrals = {"J": { (0,0): [], (0,1): [], (0,2): [],
                                        (1,0): [], (1,1): [] },
                                  "K": { (0,0): [], (0,1): [], (0,2): [],
                                        (1,0): [], (1,1): [] }
                                  }
        if QMin.template['calculation']['active_integrals'] == 'all':
            for JK in ['J','K']:
                for int_type in [ (0,0), (0,1), (0,2), (1,0), (1,1) ]:
                    for fpair in itertools.combinations( QMin.template['fragments'], 2 ):
                        fpair = set(fpair)
                        active_integrals[JK][int_type].append(fpair)
        else: 
            for JK, value in QMin.template['calculation']['active_integrals'].items(): 
                if value == 'all':
                    for int_type in [ (0,0), (0,1), (0,2), (1,0), (1,1) ]:
                        for fpair in itertools.combinations( QMin.template['fragments'], 2 ):
                            fpair = set(fpair)
                            active_integrals[JK][int_type].append(fpair)
                else:
                    for int_type, pairs in value.items():
                        if pairs == 'all':
                            for fpair in itertools.combinations( QMin.template['fragments'], 2 ):
                                fpair = set(fpair)
                                active_integrals[JK][int_type].append(fpair)
                        else:
                            for fpair in pairs:
                                active_integrals[JK][int_type].append(set(fpair))
        QMin.template['calculation']['active_integrals'] = active_integrals

        # Build actual excitations and charge_transfer items from the template
        basis = QMin.template['calculation']['excitonic_basis']
        # excitations
        ECI = {}
        ECI[0] = basis['ECI'].get(0,False)
        del basis['ECI'][0] 
        for rank, value in basis['ECI']:
            if rank == 0:
                ECI[0] = True
            else:
                ECI[rank] = []
                if value == 'all':
                    for subset in itertools.combinations( QMin.template['fragments'], rank ):
                        ECI[rank].append(set(subset))
                else:
                    for subset in value:
                        ECI[rank].append(set(subset))
        basis['ECI'] = ECI
        # charge-transfers
        CT = {}
        CT[0] = basis['CT'].get(0,False)
        del basis['CT'][0] 
        for rank, value in basis['CT'].items():
            if rank == 0:
                CT[0] = True
            else:
                CT[rank] = []
                if value == 'all':
                    for subset in itertools.combinations( QMin.template['fragments'], rank ):
                        CT[rank].append(set(subset))
                else:
                    for subset in value:
                        CT[rank].append(set(subset))
        basis['CT'] = CT

        # Set inevitable requests to the children
        # Set point-charge request to all embedding children
        for (label,C), child in self.embedding_kindergarden.items():
            s = QMin.template['fragments'][label]['embedding_site_state'][C] 
            child.QMin.requests['multipolar_fit'] = [ (s.S+1, s.N, s.S+1, s.N) ]
            child._request_logic()

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
            if value:
                for child in children:
                    child.QMin.requests['density_matrices'] = [value]

            # Set H
            for child in children:
                child.QMin.requests['H'] = True

        # Setup children interfaces
        for child in self.embedding_kindergarden.values():
            child.setup_interface()
        for child in self._kindergarden.values():
            child.setup_interface()



        return

    def read_requests(self, requests_file: str= "QM.in") -> None:
        super().read_requests(requests_file)

        # Make requests for all children, depending on which requests ECI interface got
        # Inevitable requests (energies and densities) are already given in setup_interface
        QMin = self.QMin
        QMout = self.QMout

        for request in ['dm']:
            if request in self.QMin.requests:
                for child in self._kindergarden.values():
                    child.QMin.requests[request] = True

        return

    def set_coords(self, xyz: str='QM.in') -> None:
        super().set_coords(xyz)

        QMin = self.QMin
        charges_to_do = [ c for i, c in enumerate(QMin.template['charge']) if QMin.molecule['states'][i] > 0 ]
        for C in charges_to_do:
            garden = { (key[0],key[1]):interface for key,interface in self._kindergarden.items() if key[2] == C }
            egarden = { key[0]:interface for key,interface in self.embedding_kindergarden.items() if key[1] == C }

            # write coords
            for label, child in egarden.items():
                atoms = QMin.template['fragments'][label]['atoms']
                child.set_coords( QMin.coords['coords'][ atoms ] ) 
            for (label,c), child in garden.items():
                atoms = QMin.template['fragments'][label]['atoms']
                child.set_coords( QMin.coords['coords'][ atoms ] ) 

            # write pccoords
            for label1, child1 in egarden.items():
                pccoords = [ child2.QMin.coords['coords'] for label2, child2 in egarden.items() if label2 != label1 ]
                if QMin.coords['pccoords'] != None: 
                    pccoords += QMin.coords['pccoords']
                    pccoords = np.concatenate( pccoords, axis=0 )
                child1.set_coords( pccoords, pc=True )
            for (label1,c1), child1 in garden.items():
                pccoords = [ child2.QMin.coords['coords'] for label2, child2 in egarden.items() if label2 != label1 ]
                if QMin.coords['pccoords'] != None: 
                    pccoords += QMin.coords['pccoords']
                    pccoords = np.concatenate( pccoords, axis=0 )
                child1.set_coords( pccoords, pc=True )
        return

    def run(self):
        QMin = self.QMin
        charges_to_do = [ c for i, c in enumerate(QMin.template['charge']) if QMin.molecule['states'][i] > 0 ]
        for C in charges_to_do:
            APCs = { label: None for label in QMin.template['fragments'] }
            garden = { (key[0],key[1]):interface for key,interface in self._kindergarden.items() if key[2] == C }
            egarden = { key[0]:interface for key,interface in self.embedding_kindergarden.items() if key[1] == C }
            estates = {label:QMin.template['fragments'][label]['embedding_site_state'][C] for label in QMin.template['fragments']}

            # Read guesses for embedding charges
            for label, child in egarden.items():
                estate = QMin.template['fragments'][label]['embedding_site_state'][C] # This should already be electronic_state instance
                try:
                    guess_child = SHARC_QMout( os.path.join( QMin.save['savedir'],label+'_C'+str(C)+'eQM.out' ) )
                    APCs[label] = guess_child.QMout['multipolar_fit'][(estates[label],estates[label])]
                except: 
                    APCs[label] = np.zeros(child.QMin.molecule["natom"])

            # Check whether site-state calcuations neeed to be done or the site-state data can already be read from the QM.out files of children
            if QMin.template['calculation']['read_children']:
                for (label,c), child in garden.items():
                    child = SHARC_QMout( os.path.join( child.QMin.control['workdir'], 'QM.out' ) )
            else:
                # Run frozen fragments
                frozen = [ label for label in egarden if QMin.template['fragments'][label]['frozen'] ]
                for joblist in QMin.resources['EHF_frozen_sitejobs']: 
                    print('Here I am 1')
                    #  for label in joblist:
                        #  egarden[label].run()
                    errors = {}
                    with Pool(processes=len(joblist)) as pool:
                        print('Here I am 2')
                        print(joblist)
                        for label in joblist:
                            #  print(egarden[label].QMin)
                            with InDir(label+'_embedding_C'+str(C)):
                                errors[label] = pool.apply_async( egarden[label].run )
                        pool.close()
                        pool.join()
                        results = {v.get() for v in errors.values()}
                for label in frozen:
                    child = egarden[label]
                    print(os.getcwd())
                    child.getQMout()
                    APCs[label] = child.QMout['multipolar_fit'][(estates[label], estates[label])]

                # Run EHF cycles while relaxing non-frozen fragments and keeping fixed frozen fragments
                relaxed = [ label for label in egarden if not QMin.template['fragments'][label]['frozen'] ]
                if len(relaxed) > 0:
                    convergence = {}
                    dAPCs = {}
                    # Main EHF loop
                    for cycle in range(QMin.template['calculation']['EHF_maxcycle']):
                        # Provide current APCs of all other fragments to a relaxed fragment
                        for label1 in relaxed:
                            child1 = egarden[label1]
                            PCs = np.concatenate( [ APCs[label2] for label2 in egarden.keys() if label1 != label2 ], axis=0)
                            if QMin.coords['pccharge'] != None: PCs = np.concatenate( (PCs, QMin.coords['pccharge']) )
                            child1.QMin.coords['pccharges'] = PCs
                        # Run all relaxed fragments
                        for joblist in QMin.resources['EHF_relaxed_sitejobs']:
                            with Pool(processes=len(joblist)) as pool:
                                for label in joblist:
                                    pool.apply_async( egarden[label].run )
                                pool.close()
                                pool.join()
                        # Get new APCs of all relaxed fragments and check convergence
                        for label in relaxed:
                            child = egarden[label]
                            child.getQMout()
                            newAPCs = child.QMout['multipolar_fit'][(estates[label],estates[label],1)] 
                            dAPCs[label] = newAPCs - APCs[label]
                            APCs[label] = newAPCs 
                            convergence[label] = np.nonzero(np.abs(dAPCs[label]) < QMin.template['calculation']['tQ'])[0]
                            for i in range(child.QMin.molecule['natom']):
                                converged = 'NO'
                                if convergence[label][i]: converged = 'YES'
                                self.log.print('      '.join( [ f"{APCs[label][i]: 8.5f}", f"{dAPCs[label][i]: 8.5f}", converged ] ))
                        if all( [ all(convergence[label]) for label in relaxed ] ):
                            self.log.print(' EHF convergence reached in '+str(cycle+1)+' cycles!')
                            break
                    if not all( [ all(convergence[label]) for label in relaxed ] ):
                        self.log.warning(' Maximum number in EHF is exceeded but some charges are still not converged! Proceeding nevertheless...')

                # Writting final APCs to the actual children
                for (label1,c1), child1 in garden.items():
                    PCs = np.concatenate( [ APCs[label2] for label2 in egarden.keys() if label1 != label2 ], axis=0)
                    PCs = np.concatenate( (PCs, QMin.coords['pccharge']), axis=0)
                    child1.QMin.coords['pccharges'] = PCs

                # Running actual children
                for joblist in QMin.resources['ECI_sitejobs']:
                    with Pool(processes=len(joblist)) as pool:
                        for job in joblist:
                            with InDir(garden[job].QMin.control['workdir']):
                                pool.apply_async( garden[job].run )
                        pool.close()
                        pool.join()

                # Reading QMout of actual children
                for child in garden.values():
                    child.getQMout()

            # Calculate site-specific Dyson orbitals if needed
            # Dyson orbitals cannot be read from children's QM.out because even if read_children = true, CT level is changed
            for label in QMin.template['fragments']:
                involved, active = False, False
                for ct, value in QMin.template['calculation']['excitonic_basis']['CT'].items():
                    if ct > 0:
                        if label in set([ item for item in pair for pairs in value ] ):
                            involved = True
                            break
                actives = []
                for JK in ["J","K"]:
                    for int_types in [(1,0),(1,1)]:
                        actives += QMin.template['calculation']['active_integrals'][JK][int_types]  
                actives = set([ item for item in pair for pair in pairs ])
                if label in actives: active = True
                if active and involved:
                    dyson_garden = { key[1]:value for key,value in garden.items() if key[0] == label}
                    for c1, child1 in dyson_garden.items():
                        for c2, child2 in dyson_garden.items():
                            if c1 ==  c2 - 1:
                                DOs[C][label][(c1,c2)] = child1.dyson_orbitals_with_other(child2)

            ECIjob = self.ECIjobs[C]

            # Start filling the site-specific data and build fragment instances
            sites = []
            for flabel, fdict in QMin.template['fragments'].items():
                H = {}
                rho = {}
                mu = {}
                grad = {}
                phi = DOs[C][flabel]
                Z = fdict['refcharge'][C]
                mol = garden[(flabel,refZ)].QMout['mol'] 
                aufbau_states = fdict['aufbau_site_states']

                states = []
                for c in fdict['site_states']:
                    child = garden[(flabel,c)]
                    states += child.states
                    H[c] = child.QMout['H']
                    rho[c] = child.QMout['density_matrices']
                    if 'dm' in QMin.requests:
                        mu[c] = child.QMout['dm']
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
                                    phi=DOs[flabel],
                                    Q=APCs[(label,C)])
                sites.append( site )

            # Make job instance
            job = ECI.calculation( ncpu=QMin.resources['ncpu'],
                                   charge=C, 
                                   multiplicities=np.nonzero(QMin.molecule['states'] > 0)[0],
                                   tO=QMin.template['calculation']['tO'],
                                   eci_level=QMin.template['calculation']['excitonic_basis']['ECI'],
                                   ct_level=QMin.template['calculation']['excitonic_basis']['CT'],
                                   active_integrals=QMin.template['calculation']['active_integrals'], 
                                   properties=QMin.requests.keys()
                                  )
            # Initialize ECI instance
            ECIjob = ECI.ECI( job=job, sites=sites )
            # Run ECI-CT calculation
            ECIjob.run()
        return

    def create_restart_files(self) -> None:
        """
        Create restart files
        """

    def dyson_orbitals_with_other(self, other):
        pass

    def getQMout(self) -> dict[str, np.ndarray]:
        pass

    def prepare(self, INFOS: dict, dir_path: str):
        pass

if __name__ == "__main__":
    SHARC_ECI().main()

