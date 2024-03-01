import datetime
from io import TextIOWrapper
from typing import Optional
from qmin import QMinRequests
from qmout import QMout

import numpy as np
import yaml
from SHARC_HYBRID import SHARC_HYBRID

__all__ = ["SHARC_ECI"]

AUTHORS = ""
VERSION = ""
VERSIONDATE = datetime.datetime(2023, 8, 29)
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
                    "EHF": True,
                    "EHF_maxcycle": 20,
                    "tQ": 1e-4,
                    "tO": 0.95,
                    "read_site_states": False,
                    "ri": False,
                    "auxbasis": "def2-svp-jkfit",
                },
            }
        )
        self.QMin.template.types.update({"fragments": dict, "charge": list, "calculation": dict})

        self._calculation_types = {
            "EHF": bool,
            "EHF_maxcycle": int,
            "tQ": (float, int),
            "t0": (float, int),
            "CT_level": list,
            "ECI_level": list,
            "read_site_states": bool,
            "ri": bool,
            "auxbasis": str,
        }

        self.fragmentation: dict = {}


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
        for frag, key in self.QMin.template["fragments"].items():
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

        # Setting fragmentation atribute
        for label, fragment in self.QMin.template['fragments'].items():
            strings = fragment['atoms'].split(',')
            atoms = []
            for group in strings:
                group.split('-')
                if len(group) == 1:
                    atoms.append(group[0])
                elif len(group) == 2:
                    for i in range( int(group[0]), int(group[1]) + 1 ):
                        atoms.append(i)

            self.fragmentation = { label: atoms.copy() }

        # Constructing active_site_pairs dictionary from the template

        # Set full active_site_pairs dictionary
        self.active_site_pairs = {"J": { (0,0): [], (0,1): [], (0,2): [],
                                        (1,0): [], (1,1): [] },
                                  "K": { (0,0): [], (0,1): [], (0,2): [],
                                        (1,0): [], (1,1): [] }
                                  }
        for JK in ['J','K']:
            for int_type in [ (0,0), (0,1), (0,2), (1,0), (1,1) ]:
                for fpair in itertools.combinations( self.QMin.template['fragments'].keys(), 2 ):
                    fpair = set(fpair)
                    self.active_site_pairs[JK][int_type].append(fpair)

        for int_type, JK in self.QMin.template['calculation']['integral_exceptions']:
            if JK['J'] == "all":
                pass

            

        return
        
    def read_resources(self, resources_file: str = "ECI.resources") -> None:
        """
        Parser for ECI resources in yaml format

        resources_file:  Path to template file
        """
        # TODO: validate *_site_state values
        self.log.debug(f"Parsing template file {resources_file}")

        # Open resources_file file and parse yaml
        with open(resources_file, "r", encoding="utf-8") as res_file:
            res_dict = yaml.safe_load(res_file)
            self.log.debug(f"Parsing yaml file:\n{res_dict}")

        self.QMin.resources.update(res_dict)

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

    def setup_interface(self) -> None:
        """
        Load and initialize all child interfaces
        """

        # Instatiate all children
        child_dict = {}
        for C in charges_to_do: # Full-system charge
            for label, fragment in self.QMin.template['fragments']:
                for c in fragment['site_states'].keys(): # Fragment's charge
                    child_dict[(label,c,C)] = child['interface']
                    child_dict[(label,'embedding',C)] = child['embedding_interface']
        self.instantiate_children(child_dict)

        # Exctract embedding_kindergarden
        for C in charges_to_do: 
            for label, fragment in self.QMin.template['fragments']:
                self.embedding_kindergarden[(label,C)] = self.kindergarden[(label,'embedding',C)]
                del self.kindergarden[(label,'embedding',C)]
        return

    def read_requests(self, requests_file: str= "QM.in") -> None:
        super._read_requests(requests_file)
        # Make requests for all children, depending which requests ECI interface got

        # Energies
        # Densities
        for label in self.QMin.template['fragments']:
            for int_type, JK in self.QMin.template['calculation']['integral_exceptions'].items(): 
                Js = { key: value["J"] for key,value in self.QMin.template['calculation']['integral_exceptions'].items() }
                Ks = { key: value["K"] for key,value in self.QMin.template['calculation']['integral_exceptions'].items() }
                
                K = JK['K'] 
                #if K == 'all'
            
        for (label,c,C), child in self.kindergarden.items():
            requests = {'H':None}

    def run(self):
        QMin = self._QMin
        for C in charges_to_do:
            garden = { (key[0],key[1]):interface for key,interface in self.kindergarden.items() if key[2] == C }
            egarden = { key[0]:interface for key,interface in self.embedding_kindergarden.items() if key[1] == C }

            # write coords
            for label, child in egarden.items():
                egarden[label].set_coords( qmin.coords['coords'][ self.fragmentation[label] ] ) 
            for (label,c), child in garden.items():
                child.set_coords( qmin.coords['coords'][ self.fragmentation[label] ] )

            # write pccoords
            for label1, child1 in egarden.items():
                pccoords = [ child2.QMin.coords['coords'] for label2, child2 in egarden.items() if label2 != label1 ] + self.QMin.coords['pccoords'] 
                pccoords = np.concatenate( pccoords, axis=0 )
                child1.set_coords( pccoords, pc=True )
            for (label1,c1), child1 in garden.items():
                pccoords = [ child2.QMin.coords['coords'] for label2, child2 in egarden.items() if label2 != label1 ] + self.QMin.coords['pccoords'] 
                pccoords = np.concatenate( pccoords, axis=0 )
                child1.set_coords( pccoords, pc=True )

            # Read guesses for embedding charges
            APCs = {}
            estates = {}
            for label, child in egarden.items():
                estates[label] = QMin.template['fragments'][label]['embedding_site_state'][C]
                estates[label] = electronic_state( Z=estate['Z'], S=estate['S'], M=estate['S'], N=estate['N'])
                for s in child.states:
                    if s == estates[label]:
                        estates[label] = s
                        break
                try:
                    guess = SHARC_QMout( os.path.join( QMin.save['savedir'],label+'_C'+str(C)+'eQM.out' ) )
                    for s in guess.states:
                        if s == estates[label]: break
                    APCs[label1] = guess.QMout['multipolar_fit'][(s,s,1)]
                except: 
                    APCs[label] = np.zeros(child.QMin.molecule["natom"]) 

            # Run frozen fragments
            frozen = [ label for label in egarden if QMin.template['fragments'][label]['frozen'] ]
            for joblist in QMin.resources['EHF_frozen_sitejobs']: 
                with Pool(processes=len(joblist)) as pool:
                    for job in joblist:
                        pool.apply_async( egarden[job].run )
                    pool.close()
                    pool.join()
            for label, child in frozen_garden.items():
                child.getQMout()
                for s in guess.states:
                    if s == estates[label]: break
                APCs[label] = child.QMout['multipolar_fit'][(s,s,1)]

            # Run EHF cycles while relaxing non-frozen fragments and keeping fixed frozen fragments
            relaxed = [ label for label in egarden if not QMin.template['fragments'][label]['frozen'] ]
            convergence = {}
            dAPCs = {}
            # Main EHF loop
            for cycle in range(QMin.template['calculation']['EHF_maxcycle']):
                # Provide current APCs of all other fragments to a relaxed fragment
                for label1 in relaxed:
                    child1 = egarden[label1]
                    PCs = np.concatenate( [ APCs[label2] for label2 in egarden.keys() if label1 != label2 ], axis=0)
                    PCs = np.concatenate( PCs, QMin.coords['pccharge'])
                    child1.QMin.coords['pccharges'] = PCs
                # Run all relaxed fragments
                for joblist in QMin.resources['EHF_relaxed_sitejobs']:
                    with Pool(processes=len(joblist)) as pool:
                        for job in joblist:
                            pool.apply_async( egarden[job].run )
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

            # Writting final APCs to the "real" children
            for (label1,c1), child1 in garden.items():
                PCs = np.concatenate( [ APCs[label2] for label2 in egarden.keys() if label1 != label2 ], axis=0)
                PCs = np.concatenate( PCs, QMin.coords['pccharge'], axis=0)
                child1.QMin.coords['pccharges'] = PCs
                
        charges_to_do = []
        for i, nst in enumerate( self.QMin.molecule['states']):
            if nst != 0:
                charges_to_do.append( self.QMin.template['charge'][i] )
        charges_to_do = set(charges_to_do)

        ECIjobs = {}
        for c in charges_to_do:
            (label, charge)
            (label, 'embedding')
            embedding_kindergarden = { label:interface for label, interface in self.kindergarden.items() if 'embedding' in label }
            for (label, charge), interface in self.kindergarden.items():
                if charge == 'embedding':
                    pass

            # Obtaining embedding charges
            embedding_interface = factory(QMin.template['embedding_interface'][c])
            # Do EHF if needed
            if QMin.template['EHF_maxcycle'] > 0:

                EHF_kindergarden = {}
                for fname, fdict in QMin.template['fragments'].items():
                    refst = fdict['ref_site_state'][c]
                    Z, M, N = refst['Z'], refst['M'], refst['N']
                    QMin.molecule
                    QMin.template
                    QMin.resources
                    # EHF_kindergarden[fname] = kindergarden[ (fname, Z ) ].__copy__(states=[0 f...])

                    mol = EHF_kindergarden[fname].QMin.molecule
                    mol['states'] = [ 0 for i in range(M) ]
                    mol['states'][-1] = N + 1
                    mol['nstates'] = mol['states'][-1]
                    mol['nmstates'] = M*mol['states']

                    requests = EHF_kindergarden[fname].QMin.requests
                    requests = QMinRequests()
                    requests['multipolar_fit'] = [ ( M, N+1, M, N+1 ) ]
                    EHF_kindergarden[fname]._request_logic()

                deltaAPCs = 2.*QMin.template['tQ']*np.ones( QMin['natom'] )
                for cycle in range( QMin.template['EHF_maxcycle']):
                    if np.all( np.abs( deltaAPCs ) < QMin.template['tQ'] ):
                        print(' Convergence in EHF for charge '+str(c)+' reached after '+str(cycle+1)+' cycles.')
                        break
                    oldAPcs = newAPCs.copy()
                    # Set APCs to old ESPs
                    for name1, child1 in EHF_kindergarden.items():
                        APCs = []
                        for name2, child2 in EHF_kindergarden.items():
                            if name1 != name2:
                                APCs = APCs + child2.QMout.multipolar_fit
                        PCs = QMin.coord['pccharge'] + APCs
                        child1.QMin.coords['pccharge'] = PCs


                    #Read new ESP charges
                    newAPCs = []
                    for name, child in EHF_kindergarden.items():
                        child.getQMout()
                        newAPCs += child.QMout.multipolar_fit
                    deltaAPCs = newAPCs - oldAPCs
                QMin.template['APCs'] = newAPCs
                #for (name,charge) in


            # Do site-state calculations for ECI if needed
            if QMin.template['calculate_site_states']:
                for joblist in QMin.resources['ECI_sitejobs']:
                    with Pool(processes=len(joblist)) as pool:
                        for job in joblist:
                            with InDir(self.kinderdirs[job]):
                                pool.apply_async( kindergarden[job].run )
                        pool.close()
                        pool.join()

            # In any case, read site-state data (E, dH, rho, mu)
            for (name, child) in kindergarden.items():
                child.getQMout()
                #  with InDir(self.kinderdirs[job]):
                    #  child.QMout = QMout(filepath="QM.out")


            #Construct the Dyson orbitals
            if any( QMin.template['ct_level'] > 0 ):
                for flabel, fdict in self.QMin.template['fragments'].items():
                    for c1 in fdict['site_states']:
                        for c2 in fdict['site_states']:
                            if c1 == c2 - 1: # E.g. c1 = 0 and c2 = +1
                                for m1 in kindergarden[(flabel,c1)].QMin.mults:
                                    for m2 in kindergarden[(flabel,c2)].QMin.mults:
                                        file = open('dyson.inp','w')
                                        file.write('a_mo='+kindergarden[(flabel,c1)].savedir+'/mos.'+str(m1)+'.'+str(c1)+'\n')
                                        file.write('a_det='+kindergarden[(flabel,c1)].savedir+'/dets.'+str(m1)+'.'+str(c1)+'\n')
                                        file.write('b_mo='+kindergarden[(flabel,c2)].savedir+'/mos.'+str(m2)+'.'+str(c2)+'\n')
                                        file.write('b_det='+kindergarden[(flabel,c2)].savedir+'/dets.'+str(m2)+'.'+str(c2)+'\n')
                                        file.write('same_aos=.true.\n')
                                        file.write('ao_read=-1\n')
                                        file.write('moprint=1\n')
                                        file.close()
                                        os.system(QMin.resources['wfoverlap']+' -m '+QMin.resources['memory']+' -f dyson.inp')


            # Initialize ECI instance
            ECIjobs[c] = ECI( calc=QMin.template['calculation'] )

            # This that can be done before knowing desired mult/charge
            for flabel, fdict in QMin.template['fragments'].items():

                rho = {}
                E = {}
                dH = {}
                grad = {}
                nacdr = {}

                refZ = fdict['ref_site_state'][c]['Z']
                sym = kindergarden[(flabel,refZ)].QMin.molecule['elements']
                geom = kindergarden[(flabel,refZ)].QMin.coords['coords']
                basis = kindergarden[(flabel,refZ)].QMout['basis']
                for charge, nstates in fdict['site_states'].items():
                    child = kindergarden[flabel,charge]

                    # Needed for sure
                    H[charge] = child.QMout['H']
                    rho[charge] = child.QMout['density_matrices']

                    # Potentially needed
                    dH[charge] = np.zeros(( child.QMout.nmstates, child.QMout.nmstates, child.QMout.natom, 3) )
                    if 'grad' in QMin.requests:
                        grad[charge] = child.QMout['grad']
                    if 'nacdr' in QM.requests:
                        nacdr[charge] = child.QMout['nacdr']

                ECIjobs[c].load_fragment( flabel=flabel,
                                          fdict=fdict,
                                          refstate=fdict['ref_site_state'][c],
                                          geom=geom,
                                          basis=basis,
                                          H=H,
                                          grad=grad,
                                          nacdr=nacdr,
                                          dyson=dyson,
                                          mu=mu
                                     )

            ECIjobs[c].configure()                              # Configure ECI object after all sites are loaded and entire calculation setup is known
            ECIjobs[c].construct_ECI_basis( c )                 # Building ECI-CT basis
            ECIjobs[c].construct_ECI_Hamiltonain()              # Construct entire ECI-CT Hamiltonian for all multiplicities
            ECIjobs[c].calcuate_eigenstates()                   # Diagonalize Hamiltonian matrix for each multiplicity
            ECIjobs[c].calculate_properties( QMin.requests )    # Calculate all properties requested by master

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
