import datetime
from io import TextIOWrapper
from typing import Optional
from qmin import QMinRequests

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
                    "t0": 0.95,
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

    def _check_zmn(self, zmn_dict: dict[str, int]) -> bool:
        """
        Check if dictionary contains Z, M, N keys and validate if values are int
        """
        if not isinstance(zmn_dict, dict) or zmn_dict.keys() != {"Z", "M", "N"}:
            return False
        if not all(isinstance(val, int) for val in zmn_dict.values()):
            return False
        return True

    def read_resources(self, resources_file: str = "ECI.resources") -> None:
        pass

    def setup_interface(self) -> None:
        """
        Load and initialize all child interfaces
        """

    def run(self):
        """QMin = self._QMin
        kindergarden = self._kindergarden

        # Set coords and pccoords for all children
        for ( fname1, charge1 ), child1 in kindergarden.items():
            child1.QMin.coords['coords'] = [ QMin.coords['coords'][i].copy() for i in QMin.table['atomIDs'][fname1] ]
            done = []
            APCcoords = []
            for (fname2, charge2), child2 in kindergarden.items():
                if fname2 != fname1 and not fname2 in done:
                    APCcoords += [ QMin.coords['coords'][i].copy() for i in QMin.table['atomIDs'][fname2] ]
                    done.append( fname2 )
            child1.QMin.coords['pccoords'] = QM.coords['pccoords'] + APCcoords

        charges_to_do = []
        for i, nst in enumerate( self.QMin.molecule['states']):
            if nst != 0:
                charges_to_do.append( self.QMin.template['charge'][i] )
        charges_to_do = set(charges_to_do)

        ECIjobs = {}
        for c in charges_to_do:

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
                    EHF_kindergarden[fname] = kindergarden[ (fname, Z ) ].__copy__(states=[0 f...])

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

                    for joblist in QMin.resources['EHF_sitejobs']:
                        with Pool(processes=len(joblist)) as pool:
                            for job in joblist:
                                pool.apply_async( EHF_kindergarden[job].run() )
                            pool.close()
                            pool.join()

                    #Read new ESP charges
                    newAPCs = []
                    for name, child in EHF_kindergarden.items():
                        child.getQMout()
                        newAPCs += child.QMout.multipolar_fit
                    deltaAPCs = newAPCs - oldAPCs
                QMin.template['APCs'] = newAPCs
                for (name,charge) in


            # Do site-state calculations for ECI if needed
            if QMin.template['calculate_site_states']:
                for joblist in QMin.resources['ECI_sitejobs']:
                    with Pool(processes=len(joblist)) as pool:
                        for job in joblist:
                            with InDir(self.kinderdirs[job]):
                                pool.apply_async( kindergarden[job].run() )
                        pool.close()
                        pool.join()

            # In any case, read site-state data (E, dH, rho, mu)
            for (name, child) in kindergarden.items():
                child.getQMout()

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
                                        os.system(QMin.resources['wfoverlap']+' -m '+QMin.resources['memory']+' -f dyson.inp'"


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
                                          nacdr=nacdr
                                          dyson=dyson,
                                          mu=mu
                                     )

            ECIjobs[c].configure()                              # Configure ECI object after all sites are loaded and entire calculation setup is known
            ECIjobs[c].construct_ECI_basis( c )                 # Building ECI-CT basis
            ECIjobs[c].construct_ECI_Hamiltonain()              # Construct entire ECI-CT Hamiltonian for all multiplicities
            ECIjobs[c].calcuate_eigenstates()                   # Diagonalize Hamiltonian matrix for each multiplicity
            ECIjobs[c].calculate_properties( QMin.requests )    # Calculate all properties requested by master"""

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
