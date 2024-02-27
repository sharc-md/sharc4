import datetime
import math
import os
import re
import subprocess as sp
import time
from abc import abstractmethod
from datetime import date
from io import TextIOWrapper
from textwrap import dedent
from multiprocessing import Pool
from typing import Optional
from itertools import starmap

import numpy as np
import sympy
from sympy.physics.wigner import wigner_3j
from qmin import QMin
from SHARC_INTERFACE import SHARC_INTERFACE
from utils import containsstring, readfile, safe_cast, link, writefile, shorten_DIR, mkdir, itmult, convert_list, is_exec
from constants import ATOMIC_RADII, MK_RADII, IToMult
from resp import Resp
from asa_grid import GRIDS
import wf2rho

all_features = {
    "h",
    "soc",
    "dm",
    "grad",
    "nacdr",
    "overlap",
    "phases",
    "ion",
    "dmdr",
    "socdr",
    "multipolar_fit",
    "theodore",
    "point_charges",
    # raw data request
    "basis_set",
    "wave_functions",
    "density_matrices",
}


class SHARC_ABINITIO(SHARC_INTERFACE):
    """
    Abstract base class for ab-initio interfaces
    """

    _theodore_settings = {}
    _density_calculation_methods = ("from_gs2es", "from_determinants")

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Add ab-initio specific keywords to template
        self.QMin.template.update({"charge": None, "paddingstates": None, "density_calculation_methods" : ["from_gs2es", "from_determinants"], "tCI": 1e-7 })

        self.QMin.template.types.update({"charge": list, "paddingstates": list, "density_calculation_methods": list, "tCI" : float})
        # Add ab-initio specific keywords to resources
        self.QMin.resources["delay"] = 0.0

        self.QMin.resources.types["delay"] = float

        # Add list of slots per pool to control
        self.QMin.control["nslots_pool"] = []

        self.QMin.control.types["nsplots_pool"] = list

        self.QMin.resources.update(
            {
                "theodir": None,
                "theodore_prop": [],
                "theodore_fragment": [],
                "wfoverlap": "wfoverlap.x",
                "wfthres": 0.998,
                "resp_shells": [],  # default calculated from other values = [1.4, 1.6, 1.8, 2.0]
                "resp_vdw_radii_symbol": {},
                "resp_vdw_radii": [],
                "resp_betas": [0.0005, 0.0015, 0.003],
                "resp_layers": 4,
                "resp_first_layer": 1.4,
                "resp_density": 10.0,
                "resp_fit_order": 2,
                "resp_mk_radii": True,  # use radii for original Merz-Kollmann-Singh scheme for HCNOSP
                "resp_grid": "lebedev",
            }
        )

        self.QMin.resources.types.update(
            {
                "theodir": str,
                "theodore_prop": list,
                "theodore_fragment": list,
                "wfoverlap": str,
                "wfthres": float,
                "resp_shells": list,  # default calculated from other values = [1.4, 1.6, 1.8, 2.0]
                "resp_vdw_radii_symbol": dict,
                "resp_vdw_radii": list,
                "resp_betas": list,
                "resp_layers": int,
                "resp_first_layer": float,
                "resp_density": float,
                "resp_fit_order": int,
                "resp_mk_radii": bool,  # use radii for original Merz-Kollmann-Singh scheme for HCNOSP
                "resp_grid": str,
            }
        )

    @staticmethod
    @abstractmethod
    def authors() -> str:
        return "Severin Polonius, Sebastian Mai"

    @staticmethod
    @abstractmethod
    def version() -> str:
        return "3.0"

    @staticmethod
    @abstractmethod
    def versiondate() -> date:
        return date(2021, 7, 15)

    @staticmethod
    @abstractmethod
    def name() -> str:
        return "base"

    @staticmethod
    @abstractmethod
    def description() -> str:
        return "Abstract base class for SHARC interfaces."

    @staticmethod
    @abstractmethod
    def changelogstring() -> str:
        return "This is the changelog string"

    @staticmethod
    @abstractmethod
    def about() -> str:
        return "Name and description of the interface"

    @abstractmethod
    def get_features(self, KEYSTROKES: Optional[TextIOWrapper] = None) -> set[str]:
        """return availble features

        ---
        Parameters:
        KEYSTROKES: object as returned by open() to be used with question()
        """
        return all_features

    @abstractmethod
    def get_infos(self, INFOS: dict, KEYSTROKES: Optional[TextIOWrapper] = None) -> dict:
        """prepare INFOS obj

        ---
        Parameters:
        INFOS: dictionary with all previously collected infos during setup
        KEYSTROKES: object as returned by open() to be used with question()
        """
        return INFOS

    @abstractmethod
    def prepare(self, INFOS: dict, dir_path: str):
        "setup the calculation in directory 'dir'"
        return

    def print_qmin(self) -> None:
        self.log.info(f"{self.QMin}")

    @abstractmethod
    def execute_from_qmin(self, workdir: str, qmin: QMin) -> tuple[int, datetime.timedelta]:
        """
        Erster Schritt, setup_workdir ( inputfiles schreiben, orbital guesses kopieren, xyz, pc)
        Programm aufrufen (z.b. run_program)
        checkstatus(), check im workdir ob rechnung erfolgreich
            if success: pass
            if not: try again or return error
        postprocessing of workdir files (z.b molden file erzeugen, stripping)

        returns (exit code, runtime)
        """

    @abstractmethod
    def read_template(self, template_file: str, kw_whitelist: Optional[list[str]] = None) -> None:
        super().read_template(template_file, kw_whitelist)

        # Check if charge in template and autoexpand if needed
        if self.QMin.template["charge"]:
            if len(self.QMin.template["charge"]) == 1:
                charge = int(self.QMin.template["charge"][0])
                if (self.QMin.molecule["Atomcharge"] + charge) % 2 == 1 and len(self.QMin.molecule["states"]) > 1:
                    self.log.info("HINT: Charge shifted by -1 to be compatible with multiplicities.")
                    charge -= 1
                self.QMin.template["charge"] = [i % 2 + charge for i in range(len(self.QMin.molecule["states"]))]
                self.log.info(
                    f'HINT: total charge per multiplicity automatically assigned, please check ({self.QMin.template["charge"]}).'
                )
                self.log.info('You can set the charge in the template manually for each multiplicity ("charge 0 +1 0 ...")')
            elif len(self.QMin.template["charge"]) >= len(self.QMin.molecule["states"]):
                self.QMin.template["charge"] = [
                    int(self.QMin.template["charge"][i]) for i in range(len(self.QMin.molecule["states"]))
                ]

                for imult, cha in enumerate(self.QMin.template["charge"]):
                    if not (self.QMin.molecule["Atomcharge"] + cha + imult) % 2 == 0:
                        self.log.warning(
                            "Charges from template not compatible with multiplicities!  (this is probably OK if you use QM/MM)"
                        )
                        break
            else:
                raise ValueError('Length of "charge" does not match length of "states"!')
        else:
            self.QMin.template["charge"] = [i % 2 for i in range(len(self.QMin.molecule["states"]))]
        if self.QMin.template["paddingstates"]:
            self.QMin.template["paddingstates"] = convert_list(self.QMin.template["paddingstates"])

    @abstractmethod
    def read_resources(self, resources_file: str, kw_whitelist: Optional[list[str]] = None) -> None:
        kw_whitelist = [] if kw_whitelist is None else kw_whitelist
        super().read_resources(resources_file, kw_whitelist + ["theodore_fragment"])

        if self.QMin.resources["theodore_fragment"]:
            self.QMin.resources["theodore_fragment"] = convert_list(self.QMin.resources["theodore_fragment"])

    @abstractmethod
    def read_requests(self, requests_file: str = "QM.in") -> None:
        super().read_requests(requests_file)

    def printQMout(self) -> None:
        super().writeQMout()

    @abstractmethod
    def setup_interface(self) -> None:
        # Setup charge and paddingstates
        if not self.QMin.template["charge"]:
            self.QMin.template["charge"] = [i % 2 for i in range(len(self.QMin.molecule["states"]))]
            self.log.info(f"charge not specified setting default, {self.QMin.template['charge']}")

        if not self.QMin.template["paddingstates"]:
            self.QMin.template["paddingstates"] = [0 for _ in self.QMin.molecule["states"]]
            self.log.info(
                f"paddingstates not specified setting default, {self.QMin.template['paddingstates']}",
            )

        # Setup chargemap
        self.log.debug("Building chargemap")
        self.QMin.maps["chargemap"] = {idx + 1: int(chrg) for (idx, chrg) in enumerate(self.QMin.template["charge"])}

        # Setup jobs
        self.QMin.control["states_to_do"] = [
            v + int(self.QMin.template["paddingstates"][i]) if v > 0 else v for i, v in enumerate(self.QMin.molecule["states"])
        ]

    @staticmethod
    def density_logic(s1,s2,spin):
        if s1.M == s2.M and s2.S - s1.S in [-2,0,2]:
                if s1.S == s2.S and spin == 'tot': return 1
                if spin in ['aa', 'bb' ]: return 1
                if (s1.S != s2.S or (s1.S == s2.S and s1.M !=0)) and spin == 'q': return 1
                #  if (s1.S != s2.S or s1.M !=0) and spin == 'q': return 1
        elif s2.M - s1.M == 2 and s2.S - s1.S in [-2, 0, 2] and spin == 'ba':
            return 1
        elif s2.M - s1.M == -2 and s2.S - s1.S in [-2, 0, 2] and spin == 'ab':
            return 1
        return 0

    def _request_logic(self):
        """
        Create maps from QMin object
        """
        self.log.debug("Setup interface -> building maps")
        super()._request_logic()
        # Setup gradmap
        if self.QMin.requests["grad"]:
            self.log.debug("Building gradmap")
            self.QMin.maps["gradmap"] = set({tuple(self.QMin.maps["statemap"][i][0:2]) for i in self.QMin.requests["grad"]})

        # Setup densmap
        if self.QMin.requests["multipolar_fit"] or self.QMin.requests["density_matrices"]:

            requested_densities = set()
            if self.QMin.requests["density_matrices"]:
                if self.QMin.requests["density_matrices"] == ["all"]:
                    for s1 in self.states:
                        for s2 in self.states:
                            for spin in ['tot', 'q', 'aa', 'bb', 'ab', 'ba']:
                                #if s1.S == s2.S and spin == 'q':
                                #    print(repr(s1), repr(s2), spin, 'logic = ', self.density_logic(s1,s2,spin)  )
                                if self.density_logic(s1,s2,spin): requested_densities.add((s1,s2,spin))
                else:
                    density_matrices = self.QMin.requests["density_matrices"] 
                    match len(density_matrices[0]):
                        case 2:
                            for d in density_matrices:
                                is1, is2 = d
                                s1 = self.states[is1]
                                s2 = self.states[is2]
                                for spin in ['tot', 'q', 'aa', 'bb', 'ab', 'ba']:
                                    if self.density_logic(s1,s2,spin): requested_densities.add((s1,s2,spin))
                        case 3:
                            for d in density_matrices:
                                is1, is2, spin = d
                                s1 = self.states[is1]
                                s2 = self.states[is2]
                                if self.density_logic(s1,s2,spin): 
                                    requested_densities.add((s1,s2,spin))
                                else:
                                    self.log.warning(f"Requested density {(s1, s2, spin)} is zero. Hence skipping it...")
                        case 6:
                            for density in density_matrices:
                                S1, M1, N1, S2, M2, N2 = density
                                if N1 > self.QMin.maps["states"][int(2*S1)-1] or N2 > self.QMin.maps["states"][int(2*S2)-1]:
                                    self.log.warning("Requested density ", density, 'refers to the states that are not going to be calculated. Hence skipping it...')
                                    continue
                                for s1 in self.states:
                                    if s1.S == int(2*S1) and s1.M == int(2*M1) and s1.N == N1:
                                        break
                                for s2 in self.states:
                                    if s2.S == int(2*S2) and s2.M == int(2*M2) and s2.N == N2:
                                        break
                                for spin in ['tot', 'q', 'aa', 'bb', 'ab', 'ba']:
                                    if self.density_logic(s1, s2, spin): requested_densities.add(d)
                        case 7:
                            for density in density_matrices:
                                S1, M1, N1, S2, M2, N2, spin = density
                                if N1 > self.QMin.maps["states"][int(2*S1)-1] or N2 > self.QMin.maps["states"][int(2*S2)-1]:
                                    self.log.warning(f"Requested density {density} refers to the states that are not going to be calculated. Hence skipping it...")
                                    continue
                                for s1 in self.states:
                                    if s1.S == int(2*S1) and s1.M == int(2*M1) and s1.N == N1:
                                        break
                                for s2 in self.states:
                                    if s2.S == int(2*S2) and s2.M == int(2*M2) and s2.N == N2:
                                        break
                                if self.density_logic(s1, s2, spin): 
                                    requested_densities.add(d)
                                else:
                                    self.log.warning(f"Requested density {density} is zero. Hence skipping it...")
                        case _:
                            raise NotImplementedError()

            if self.QMin.requests["multipolar_fit"]:
                requested_dmes = set()
                if self.QMin.requests["multipolar_fit"] == ["all"]:
                    for s1 in self.states:
                        for s2 in self.states:
                            if s1.S == s2.S and s1.M == s2.M and s1.S == s1.M and s2.S == s2.M:
                                requested_densities.add((s1,s2,'tot'))
                                requested_dmes.add((s1,s2))
                else:
                    multipolar_fit = self.QMin.requests["multipolar_fit"] 
                    match len(multipolar_fit[0]):
                        case 4:
                            for fit in multipolar_fit:
                                S1, N1, S2, N2 = density
                                if N1 > self.QMin.maps["states"][int(2*S1)-1] or N2 > self.QMin.maps["states"][int(2*S2)-1]:
                                    self.log.warning("Requested multipolar expansion ", fit, 'refers to the states that are not going to be calculated. Hence skipping it...')
                                    continue
                                for s1 in self.states:
                                    if s1.S == int(2*S1) and s1.M == int(2*S1) and s1.N == N1:
                                        break
                                for s2 in self.states:
                                    if s2.S == int(2*S2) and s2.M == int(2*S2) and s2.N == N2:
                                        break
                                if self.density_logic(s1, s2, 'tot'): 
                                    requested_densities.add((s1,s2,'tot'))
                                    requested_dmes.add((s1,s2))
                                else:
                                    self.log.warning(f"Requested multipolar exansion {fit} is zero. Hence skipping it...")
                        case _:
                            raise NotImplementedError()
                resp_layers = self.QMin.resources["resp_layers"]
                resp_density = self.QMin.resources["resp_density"]
                resp_flayer = self.QMin.resources["resp_first_layer"]
                resp_order = self.QMin.resources["resp_fit_order"]
                resp_grid = self.QMin.resources["resp_grid"]
                self.QMout.notes[
                    "multipolar_fit"
                ] = f" settings [order grid firstlayer density layers] {resp_order} {resp_grid} {resp_flayer} {resp_density} {resp_layers}"
                self.QMin.requests.types["multipolar_fit"] = dict
                self.QMin.requests["multipolar_fit"] = {dme: [] for dme in requested_dmes }

            self.QMin.requests.types["density_matrices"] = dict
            self.QMin.requests["density_matrices"] = {d: [] for d in requested_densities}
            #for key in self.QMin.requests['density_matrices']:
            #    s1, s2, spin = key
            #    print(s1.S, s1.M, s1.N, '|', s2.S, s2.M, s2.N, '|', spin)
            #sys.exit()
            self.get_density_recipes()

        # Setup nacmap
        if self.QMin.requests["nacdr"]:
            if self.QMin.requests["nacdr"] == ["all"]:
                mat = [
                    (i + 1, j + 1) for i in range(self.QMin.molecule["nmstates"]) for j in range(self.QMin.molecule["nmstates"])
                ]
                # self.QMin.requests["nacdr"] = mat
            else:
                mat = self.QMin.requests["nacdr"]
            self.log.debug("Building nacmap")
            self.QMin.maps["nacmap"] = set()
            for i in mat:
                m1, s1, ms1 = self.QMin.maps["statemap"][int(i[0])]
                m2, s2, ms2 = self.QMin.maps["statemap"][int(i[1])]
                if m1 != m2 or i[0] == i[1] or ms1 != ms2 or s1 > s2:
                    continue
                self.QMin.maps["nacmap"].add(tuple([m1, s1, m2, s2]))

        if self.QMin.requests["multipolar_fit"]:
            # construct shells
            shells, first, nlayers = map(self.QMin.resources.get, ("resp_shells", "resp_first_layer", "resp_layers"))

            # collect vdw radii for atoms from settings
            if self.QMin.resources["resp_vdw_radii"]:
                if len(self.QMin.resources["resp_vdw_radii"]) != len(self.QMin.molecule["elements"]):
                    raise RuntimeError("specify 'resp_vdw_radii' for all atoms!")
                self.QMin.resources["resp_vdw_radii"] = [float(x) for x in self.QMin.resources["resp_vdw_radii"]]
            else:
                # populate vdW radii
                radii = ATOMIC_RADII
                if self.QMin.resources["resp_mk_radii"]:
                    radii.update(MK_RADII)
                for e in filter(lambda x: x not in self.QMin.resources["resp_vdw_radii_symbol"], self.QMin.molecule["elements"]):
                    self.QMin.resources["resp_vdw_radii_symbol"][e] = radii[e]
                self.QMin.resources["resp_vdw_radii"] = [
                    self.QMin.resources["resp_vdw_radii_symbol"][s] for s in self.QMin.molecule["elements"]
                ]

            if self.QMin.resources["resp_betas"]:
                if len(self.QMin.resources["resp_betas"]) != self.QMin.resources["resp_fit_order"] + 1:
                    raise RuntimeError(
                        f"specify one beta parameter for each multipole order (order + 1)!\n needed {self.QMin.resources['resp_fit_order']+1:d}"
                    )
                self.log.info(f"using non-default beta parameters for resp fit {self.QMin.resources['resp_betas']}")

            if not shells:
                self.log.debug(f"Calculating resp layers as: {first} + 4/sqrt({nlayers})")
                incr = 0.4 / math.sqrt(nlayers)
                self.QMin.resources["resp_shells"] = [first + incr * x for x in range(nlayers)]
            if self.QMin.resources["resp_grid"] not in GRIDS:
                raise RuntimeError(
                    f"specified grid {self.QMin.resources['resp_grid']} not available.\n Possible options are 'lebedev', 'random', 'golden_spiral', 'gamess', 'marcus_deserno'"
                )

        if self.QMin.requests["ion"] or self.QMin.requests["overlap"]:
            self.log.debug(self.QMin.resources["wfoverlap"])
            assert is_exec(self.QMin.resources["wfoverlap"])

    @abstractmethod
    def getQMout(self):
        pass

    @abstractmethod
    def create_restart_files(self):
        pass

    @abstractmethod
    def get_mole(self):
        pass

    def run_program(self, workdir: str, cmd: str, out: str, err: str) -> int:
        """
        Runs a ab-initio programm and returns the exit_code

        workdir:    Path of the working directory
        cmd:        Contains path and arguments for execution of ab-initio program
        out:        Name of the output file
        err:        Name of the error file (optional)
        """
        current_dir = os.getcwd()
        os.chdir(workdir)
        self.log.debug(f"ab-initio call:\t {cmd}")
        self.log.debug(f"Working directory:\t {workdir}")

        with open(out, "w", encoding="utf-8") as outfile, open(err, "w", encoding="utf-8") as errfile:
            try:
                exit_code = sp.call(cmd, shell=True, stdout=outfile, stderr=errfile)
            except OSError as error:
                self.log.error(f"Execution of {cmd} failed!")
                raise OSError from error

        os.chdir(current_dir)
        return exit_code

    def runjobs(self, schedule: list[dict[str, QMin]]) -> dict[str, int]:
        """
        Runs all jobs in the schedule in a parallel queue

        schedule:   List of jobs (dictionary with jobnames and QMin objects)
                    First entry is a list with number of threads for the pool
                    for each job.
        """
        self.log.info("Starting job execution")
        error_codes = {}

        # Submit jobs to queue
        self.log.debug("Submit jobs to pool")
        for job_idx, jobset in enumerate(schedule):
            self.log.debug(f"Processing jobset number {job_idx} from schedule list")
            if not jobset:
                continue
            with Pool(processes=self.QMin.control["nslots_pool"][job_idx]) as pool:
                for job, qmin in jobset.items():
                    self.log.debug(f"Adding job: {job}")
                    workdir = os.path.join(self.QMin.resources["scratchdir"], job)
                    error_codes[job] = pool.apply_async(self.execute_from_qmin, args=(workdir, qmin))
                    time.sleep(self.QMin.resources["delay"])
                pool.close()
                pool.join()

        # Processing error codes
        error_string = "All jobs finished:\n"
        for job, code in error_codes.items():
            error_string += f"job: {job:<10s} code: {code.get()[0]:<4d} runtime: {code.get()[1]}\n"
        self.log.info(f"{error_string}")

        if any(map(lambda x: x.get()[0] != 0, error_codes.values())):
            raise RuntimeError("Some subprocesses did not finish successfully!")

        return error_codes

    @staticmethod
    def divide_slots(ncpu: int, ntasks: int, scaling: float, min_cpu=1) -> tuple[int, int, list[int]]:
        """
        This routine figures out the optimal distribution of the tasks over the CPU cores
        returns the number of rounds (how many jobs each CPU core will contribute to),
        the number of slots which should be set in the Pool,
        and the number of cores for each job.
        """
        ntasks_per_round = min(ncpu // min_cpu, ntasks)
        optimal = {}
        for i in range(1, 1 + ntasks_per_round):
            nrounds = int(math.ceil(ntasks / i))
            ncores = ncpu // i
            optimal[i] = nrounds / 1.0 / ((1 - scaling) + scaling / ncores)
        best = min(optimal, key=optimal.get)
        nrounds = int(math.ceil(ntasks / best))
        ncores = ncpu // best

        cpu_per_run = [0] * ntasks
        if nrounds == 1:
            itask = 0
            for _ in range(ncpu):
                cpu_per_run[itask] += 1
                itask += 1
                if itask >= ntasks:
                    itask = 0
            nslots = ntasks
        else:
            for itask in range(ntasks):
                cpu_per_run[itask] = ncores
            nslots = ncpu // ncores
        return nrounds, nslots, cpu_per_run

    @staticmethod
    def clean_savedir(path: str, retain: int, step: int) -> None:
        """
        Remove files older than step-retain

        path:       Path to savedir
        retain:     Number of timesteps to keep (-1 = all)
        step:       Current step
        """
        if retain < 0:
            return

        if not os.path.isdir(path):
            raise FileNotFoundError(f"{path} is not a directory!")

        files = os.listdir(path)
        for file in files:
            ext = os.path.splitext(file)[1].replace(".", "")
            if not re.match(r"^\d+$", ext):  # Skip if extension is not a number
                continue
            if int(ext) < step - retain:
                os.remove(os.path.join(path, file))

    #Start TOMI

    @staticmethod
    def density_representation(d):
        s1, s2, spin = d
        middle = '---'+spin+''.join(['-' for i in range(6-len(spin))])+'>'
        return '[ '+repr(s1)+' '+middle+' '+repr(s2)+ ' ]'
        
    def get_density_recipes(self):
        requested_densities = self.QMin.requests['density_matrices']
        readable_densities = self.get_readable_densities()
        doable_densities = readable_densities.copy()
        all_doable = all( [ True if density in doable_densities else False for density in requested_densities ] )
        if not all_doable:
            changed = True
            while changed:
                changed = self.append_constructable_densities( doable_densities )
        for method in self.QMin.template["density_calculation_methods"]:
            all_doable = all( [ True if density in doable_densities else False for density in requested_densities ] )
            if not all_doable:
                self.append_calculatable_densities( doable_densities, method )
                changed = True
                while changed:
                    changed = self.append_constructable_densities( doable_densities )
            else:
                break

        for d, v in doable_densities.items():
            v['repr'] = self.density_representation(d)

        self.log.trace('DOABLE DENSITIES:')
        for key, value in doable_densities.items():
            self.log.trace('        '+value['repr']+'    ' + str(value), format=False)

        all_doable = all( [ True if density in doable_densities else False for density in requested_densities ] )
        if not all_doable:
            self.log.error(' Following densities are not doable even after self-consistently joining readable, calculatable and constructable. Something is very wrong!')
            not_doables = [ value['repr'] for (key,value) in requested_densities.items() if not key in doable_densities ]
            for repr in not_doables:
                self.log.error(repr)
            raise RuntimeError

        # All requested densities need to be done...
        densities_to_be_done = { key:value for (key,value) in doable_densities.items() if key in requested_densities }
        # ...but maybe also some on which requested ones depend
        added = True
        while added:
            added = False
            to_append = [] 
            for key, value in densities_to_be_done.items():
                if isinstance( value.get('needed'), list ):
                    for dens in value['needed']:
                        if not dens in densities_to_be_done:
                            to_append.append(dens)
                            added = True
            densities_to_be_done.update({dens: doable_densities[dens] for dens in to_append})
        self.log.trace('\n'.join(map(lambda x: f"{x[0]}, {x[1]}", densities_to_be_done.items())))
        densities_to_be_read = { key:value for (key,value) in densities_to_be_done.items() if value.get('how') == 'read' }
        densities_to_be_calculated_from_determinants = { key:value for (key,value) in densities_to_be_done.items() if value.get('how') == 'from_determinants'}
        densities_to_be_calculated_from_gs2es = { key:value for (key,value) in densities_to_be_done.items() if value.get('how') == 'from_gs2es'}
        densities_to_be_constructed = { key:value for (key,value) in densities_to_be_done.items() if isinstance(value.get('how'),list) }
        self.density_recipes['read'] = densities_to_be_read
        self.density_recipes['from_determinants'] = densities_to_be_calculated_from_determinants
        self.density_recipes['from_gs2es'] = densities_to_be_calculated_from_gs2es
        self.density_recipes['construct'] = densities_to_be_constructed
        self.log.print('DENSITY RECIPES:')
        if self.QMin.resources['debug']: 
            self.log.print('   To be read:')
            for d, v in self.density_recipes['read'].items():
                self.log.debug('      '+v['repr'])
            self.log.debug('   To be calculated from gs2es:')
            for d, v in self.density_recipes['from_gs2es'].items():
                self.log.debug('      '+v['repr']+': GS = ',d[0].C['its_gs'])
            self.log.debug('   To be calculated from CI vectors:')
            for d, v in self.density_recipes['from_determinants'].items():
                self.log.debug('      '+v['repr']+': Multiplicities of dets files = ',set(v['needed']))
            self.log.debug('   To be constructed by Wigner-Eckart theorem:')
            for d, v in self.density_recipes['construct'].items():
                string = ''
                for coeff, dens in zip(v['how'], v['needed']):
                    sing = '+'
                    if coeff < 0.: sing = '-'
                    string = string+' '+sing+' '+str(sympy.nsimplify(abs(coeff)))+'['+doable_densities[dens]['repr']+']'
                    if doable_densities[dens].get('transpose',False): string += '.T'
                print('      '+v['repr']+' = '+string)
        return

    @abstractmethod
    def get_readable_densities(self):
        pass

    def append_calculatable_densities(self, doables, method):
        QMin = self.QMin
        if method == 'from_determinants':
            for s1 in self.states:
                for s2 in self.states:
                    if s1.Z == s2.Z and s1.M == s1.S and s2.M == s2.S:
                        if s1.S == s2.S:
                            density = ( s1, s2, 'aa' )
                            if not density in doables: doables[density] = { 'needed': (s1.S,s2.S), 'how': 'from_determinants' }
                            density = ( s1, s2, 'bb' )
                            if not density in doables: doables[density] = { 'needed': (s1.S,s2.S), 'how': 'from_determinants' }
                        elif s1.S == s2.S - 2:
                            density = ( s1, s2, 'ba' )
                            if not density in doables: doables[density] = { 'needed': (s1.S,s2.S), 'how': 'from_determinants' }
        elif method == 'from_gs2es':
            to_append = {}

            ground_states = []
            for s in self.states:
                if s.C['is_gs']: ground_states.append(s)

            for s1 in self.states:
                for s2 in self.states:
                    if self.density_logic(s1,s2,'aa') and not (s1,s2,'aa') in doables and not s1 is s2:
                        for gs in ground_states:
                            if gs is s1.C['its_gs'] and gs is s2.C['its_gs']:
                                break
                        if (s1,gs,'aa') in doables and (gs,s2,'aa') in doables and (s1,gs,'bb') in doables and (gs,s2,'bb') in doables:
                            to_append[(s1,s2,'aa')] = { 'how' :'from_gs2es', 'needed' : [(s1,gs,'aa'),(gs,s2,'aa') ] }
                            to_append[(s1,s2,'bb')] = { 'how' :'from_gs2es', 'needed' : [(s1,gs,'aa'),(gs,s2,'aa') ] }
            for key, value in to_append.items():
                doables[key] = value
        return

    def append_constructable_densities(self, doables):
        QMin = self.QMin
        added = False
        self.log.trace('Tu sam 6')
        #  for d in doables:
            #  print('Doable ', d)
        for s1 in self.states:
            for s2 in self.states:
                for spin in ['tot', 'q', 'aa', 'bb', 'ab', 'ba' ]:
                    density = (s1,s2,spin)
                    #  print(density, self.density_logic(s1,s2,spin), self.is_constructable( density, doables )[0])
                    if self.density_logic(s1,s2,spin) and not density in doables:
                        constructable, recipe = self.is_constructable( density, doables )
                        if constructable:
                            doables[density] = recipe
                            added = True
        return added

    @staticmethod
    def is_constructable(density, densities):
        thes1, thes2, thespin = density

        is_transposed_present = any( [ (thes1 is s2 and thes2 is s1 and thespin == spin[::-1]) for (s1,s2,spin) in densities ] )
        if is_transposed_present:
            return True, { 'needed': [(thes2,thes1,thespin[::-1])], 'how': [1.], 'transpose': True }

        equalZSNM = [ (s1,s2,spin) for (s1,s2,spin) in densities if s1 is thes1 and s2 is thes2 ]
        if len( equalZSNM ) > 0: # Only spin differs
            present_spins = [ spin for (s1,s2,spin) in equalZSNM ]
            if thespin == 'tot':
                if 'aa' in present_spins and 'bb' in present_spins:
                    return True, { 'needed': [ (thes1,thes2,'aa'), (thes1,thes2,'bb')], 'how': [1.,1.], 'transpose': False }
            elif thespin == 'q':
                if 'aa' in present_spins and 'bb' in present_spins:
                    return True, { 'needed': [ (thes1,thes2,'aa'), (thes1,thes2,'bb')], 'how': [1.,-1.], 'transpose': False }
            elif thespin == 'aa':
                if 'tot' in present_spins and 'bb' in present_spins:
                    return True, { 'needed': [ (thes1,thes2,'tot'), (thes1,thes2,'bb')], 'how': [1.,-1.], 'transpose': False }
                elif 'tot' in present_spins and 'q' in present_spins:
                    return True, { 'needed': [ (thes1,thes2,'tot'), (thes1,thes2,'q')], 'how': [0.5,0.5], 'transpose': False }
                elif 'q' in present_spins and 'bb' in present_spins:
                    return True, { 'needed': [ (thes1,thes2,'q'), (thes1,thes2,'bb')], 'how': [1.,1.], 'transpose': False }
                elif not SHARC_ABINITIO.density_logic(thes1,thes2,'tot') and 'q' in present_spins:
                    return True, { 'needed': [ (thes1,thes2,'q')], 'how': [-0.5], 'transpose': False } 
                elif not SHARC_ABINITIO.density_logic(thes1,thes2,'q') and 'tot' in present_spins:
                    return True, { 'needed': [ (thes1,thes2,'tot')], 'how': [0.5], 'transpose': False } 

            elif thespin == 'bb':
                if 'tot' in present_spins and 'aa' in present_spins:
                    return True, { 'needed': [ (thes1,thes2,'tot'), (thes1,thes2,'aa')], 'how': [1.,-1.], 'transpose': False }
                elif 'tot' in present_spins and 'q' in present_spins:
                    return True, { 'needed': [ (thes1,thes2,'tot'), (thes1,thes2,'q')], 'how': [0.5,-0.5], 'transpose': False }
                elif 'q' in present_spins and 'aa' in present_spins:
                    return True, { 'needed': [ (thes1,thes2,'aa'), (thes1,thes2,'q')], 'how': [1.,-1.], 'transpose': False }
                elif not SHARC_ABINITIO.density_logic(thes1,thes2,'tot') and 'q' in present_spins:
                    return True, { 'needed': [ (thes1,thes2,'q')], 'how': [0.5], 'transpose': False } 
                elif not SHARC_ABINITIO.density_logic(thes1,thes2,'q') and 'tot' in present_spins:
                    return True, { 'needed': [ (thes1,thes2,'tot')], 'how': [-0.5], 'transpose': False } 

        equalZSN = [ (s1,s2,spin) for (s1,s2,spin) in densities if (s1 == thes1 and s2 == thes2) ]
        if len( equalZSN ) > 0:
            if thespin == 'tot': # Assumes that thes1.M == thes2.M and thes1.S == thes2.S
                for d in equalZSN:
                    s1, s2, spin = d
                    if s1.M == s2.M and spin == 'tot':
                        return True, { 'needed': [d], 'how': [1.], 'transpose': False }
            elif thespin == 'q': # Assumes that thes1.M == thes2.M and ( thes1.S - thes2.S ) in [-2,0,2]
                for d in equalZSN:
                    s1, s2, spin = d
                    if spin == 'q':
                        A = wigner_3j( thes2.S/2, 1, thes1.S/2, thes2.M/2, 0, -thes1.M/2 ).evalf() 
                        B = wigner_3j( s2.S/2, 1, s1.S/2, s2.M/2, 0, -s1.M/2 ).evalf()  
                        coeff = float(A/B) 
                        return True, { 'needed': [d], 'how': [coeff], 'transpose': False }
                    elif s1.M == s2.M - 2 and spin == 'ba':
                        A = wigner_3j( thes2.S/2, 1, thes1.S/2, thes2.M/2, 0, -thes1.M/2 ).evalf() 
                        B = wigner_3j( s2.S/2, 1, s1.S/2, s2.M/2, -1, -s1.M/2 ).evalf() 
                        coeff = float(math.sqrt(2.)*(-1.)**(float(thes1.M)/2.-float(s1.M)/2. )*A/B)
                        return True, { 'needed': [d], 'how': [coeff], 'transpose': False }
            elif thespin == 'ba': 
                for d in equalZSN:
                    s1, s2, spin = d
                    if spin == 'q':
                        A = wigner_3j( thes2.S/2, 1, thes1.S/2, thes2.M/2, -1, -thes1.M/2 ).evalf() #sqr(3) 
                        B = wigner_3j( s2.S/2, 1, s1.S/2, s2.M/2, 0, -s1.M/2 ).evalf() #-sqrt(3)
                        coeff = float(1./math.sqrt(2.)*(-1.)**(float(thes1.M)/2.-float(s1.M)/2.)*A/B)
                        return True, { 'needed': [d], 'how': [coeff], 'transpose': False }
        # If all options are exceeded, it cannot be generated
        return False, {}

    def get_densities(self): 
        self.read_and_append_densities() # It has to take a look at self.density_recipes['read'] and actually read those densities and write them to QMout['density_matrices']
        self.calculate_from_determinants_and_append_densities()
        self.log.trace('Tu sam 2')
        for d in self.QMout['density_matrices']:
            print(d)
        missing_to_construct, missing_to_calculate = True, True
        if "from_gs2es" not in self.QMin.template["density_calculation_methods"]:
            missing_to_calculate = False
        i = 0
        while missing_to_calculate or missing_to_construct:
            i += 1
            print(missing_to_calculate, missing_to_construct)
            if missing_to_construct:
                missing_to_construct = self.construct_and_append_densities()
            self.log.trace('Tu sam 4', i)
            for d in self.QMout['density_matrices']:
                print(d)
            if missing_to_calculate:
                missing_to_calculate = self.calculate_from_gs2es_and_append_densities()
            self.log.trace('Tu sam 3', i)
            for d in self.QMout['density_matrices']:
                print(d)

        if self.QMin.resources['debug']:
            mol = self.QMin.molecule['mol'] 
            self.log.debug('NUMBERS OF ELECTRONS FROM DIFFERENT DENSITY MATRICES:')
            SAO = self.QMin.molecule['SAO'] 
            for d, rho in self.QMout['density_matrices'].items():
                s1,s2,spin = d
                N = np.einsum('ij,ij->', SAO, rho)
                self.log.debug(repr(s1)+" ---"+spin+"---> "+repr(s2)+" :"f"{N: 12.8f}")
            self.log.debug('State and transition dipole moments calculated from the total densities matrices:')
            nuclear_moment = np.sum( np.array( [ mol.atom_charge(j)*mol.atom_coord(j) for j in range(mol.natm) ] ), axis=0 )  
            mu = mol.intor('int1e_r')
            for d, rho in self.QMout['density_matrices'].items():
                s1,s2,spin = d
                if spin == 'tot' and s1.N <= s2.N:
                    x = -np.einsum('xij,ij->x', mu, rho)
                    if s1 is s2:
                        x += nuclear_moment
                    self.log.debug(repr(s1)+" ---> "+repr(s2)+": "+" ".join([f"{x[c]: 8.5f}" for c in range(3) ] )+" a.u.")



        return

    @abstractmethod
    def read_and_append_densities(self):
        pass

    def construct_and_append_densities(self):
        QMout = self.QMout
        densities = {}
        missing = False
        self.log.trace('Tu sam 8')
        for (density, recipe) in self.density_recipes['construct'].items():
            if not density in QMout["density_matrices"]:
                dens, coeffs = recipe['needed'], recipe['how']
                if all( [ True if d in QMout["density_matrices"] else False for d in dens ] ):
                    nao = self.QMin.molecule['mol'].nao
                    rho = np.zeros((nao,nao), dtype=float)
                    #  rho = reduce(lambda agg, (c,d): agg + c*self.QMout["density_matrices"][d], zip(dens,coeffs))
                    for d,c in zip(dens,coeffs):
                        rho += c*self.QMout["density_matrices"][d] 
                    #rho = np.sum(map(lambda x: x[0]*self.QMout["density_matrices"][x[1]], zip(dens,coeffs)), axis=0 )
                    if recipe['transpose']: rho = rho.T
                    QMout['density_matrices'][density] = rho
                    print('Success')
                    print(density)
                else:
                    print(density)
                    self.log.trace([d for d in dens if d not in QMout["density_matrices"]])
                    missing = True
        return missing

    def calculate_from_determinants_and_append_densities(self):
        QMin = self.QMin
        QMout = self.QMout
        from_determinants = self.density_recipes['from_determinants']
        self.log.trace(self.density_recipes['from_determinants'])
        if len(from_determinants) > 0:
            determinant_jobs = {}
            for (density,recipe) in from_determinants.items():
                if not density in QMout['density_matrices']:
                    if not recipe['needed'] in determinant_jobs:
                        determinant_jobs[recipe['needed']] = [density]
                    else:
                        determinant_jobs[recipe['needed']].append(density)
            for ( (S1, S2), densities ) in determinant_jobs.items(): 
                if S1 == S2:
                    print('Doing dM0 densities from determinants for multiplicity = ', S1+1)
                    nst, dets, CI, mos = self.read_dets_and_mos( QMin.save['savedir'], S1, QMin.save['step'] ) 
                    t1 = time.time()
                    rhos = wf2rho.deltaS0( QMin.template['tCI'], nst, dets, CI, mos )
                    t2 = time.time()
                    print(" Time elapsed in CI2rho_dM0 = ", round(t2-t1,3), "sec.")
                    for density in densities:
                        s1, s2, spin = density
                        if spin == 'aa':
                            QMout['density_matrices'][density] = rhos[0,s1.N-1, s2.N-1, :, :]
                        elif spin == 'bb':
                            QMout['density_matrices'][density] = rhos[1,s1.N-1, s2.N-1, :, :]
                elif S1 == S2 - 2:
                    print('Doing dM1 densities from determinants for multiplicities = ', S1+1, S2+1)
                    nst1, dets1, CI1, mos1 = self.read_dets_and_mos( QMin.save['savedir'], S1, QMin.save['step'] )
                    nst2, dets2, CI2, mos2 = self.read_dets_and_mos( QMin.save['savedir'], S2, QMin.save['step'] )
                    rhos = wf2rho.deltaS1( QMin.template['tCI'], nst1, nst2, dets1, dets2, CI1, CI2, mos1, mos2 )
                    for density in densities:
                        s1, s2, spin = density
                        QMout['density_matrices'][density] = rhos[s1.N-1, s2.N-1, :, :]
        return

    def calculate_from_gs2es_and_append_densities(self):
        QMin = self.QMin
        QMout = self.QMout
        SAO = QMin.molecule['SAO']
        missing = False
        self.log.trace('Tu sam 5')
        from_gs2es = self.density_recipes['from_gs2es']
        for d, r in from_gs2es.items():
            s1, s2, spin = d
            d1, d2 = r['needed']
            rho10 = QMout['density_matrices'].get(d1, None )
            rho02 = QMout['density_matrices'].get(d2, None )
            print(rho10, rho02)
            if rho10 is None or rho02 is None:
                missing = True
            else:
                rho = rho10 @ SAO @ rho02 - rho02 @ SAO @ rho10
                QMout['density_matrices'][(s1,s2,spin)] = rho
        return missing


    @staticmethod
    def read_dets_and_mos( dir, S, step ):
        file = dir+'/dets.'+str(S+1)+'.'+str(step) 
        nst = np.loadtxt(file,usecols=(0,),max_rows=1,dtype=int)
        nst = int(nst)
        dets = np.loadtxt(file,usecols=(0,),skiprows=1,dtype=str).tolist()
        CI = np.loadtxt(file,skiprows=1,usecols=[i for i in range(1,nst+1)],ndmin=2,dtype=float)
        file = dir+'/mos.'+str(S+1)+'.'+str(step) 
        nao = np.loadtxt(file, skiprows=5, max_rows=1, usecols=(0,),dtype=int)
        nmo = np.loadtxt(file, skiprows=5, max_rows=1, usecols=(1,),dtype=int)
        mos = np.zeros((nao,nmo))
        nr = nao//3
        if nao%3 != 0: nr += 1
        for i in range(nmo):
            mos[:,i] = np.concatenate( ( np.loadtxt(file,skiprows=9+i*nr,max_rows=nr-1).flatten(), np.loadtxt(file,skiprows=9+i*nr+nr-1,max_rows=1,ndmin=1) ))
        mos = np.ascontiguousarray(mos)
        dets = np.char.replace(dets,old='d',new='7,')
        dets = np.char.replace(dets,old='a',new='5,')
        dets = np.char.replace(dets,old='b',new='1,')
        dets = np.char.replace(dets,old='e',new='-1,')
        dets = np.array( [ np.fromstring(i,dtype=int,sep=',') for i in dets ] )
        return nst, dets, CI, mos

            

            





#    def dyson_orbitals_with_other(self, other):
#        QMin1 = self.QMin
#        QMin2 = other.QMin
#        save1 = QMin1.save["savedir"]
#        save2 = QMin2.save["savedir"]
#        step1 = QMin1.save["step"]
#        step2 = QMin2.save["step"]
#        nmos = 
#        phi_norms = {}
#        dyson_jobs = {}
#        for s1 in self.states:
#            for s2 in other.states:
#                if s1.Z == s2.Z -1 and s2.S - s1.S in [-1,1] and  s1.M == s1.S and s2.M == s2.S:
#                    if s2.M == s1.M + 1: spin = 'a' 
#                    if s2.M == s1.M - 1: spin = 'b' 
#
#                        if (s1.S,s2.S) in dyson_jobs:
#                            dyson_jobs[(s1.S,s2.S)].append( (s1,s2,'a') )
#                        else:
#                            dyson_jobs[(s1.S,s2.S)] = [ (s1,s2,'a') ]
#        for ( (S1, S2), DO ) in dyson_jobs.items():
#            mos1 = f"mos.{S1}.{step1}"
#            mos2 = f"mos.{S2}.{step2}"
#            dets1 = f"dets.{S1}.{step1}"
#            dets2 = f"dets.{S2}.{step2}"
#            generate_wfinp( mos1, mos2, dets1, dets2 )
#            call_wfoverlap(self)
#            phi[(m1,m2)] = read_wfout()
#        for (m1,m2), orbital in phi:
#            for ms1 in range(-m1+1,m1,2):
#                for ms2 in range(-m2+1,m2,2):
#                    if ms1 == -m1+1 and ms2 == -m2+1:
#                        dyson_orbitals[(m1,ms1,m2,ms2,'b')] = orbital
#                    elif ms1 == m1-1 and ms2 == m2-1:
#                        dyson_orbitals[(m1,ms1,m2,ms2,'a')] = orbital
#                    elif 
#
#    # End TOMI
#
#        def generate_wfinp( mos1, mos2, dets1, dets2 ):
#            f = open('wfovl.inp','w')
#            f.write('a_mo='+mos1+'\n')
#            f.write('b_mo='+mos2+'\n')
#            f.write('a_det='+dets1+'\n')
#            f.write('b_det='+dets2+'\n')
#            f.write('ao_read=-1\n')
#            f.write('same_aos=.true.\n')
#            f.write('moprint=1\n')
#            f.close()
#            return
#
#        def call_wfoverlap(interface):
#            QMin = interface.QMin
#            workdir = os.path.join(interface.QMin.resources["scratchdir"])
#            wf_cmd = f"{interface.QMin.resources['wfoverlap']} -m {interface.QMin.resources['memory']} -f wfovl.inp"
#            starttime = datetime.datetime.now()
#            os.environ["OMP_NUM_THREADS"] = str(QMin.resources["ncpu"])
#            code = interface.run_program(workdir, wf_cmd, "wfovl.out", "wfovl.err")
#            interface.log.info(
#                f"Finished wfoverlap job: {str(ion_pair):<10s} code: {code:<4d} runtime: {datetime.datetime.now()-starttime}"
#            )
#            if code != 0:
#                interface.log.error("wfoverlap did not finish successfully!")
#                with open(os.path.join(workdir, "wfovl.err"), "r", encoding="utf-8") as err_file:
#                    interface.log.error(err_file.read())
#                raise OSError()
#            return
                

    def _run_wfoverlap(self) -> None:
        """
        Prepare files and folders for wfoverlap and execute wfoverlap
        """

        # Content of wfoverlap input file
        wf_input = dedent(
            """\
        mix_aoovl=aoovl
        a_mo=mo.a
        b_mo=mo.b
        a_det=det.a
        b_det=det.b
        a_mo_read=0
        b_mo_read=0
        ao_read=0
        """
        )
        if self.QMin.resources["numocc"]:
            wf_input += f"\nndocc={self.QMin.resources['numocc']}"

        if self.QMin.resources["ncpu"] >= 8:
            wf_input += "\nforce_direct_dets"

        # cmdline string
        wf_cmd = f"OMP_NUM_THREADS={self.QMin.resources['ncpu']} {self.QMin.resources['wfoverlap']} -m {self.QMin.resources['memory']} -f wfovl.inp"

        # vars
        savedir = self.QMin.save["savedir"]
        step = self.QMin.save["step"]

        # Dyson calculations
        #self.dyson_orbitals_with_other(self,self)
        if self.QMin.requests["ion"]:
            for ion_pair in self.QMin.maps["ionmap"]:
                workdir = os.path.join(self.QMin.resources["scratchdir"], "Dyson_" + "_".join(str(ion) for ion in ion_pair))
                mkdir(workdir)
                # Write input
                writefile(os.path.join(workdir, "wfovl.inp"), wf_input)

                # Link files
                link(os.path.join(savedir, "AO_overl"), os.path.join(workdir, "aoovl"))
                link(os.path.join(savedir, f"dets.{ion_pair[0]}.{step}"), os.path.join(workdir, "det.a"))
                link(os.path.join(savedir, f"dets.{ion_pair[2]}.{step}"), os.path.join(workdir, "det.b"))
                link(os.path.join(savedir, f"mos.{ion_pair[1]}.{step}"), os.path.join(workdir, "mo.a"))
                link(os.path.join(savedir, f"mos.{ion_pair[3]}.{step}"), os.path.join(workdir, "mo.b"))

                # Execute wfoverlap
                starttime = datetime.datetime.now()
                # setting the env variable will influence subsequent numpy calls etc.
                # os.environ["OMP_NUM_THREADS"] = str(self.QMin.resources["ncpu"])
                code = self.run_program(workdir, wf_cmd, "wfovl.out", "wfovl.err")
                self.log.info(
                    f"Finished wfoverlap job: {str(ion_pair):<10s} code: {code:<4d} runtime: {datetime.datetime.now()-starttime}"
                )
                if code != 0:
                    self.log.error("wfoverlap did not finish successfully!")
                    with open(os.path.join(workdir, "wfovl.err"), "r", encoding="utf-8") as err_file:
                        self.log.error(err_file.read())
                    raise OSError()

        # Overlap calculations
        if self.QMin.requests["overlap"]:
            self._create_aoovl()
            for m in itmult(self.QMin.molecule["states"]):
                job = self.QMin.maps["multmap"][m]
                workdir = os.path.join(self.QMin.resources["scratchdir"], f"WFOVL_{m}_{job}")
                mkdir(workdir)
                # Write input
                writefile(os.path.join(workdir, "wfovl.inp"), wf_input)

                # Link files
                link(os.path.join(savedir, "AO_overl.mixed"), os.path.join(workdir, "aoovl"))
                link(os.path.join(savedir, f"dets.{m}.{step-1}"), os.path.join(workdir, "det.a"))
                link(os.path.join(savedir, f"dets.{m}.{step}"), os.path.join(workdir, "det.b"))
                link(os.path.join(savedir, f"mos.{self.QMin.maps['multmap'][m]}.{step-1}"), os.path.join(workdir, "mo.a"))
                link(os.path.join(savedir, f"mos.{self.QMin.maps['multmap'][m]}.{step}"), os.path.join(workdir, "mo.b"))

                # Execute wfoverlap
                starttime = datetime.datetime.now()
                os.environ["OMP_NUM_THREADS"] = str(self.QMin.resources["ncpu"])
                code = self.run_program(workdir, wf_cmd, "wfovl.out", "wfovl.err")
                self.log.info(
                    f"Finished wfoverlap job: {str(m):<10s} code {code:<4d} runtime: {datetime.datetime.now()-starttime}"
                )
                if code != 0:
                    self.log.error("wfoverlap did not finish successfully!")
                    with open(os.path.join(workdir, "wfovl.err"), "r", encoding="utf-8") as err_file:
                        self.log.error(err_file.read())
                    raise OSError()

    @abstractmethod
    def _create_aoovl(self) -> None:
        """
        Create AO_overl.mixed for overlap calculations
        """

    @staticmethod
    def parse_wfoverlap(overlap_file: str) -> np.ndarray:
        """
        Parse (Dyson) overlap matrix from wfoverlap output

        overlap_file: path to wfovlp.out
        """
        overlap_mat = []
        with open(overlap_file, "r", encoding="utf-8") as wffile:
            overlap_mat = []
            while True:
                line = next(wffile, False)
                if not line or containsstring("matrix <PsiA_i|PsiB_j>", line):
                    dim = -1 if not line else len(next(wffile).split()) // 2
                    break

            for line in wffile:
                if containsstring("<PsiA", line):
                    overlap_mat.append([float(x) for x in line.split()[2:]])
                else:
                    break

            if len(overlap_mat) != dim:
                raise ValueError(f"File {overlap_file} does not contain an overlap matrix!")
        return np.asarray(overlap_mat)

    def get_dyson(self, wfovl: str) -> np.ndarray:
        """
        Parse wfovlp output file and extract Dyson norm matrix

        wfovl:  Path to wfovlp.out
        """
        with open(wfovl, "r", encoding="utf-8") as file:
            raw_matrix = re.search(r"Dyson norm matrix(.*)", file.read(), re.DOTALL)

            if not raw_matrix:
                self.log.error(f"No Dyson matrix found in {wfovl}")
                raise ValueError()

            # Extract values and create numpy matrix
            value_list = list(map(float, re.findall(r"\d+\.\d{10}", raw_matrix.group(1))))

            dim = 1 if len(value_list) == 1 else math.sqrt(len(value_list))
            if dim > 1 and dim**2 != len(value_list):
                self.log.error(f"{wfovl} does not contain a square matrix!")
                raise ValueError()
            return np.asarray(value_list).reshape(-1, int(dim))

    @staticmethod
    def format_ci_vectors(ci_vectors: list[dict[tuple[int, ...], float]]) -> str:
        """
        Converts a list of ci vectors from (list[int],float) to str
        """
        alldets = set()
        for dets in ci_vectors:
            for key in dets:
                alldets.add((key))
        trans_table = str.maketrans({"0": "e", "1": "a", "2": "b", "3": "d"})
        string = f"{len(ci_vectors)} {len(next(iter(alldets)))} {len(alldets)}\n"
        for det in sorted(alldets, reverse=True):
            string += "".join(str(x) for x in det).translate(trans_table)
            for ci_vec in ci_vectors:
                if det in ci_vec:
                    string += f" {ci_vec[det]: 11.7f} "
                else:
                    string += f" {0: 11.7f} "
            string += "\n"
        return string

    # def do_fit(fits, pair, density_matrices, order, betas):
        # s1, s2 = pair
        # charge = s1.Z if s1 == s2 else 0
        # return (s1, s2), fits.multipoles_from_dens(
            # density_matrices[(s1, s2, 'tot')],
            # include_core_charges=s1 is s2,
            # order=order,
            # charge=charge,
            # betas=betas,
        # )

    def _resp_fit_on_densities(self) -> dict[(int, int, int, int, int, int), np.ndarray]:
        """
        Performs the resp fit on all densities given and returns the fits as dict.
        All transition densities need to be already present! Generate them with tdm.es2es_tdm() if necessary

        Args:
            basis: dict  basis set object as defined in pyscf [https://pyscf.org/user/gto.html#basis-format]
            densities: dict  dictionary (m1, s1, ms1, m2, s2, ms2) for 2D array with pyscf convention [https://pyscf.org/user/gto.html#ordering-of-basis-functions]
            cartesian_basis: bool indicates whether basis contains cartesian d,f,g,... functions
            ecps: dict  definition of effective core potentials in pyscf format [https://pyscf.org/user/gto.html#ecp]

        Returns:
            fits: dict  (same key as densities) dictionary on pairs of mult and state for each fit 2D array (natom,10)
        """
        self.log.info(f"{'RESP fit':=^80}")
        self.log.info("\t Start:")
        fits = Resp(
            self.QMin.coords["coords"],
            self.QMin.molecule["elements"],
            self.QMin.resources["resp_vdw_radii"],
            self.QMin.resources["resp_density"],
            self.QMin.resources["resp_shells"],
            grid=self.QMin.resources["resp_grid"],
            logger=self.log,
        )
        mol = self.QMin.molecule["mol"]
        gsmult = self.QMin.maps["statemap"][1][0]
        charge = self.QMin.maps["chargemap"][gsmult]  # the charge is irrelevant for the integrals calculated!!
        fits.prepare(
            mol
        )  # the charge of the atom does not affect integrals

        fits_map = {}
        get_transpose = []
        with Pool(processes=self.QMin.resources["ncpu"]) as pool:
            for dens in self.QMin.requests["multipolar_fit"]:
                s1, s2 = dens
                if (s2, s1) in fits_map:
                    get_transpose.append(dens)
                    continue
                charge = s1.Z if s1 == s2 else 0
                fits_map[dens] = pool.apply_async(
                    fits.multipoles_from_dens,
                    args=(
                        self.QMout.density_matrices[(s1, s2, "tot")],
                        s1 is s2,
                        charge,
                        self.QMin.resources["resp_fit_order"],
                        self.QMin.resources["resp_betas"]
                    ),
                )
            pool.close()
            pool.join()
            # fits_map = results.get()
            fits_map = {key: val.get() for key, val in fits_map.items()}

        for dens in get_transpose:
            s1, s2 = dens
            fits_map[dens] = fits_map[(s2, s1)]

        return fits_map

    @staticmethod
    def get_theodore(sumfile: str, omffile: str) -> dict[tuple[int], list[float]]:
        """
        Read and parse theodore output
        """
        out = readfile(sumfile)

        props = {}
        for line in out[2:]:
            s = line.replace("(", " ").replace(")", " ").split()
            if len(s) == 0:
                continue
            n = int(re.search("([0-9]+)", s[0]).groups()[0])
            m = re.search("([a-zA-Z]+)", s[0]).groups()[0]
            for i in IToMult:
                if isinstance(i, str) and m in i:
                    m = IToMult[i]
                    break
            props[(m, n + (m == 1))] = [safe_cast(i, float, 0.0) for i in s[3:]]

        out = readfile(omffile)

        for line in out[1:]:
            s = line.replace("(", " ").replace(")", " ").split()
            if len(s) == 0:
                continue
            n = int(re.search("([0-9]+)", s[0]).groups()[0])
            m = re.search("([a-zA-Z]+)", s[0]).groups()[0]
            for i in IToMult:
                if isinstance(i, str) and m in i:
                    m = IToMult[i]
                    break
            props[(m, n + (m == 1))].extend([safe_cast(i, float, 0.0) for i in s[2:]])
        return props

    def _run_theodore(self) -> None:
        """
        Prepare theodore files and run theodore
        """
        theo_bin = os.path.join(self.QMin.resources["theodir"], "bin", "theodore") + " analyze_tden"
        for jobset in self.QMin.scheduling["schedule"]:
            for job, qmin in jobset.items():
                # Skip unrestricted jobs
                if not self.QMin.control["jobs"][qmin.control["jobid"]]["restr"]:
                    self.log.debug(f"Skipping theodore run for unrestricted job {job}")
                    continue
                elif qmin.control["gradonly"]:
                    continue
                else:
                    mults = self.QMin.control["jobs"][qmin.control["jobid"]]["mults"]
                    gsmult = mults[0]
                    ns = 0
                    for i in mults:
                        ns += qmin.control["states_to_do"][i - 1] - (i == gsmult)
                    if ns == 0:
                        self.log.debug("Skipping Job %s because it contains no excited states." % (qmin.control["jobid"]))
                        continue

                starttime = datetime.datetime.now()
                workdir = os.path.join(self.QMin.resources["scratchdir"], job)
                self._setup_theodore(
                    workdir,
                    prop_list=self.QMin.resources["theodore_prop"],
                    at_lists=self.QMin.resources["theodore_fragment"],
                    **self._theodore_settings,
                )

                # Run theodore
                out_file = "theodore.out"
                err_file = "theodore.err"
                code = self.run_program(workdir, theo_bin, out_file, err_file)
                self.log.info(f"Finished theodore Job: {job:<10s} code: {code:<4d} runtime: {datetime.datetime.now()-starttime}")
                if code != 0:
                    self.log.error("Theodore job did not finish successfully!")
                    with open(os.path.join(workdir, err_file), "r", encoding="utf-8") as theo_err:
                        self.log.error(theo_err.read())
                    raise OSError("Theodore job did not finish successfully!")

    def _setup_theodore(
        self,
        workdir: str,
        rtype="cclib",
        rfile="ORCA.log",
        mo_file=None,
        read_binary=True,
        jmol_orbitals=False,
        molden_orbitals=False,
        Om_formula=2,
        eh_pop=1,
        comp_ntos=True,
        print_OmFrag=True,
        output_file="tden_summ.txt",
        prop_list=[],
        at_lists=[],
        link_files=[],
    ) -> None:
        """
        Write theodore input file and link files

        workdir:    Path of working directory
        **TheoDORE Keywords** https://sourceforge.net/p/theodore-qc/wiki/Keywords/
        link_files: list[(str, str)]   list of files to link (source, dest)
        """

        self.log.debug(f"Create theodore input file in {workdir}")
        theodore_keys = {
            "rtype": rtype,
            "rfile": rfile,
            "read_binary": read_binary,
            "jmol_orbitals": jmol_orbitals,
            "molden_orbitals": molden_orbitals,
            "Om_formula": Om_formula,
            "eh_pop": eh_pop,
            "comp_ntos": comp_ntos,
            "print_OmFrag": print_OmFrag,
            "output_file": output_file,
            "prop_list": prop_list,
            "at_lists": at_lists,
        }
        if mo_file:
            theodore_keys["mo_file"] = mo_file
        self.log.debug(f"theodore input with keys: {theodore_keys}")
        theodore_input = "\n".join(starmap(lambda k, v: f'{k}="{v}"' if type(v) == str else f"{k}={v}", theodore_keys.items()))
        writefile(os.path.join(workdir, "dens_ana.in"), theodore_input)
        for s, d in link_files:
            self.log.debug(f"\ttheodore: linking file {s} -> {d}")
            link(os.path.join(workdir, s), os.path.join(workdir, d))

        self.log.debug(f"================== DEBUG input file for WORKDIR {shorten_DIR(workdir)} =================")
        self.log.debug(theodore_input)
        self.log.debug(f'TheoDORE input written to: {os.path.join(workdir, "dens_ana.in")}')
        self.log.debug("====================================================================")

    # also add staticmethod
    # routine to read wfoverlap output

    @abstractmethod
    def run(self) -> None:
        """
        request & other logic
            requestmaps anlegen -> DONE IN SETUP_INTERFACE
            pfade für verschiedene orbital restart files
        make schedule
        runjobs()
        run_wfoverlap (braucht input files)
        run_theodore
        save directory handling
        """
