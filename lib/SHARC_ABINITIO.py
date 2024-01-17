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
from qmin import QMin
from SHARC_INTERFACE import SHARC_INTERFACE
from utils import containsstring, readfile, safe_cast, link, writefile, shorten_DIR, mkdir, itmult, convert_list, is_exec, theo_float, IToMult
from constants import ATOMIC_RADII, MK_RADII
from resp import Resp
from asa_grid import GRIDS

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

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Add ab-initio specific keywords to template
        self.QMin.template.update({"charge": None, "paddingstates": None})

        self.QMin.template.types.update({"charge": list, "paddingstates": list})
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

            def density_logic(m1, s1, ms1, m2, s2, ms2, mat=None):
                if ms1 == ms2 and abs(m1 - m2) <= 2:
                    if mat is None:
                        requested_densities.add((m1, s1, ms1, m2, s2, ms2, "aa"))
                        requested_densities.add((m1, s1, ms1, m2, s2, ms2, "bb"))
                        requested_densities.add((m1, s1, ms1, m2, s2, ms2, "tot"))
                    elif mat in ["aa", "bb", "tot"]:
                        requested_densities.add((m1, s1, ms1, m2, s2, ms2, mat))
                # TODO Tomi
                elif mat == "tot":
                    requested_densities.add((m1, s1, ms1, m2, s2, ms2, mat))
                else:
                    self.log.warning(f"density {m1, s1, ms1, m2, s2, ms2, mat} can currently not be constructed")
                    # raise NotImplementedError(f"{m1, s1, ms1, m2, s2, ms2, mat}")

            if self.QMin.requests["density_matrices"]:
                if self.QMin.requests["density_matrices"] == ["all"]:
                    for state1 in self.QMin.maps["statemap"].values():
                        for state2 in self.QMin.maps["statemap"].values():
                            density_logic(*state1, *state2)
                else:
                    # check if (itm, itm), (itm, itm, 'aa') or (m,s,ms, m,s,ms) or (m,s,ms, m,s,ms, 'aa')
                    match len(self.QMin.requests["density_matrices"][0]):
                        case 2:
                            for state1, state2 in map(
                                lambda x: (self.QMin.maps["statemap"][int(x[0])], self.QMin.maps["statemap"][int(x[1])]),
                                self.QMin.requests["density_matrices"],
                            ):
                                density_logic(*state1, *state2)
                        case 3:
                            for state1, state2, mat in map(
                                lambda x: (self.QMin.maps["statemap"][int(x[0])], self.QMin.maps["statemap"][int(x[1])], x[2]),
                                self.QMin.requests["density_matrices"],
                            ):
                                density_logic(*state1, *state2, mat)
                        case 6:
                            for m1, s1, ms1, m2, s2, ms2 in self.QMin.requests["density_matrices"]:
                                density_logic(m1, s1, ms1, m2, s2, ms2)
                        case 7:
                            for m1, s1, ms1, m2, s2, ms2, mat in self.QMin.requests["density_matrices"]:
                                density_logic(m1, s1, ms1, m2, s2, ms2, mat)
                        case _:
                            raise NotImplementedError()

            if self.QMin.requests["multipolar_fit"]:
                if self.QMin.requests["multipolar_fit"] == ["all"]:
                    for state1 in self.QMin.maps["statemap"].values():
                        for state2 in self.QMin.maps["statemap"].values():
                            if state1[2] == state2[2]:
                                density_logic(*state1, *state2, "tot")
                else:
                    match self.QMin.requests["multipolar_fit"][0]:
                        case 4:
                            for m1, s1, m2, s2 in self.QMin.requests["multipolar_fit"]:
                                for k1 in range(m1):
                                    ms1 = k1 - (m1 - 1) / 2.0
                                    for k2 in range(m2):
                                        ms2 = k2 - (m2 - 1) / 2.0
                                density_logic(m1, s1, ms1, m2, s2, ms2, "tot")
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

            self.QMin.requests.types["density_matrices"] = dict
            self.QMin.requests["density_matrices"] = {k: None for k in requested_densities}

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
            assert is_exec(self.QMin.resources["wfoverlap"])

    @abstractmethod
    def getQMout(self):
        pass

    @abstractmethod
    def create_restart_files(self):
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

        # Create restart files and garbage collection
        self.create_restart_files()
        self.clean_savedir(self.QMin.save["savedir"], self.QMin.requests["retain"], self.QMin.save["step"])

        return error_codes

    @staticmethod
    def divide_slots(ncpu: int, ntasks: int, scaling: float) -> tuple[int, int, list[int]]:
        """
        This routine figures out the optimal distribution of the tasks over the CPU cores
        returns the number of rounds (how many jobs each CPU core will contribute to),
        the number of slots which should be set in the Pool,
        and the number of cores for each job.
        """
        ntasks_per_round = min(ncpu, ntasks)
        optimal = {}
        for i in range(1, 1 + ntasks_per_round):
            nrounds = int(math.ceil(ntasks / i))
            ncores = ncpu // i
            optimal[i] = nrounds / 1.0 / ((1 - scaling) + scaling / ncores)
        best = min(optimal, key=optimal.get)
        nrounds = int(math.ceil(float(ntasks) // best))
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
        wf_cmd = f"{self.QMin.resources['wfoverlap']} -m {self.QMin.resources['memory']} -f wfovl.inp"

        # vars
        savedir = self.QMin.save["savedir"]
        step = self.QMin.save["step"]

        # Dyson calculations
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
                os.environ["OMP_NUM_THREADS"] = str(self.QMin.resources["ncpu"])
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

    def _resp_fit_on_densities(
        self, basis: dict, densities: dict[(int, int, int, int, int, int), np.ndarray], cartesian_basis=True, ecps={}
    ) -> dict[(int, int, int, int, int, int), np.ndarray]:
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
        gsmult = self.QMin.maps["statemap"][1][0]
        charge = self.QMin.maps["chargemap"][gsmult]  # the charge is irrelevant for the integrals calculated!!
        fits.prepare(
            basis, gsmult - 1, charge, ecps=ecps, cart_basis=cartesian_basis
        )  # the charge of the atom does not affect integrals

        fits_map = {}

        denskeys = {}
        for m1, s1, ms1, m2, s2, ms2 in densities.keys():
            key = (m1, s1, ms1, m2, s2, ms2)
            if m1 != m2:
                self.log.warning(f"fitting density different multiplicities! {m1}_{s1},{m2}_{s2}")
                self.log.warning(f"Charge is set to {self.QMin.maps['chargemap'][m1]} according to mult {m1}")
                continue
            if (m1, s1, m2, s2) in denskeys:
                fits_map[key] = fits_map[denskeys[(m1, s1, m2, s2)]]
                continue
            charge = self.QMin.maps["chargemap"][m1] if s1 == s2 else 0
            fits_map[key] = fits.multipoles_from_dens(
                densities[key],
                include_core_charges=s1 == s2,
                order=self.QMin.resources["resp_fit_order"],
                charge=charge,
                betas=self.QMin.resources["resp_betas"],
            )
            denskeys[(m1, s1, m2, s2)] = key

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
            n = int(re.search('([0-9]+)', s[0]).groups()[0])
            m = re.search('([a-zA-Z]+)', s[0]).groups()[0]
            for i in IToMult:
                if isinstance(i, str) and m in i:
                    m = IToMult[i]
                    break
            props[(m, n + (m == 1))] = [safe_cast(i, float, 0.0) for i in s[4:]]

        out = readfile(omffile)

        for line in out[1:]:
            s = line.replace("(", " ").replace(")", " ").split()
            if len(s) == 0:
                continue
            n = int(re.search('([0-9]+)', s[0]).groups()[0])
            m = re.search('([a-zA-Z]+)', s[0]).groups()[0]
            for i in IToMult:
                if isinstance(i, str) and m in i:
                    m = IToMult[i]
                    break
            props[(m, n + (m == 1))].extend([safe_cast(i, float, 0.0) for i in s[4:]])
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
