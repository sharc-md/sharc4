import datetime
import os
import re
import shutil
from io import TextIOWrapper

import numpy as np
from constants import au2a
from qmin import QMin
from SHARC_ABINITIO import SHARC_ABINITIO
from utils import convert_list, expand_path, mkdir, writefile

AUTHORS = ""
VERSION = ""
VERSIONDATE = datetime.datetime(2023, 8, 29)
NAME = "MOLCAS"
DESCRIPTION = ""

CHANGELOGSTRING = """
"""

all_features = set(
    [
        "h",
        "dm",
        "soc",
        "nacdr",
        "grad",
        "ion",
        "overlap",
        "phases",
        "multipolar_fit",
        "molden",
        # raw data request
        "basis_set",
        "wave_functions",
        "density_matrices",
    ]
)


class SHARC_MOLCAS(SHARC_ABINITIO):
    """
    SHARC interface for MOLCAS
    """

    _version = VERSION
    _versiondate = VERSIONDATE
    _authors = AUTHORS
    _changelogstring = CHANGELOGSTRING
    _name = NAME
    _description = DESCRIPTION

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Add resource keys
        self.QMin.resources.update(
            {
                "molcas": None,
                "mpi_parallel": False,
                "schedule_scaling": 0.6,
                "delay": 0.0,
            }
        )

        self.QMin.resources.types.update(
            {
                "molcas": str,
                "mpi_parallel": bool,
                "schedule_scaling": float,
                "delay": float,
            }
        )

        # Add template keys
        self.QMin.template.update(
            {
                "basis": None,
                "baslib": None,
                "nactel": None,
                "ras2": None,
                "inactive": None,
                "roots": list(range(8)),
                "rootpad": list(range(8)),
                "method": "casscf",
                "functional": "tpbe",
                "douglas-kroll": False,
                "ipea": 0.25,
                "imaginary": 0.0,
                "frozen": -1,
                "cholesky": False,
                "gradaccudefault": 1e-4,
                "gradaccumax": 1e-2,
                "pcmset": None,
                "pcmstate": None,
                "iterations": [200, 100],
            }
        )

        self.QMin.template.types.update(
            {
                "basis": str,
                "baslib": str,
                "nactel": list,
                "ras2": int,
                "inactive": int,
                "roots": list,
                "rootpad": list,
                "method": str,
                "functional": str,
                "douglas-kroll": bool,
                "ipea": float,
                "imaginary": float,
                "frozen": int,
                "cholesky": bool,
                "gradaccudefault": float,
                "gradaccumax": float,
                "pcmset": (list, dict),
                "pcmstate": list,
                "iterations": list,
            }
        )

    @staticmethod
    def version() -> str:
        return SHARC_MOLCAS._version

    @staticmethod
    def versiondate() -> datetime.datetime:
        return SHARC_MOLCAS._versiondate

    @staticmethod
    def changelogstring() -> str:
        return SHARC_MOLCAS._changelogstring

    @staticmethod
    def authors() -> str:
        return SHARC_MOLCAS._authors

    @staticmethod
    def name() -> str:
        return SHARC_MOLCAS._name

    @staticmethod
    def description() -> str:
        return SHARC_MOLCAS._description

    @staticmethod
    def about() -> str:
        return f"{SHARC_MOLCAS._name}\n{SHARC_MOLCAS._description}"

    def get_features(self, KEYSTROKES: TextIOWrapper | None = None) -> set[str]:
        """return availble features

        ---
        Parameters:
        KEYSTROKES: object as returned by open() to be used with question()
        """
        return all_features

    def get_infos(self, INFOS: dict, KEYSTROKES: TextIOWrapper | None = None) -> dict:
        """prepare INFOS obj

        ---
        Parameters:
        INFOS: dictionary with all previously collected infos during setup
        KEYSTROKES: object as returned by open() to be used with question()
        """
        return INFOS

    def prepare(self, INFOS: dict, dir_path: str) -> None:
        pass

    def create_restart_files(self) -> None:
        pass

    def read_resources(self, resources_file: str = "MOLCAS.resources", kw_whitelist: list[str] | None = None) -> None:
        super().read_resources(resources_file, kw_whitelist)

        # Path to MOLCAS
        if not self.QMin.resources["molcas"]:
            self.log.error(f"molcas key not found in {resources_file}")
            raise ValueError()

        self.QMin.resources["molcas"] = expand_path(self.QMin.resources["molcas"])
        os.environ["MOLCAS"] = self.QMin.resources["molcas"]

        if self.get_molcas_version(self.QMin.resources["molcas"]) < (18, 0):
            self.log.error("This version of SHARC-MOLCAS is only compatible with MOLCAS 18 or higher!")
            raise ValueError()

        # MOLCAS driver
        driver = os.path.join(self.QMin.resources["molcas"], "bin", "pymolcas")
        if os.path.isfile(driver):
            self.QMin.resources.update({"driver": driver})

        driver = os.path.join(self.QMin.resources["molcas"], "bin", "molcas.exe")
        if os.path.isfile(driver) and "driver" not in self.QMin.resources:
            self.QMin.resources.update({"driver": driver})

        if "driver" not in self.QMin.resources:
            self.log.error(f"No driver found in {os.path.join(self.QMin.resources['molcas'], 'bin')}")
            raise ValueError()

        # WFOVERLAP
        if self.QMin.resources["wfoverlap"]:
            self.QMin.resources["wfoverlap"] = expand_path(self.QMin.resources["wfoverlap"])

        # Check orb init and guess
        if self.QMin.save["always_guess"] and self.QMin.save["always_orb_init"]:
            self.log.error("always_guess and always_orb_init cannot be used together!")
            raise ValueError()

        # Check ncpu
        if not self.QMin.resources["mpi_parallel"] and self.QMin.resources["ncpu"] != 1:
            self.log.warning("mpi_parallel not set but ncpu > 1, changeing ncpu to 1")
            self.QMin.resources["ncpu"] = 1

    def read_template(self, template_file: str = "MOLCAS.template", kw_whitelist: list[str] | None = None) -> None:
        super().read_template(template_file, kw_whitelist)

        # Roots
        self.QMin.template["roots"] = convert_list(self.QMin.template["roots"])
        self.QMin.template["rootpad"] = convert_list(self.QMin.template["rootpad"])

        if not all(map(lambda x: x >= 0, [*self.QMin.template["roots"], *self.QMin.template["rootpad"]])):
            self.log.error("roots and rootpad must contain positive integers.")
            raise ValueError()

        for idx, val in enumerate(self.QMin.molecule["states"]):
            if val > self.QMin.template["roots"][idx]:
                self.log.error(f"Too few states in state-averaging in multiplicity {idx+1}!")
                raise ValueError()

        for idx, val in enumerate(reversed(self.QMin.template["roots"])):
            if val == 0:
                self.QMin.template["roots"] = self.QMin.template["roots"][: -1 * idx]
                break

        self.QMin.template["rootpad"] = self.QMin.template["rootpad"][: len(self.QMin.template["roots"])]

        # Path to baslib
        if self.QMin.template["baslib"]:
            self.QMin.template["baslib"] = expand_path(self.QMin.template["baslib"])

        # Iterations
        match len(self.QMin.template["iterations"]):
            case 2:
                self.QMin.template["iterations"] = convert_list(self.QMin.template["iterations"])
            case 1:
                self.QMin.template["iterations"] = [int(self.QMin.template["iterations"][0]), 100]
            case _:
                self.log.error(f"{self.QMin.template['iterations']} is not a valid iteration value!")
                raise ValueError()

        # PCM
        if isinstance(self.QMin.template["pcmset"], list):
            if len(self.QMin.template["pcmset"][0]) != 3:
                self.log.error("pcmset must contain three parameter!")
                raise ValueError()

            self.QMin.template["pcmset"] = {
                "solvent": self.QMin.template["pcmset"][0][0],
                "aare": float(self.QMin.template["pcmset"][0][1]),
                "r-min": float(self.QMin.template["pcmset"][0][2]),
            }
        if self.QMin.template["pcmstate"]:
            self.QMin.template["pcmstate"] = convert_list(self.QMin.template["pcmstate"])

        # Check for basis and cas settings
        for i in ["basis", "nactel", "ras2", "inactive"]:
            if not self.QMin.template[i]:
                self.log.error(f"Key {i} is missing in template file!")
                raise ValueError()

        # Check nactel
        match len(self.QMin.template["nactel"]):
            case 1:
                self.QMin.template["nactel"] = [*self.QMin.template["nactel"], 0, 0]
            case 3:
                pass
            case _:
                self.log.error("nactel must contain either 1 or 3 numbers!")
                raise ValueError()

        # Validate method
        if self.QMin.template["method"].lower() not in [
            "casscf",
            "caspt2",
            "ms-caspt2",
            "mc-pdft",
            "xms-pdft",
            "cms-pdft",
            "xms-caspt2",
        ]:
            self.log.error(f"{self.QMin.template['method']} is not a valid method!")
            raise ValueError()

        # Validate functional
        if self.QMin.template["method"] == "cms-pdft" and self.QMin.template["functional"] not in [
            "tpbe",
            "t:pbe",
            "ft:pbe",
            "t:blyp",
            "ft:blyp",
            "t:revPBE",
            "ft:revPBE",
            "t:LSDA",
            "ft:LSDA",
        ]:
            self.log.error(f"No analytical gradients for cms-pdft and {self.QMin.template['functional']}.")
            raise ValueError()
            # TODO: other functionals for numerical?

        # TODO: GRADMODE stuff

    def setup_interface(self) -> None:
        """
        Setup MOLCAS interface
        """
        super().setup_interface()

    def run(self) -> None:
        starttime = datetime.datetime.now()

        # Generate schedule and run jobs
        self.log.debug("Generate schedule")
        self.QMin.scheduling["schedule"] = [{"master_1", self.QMin.copy()}]

        self.log.debug("Execute schedule")
        self.runjobs(self.QMin.scheduling["schedule"])

        # TODO: wfoverlap, theodore?

        self.log.debug("All hobs finished successful")

        self.QMout["runtime"] = datetime.datetime.now() - starttime

    def execute_from_qmin(self, workdir: str, qmin: QMin) -> tuple[int, datetime.timedelta]:
        """
        Setup workdir, write input file, copy initial guess, execute
        """
        self.log.debug(f"Create workdir {workdir}")
        mkdir(workdir)

        # Write files
        writefile(os.path.join(workdir, "MOLCAS.xyz"), self._write_geom(qmin.molecule["elements"], qmin.coords["coords"]))
        # TODO: input file

        # Make subdirs
        if qmin.resources["mpi_parallel"]:
            for i in range(qmin.resources["ncpu"]):
                self.log.debug(f"Create subdir tmp_{i+1}")
                mkdir(os.path.join(workdir, f"tmp_{i+1}"))

        

    def _gen_tasklist(self, qmin: QMin) -> list[list[str]]:
        """
        Generate tasklist
        """
        tasks = [["gateway"], ["seward"]]
        # TODO: qmmm

        if qmin.template["pcmset"]:  # TODO only with num grad?
            pcm_mult = qmin.template["pcmstate"][0]
            list_to_do = [(pcm_mult - 1, qmin.molecule["states"][pcm_mult - 1])]
            for idx, state in enumerate(qmin.molecule["states"]):
                if idx + 1 != pcm_mult:
                    list_to_do.append((idx, state))
        else:
            list_to_do = list((i, j) for i, j in enumerate(qmin.molecule["states"]))

        for mult, states in list_to_do:
            if states == 0:
                continue

            # TODO: do copy separate?

            is_jobiph = False
            is_rasorb = False
            if not qmin.save["always_guess"]:
                match qmin.save:
                    case {"init": True} | {"always_orb_init": True}:
                        if os.path.isfile(os.path.join(qmin.resources["pwd"], f"MOLCAS.{mult+1}.JobIph.init")):
                            tasks.append(["link", os.path.join(qmin.resources["pwd"], f"MOLCAS.{mult+1}.JobIph.init"), "JOBOLD"])
                            is_jobiph = True
                        elif os.path.isfile(os.path.join(qmin.resources["pwd"], f"MOLCAS.{mult+1}.RasOrb.init")):
                            tasks.append(["link", os.path.join(qmin.resources["pwd"], f"MOLCAS.{mult+1}.RasOrb.init"), "INPORB"])
                            is_rasorb = True

                    case {"samestep": True}:
                        tasks.append(
                            ["link", os.path.join(qmin.save["savedir"], f"MOLCAS.{mult+1}.JobIph.{qmin.save['step']}"), "JOBOLD"]
                        )
                        is_jobiph = True
                    case _:
                        tasks.append(
                            [
                                "link",
                                os.path.join(qmin.save["savedir"], f"MOLCAS.{mult+1}.JobIph.{qmin.save['step']-1}"),
                                "JOBOLD",
                            ]
                        )
                        is_jobiph = True

            # RASSCF
            allowed_functionals = [
                "tpbe",
                "t:pbe",
                "ft:pbe",
                "t:blyp",
                "ft:blyp",
                "t:revPBE",
                "ft:revPBE",
                "t:LSDA",
                "ft:LSDA",
            ]
            # WTF is this?!
            if not qmin.save["samestep"] or qmin.save["always_orb_init"]:
                if qmin.template["method"] == "cms-pdft" and qmin.template["functional"] in allowed_functionals:
                    if not qmin.save["init"]:
                        tasks.append(["copy", os.path.join(qmin.save["savedir"], f"Do_Rotate.{mult+1}.txt"), "Do_Rotate.txt"])
                tasks.append(["rasscf", mult + 1, qmin.template["roots"][mult], is_jobiph, is_rasorb])
                if qmin.template["method"] == "xms-pdft":
                    tasks[-1].append(["XMSI"])
                elif qmin.template["method"] == "cms-pdft":
                    tasks[-1].append(["CMSI"])
                if is_jobiph and not (
                    qmin.template["method"] == "cms-pdft" and qmin.template["functional"] in allowed_functionals
                ):
                    tasks.append(["rm", "JOBOLD"])

                if qmin.template["method"] in ["casscf", "mc-pdft"]:
                    tasks.append(["copy", "MOLCAS.JobIph", f"MOLCAS.{mult+1}.JobIph"])
                elif qmin.template["method"] == "cms-pdft" and qmin.template["functional"] in allowed_functionals:
                    tasks.append(["copy", "MOLCAS.JobIph", f"MOLCAS.{mult+1}.JobIph"])
                    if not qmin.save["init"]:
                        tasks.append(["rm", "JOBOLD"])

                if qmin.requests["ion"]:
                    tasks.append(["copy", "MOLCAS.RasOrb", f"MOLCAS.{mult+1}.RasOrb"])
                if qmin.requests["molden"]:
                    tasks.append(["copy", "MOLCAS.rasscf.molden", f"MOLCAS.{mult+1}.molden"])

                if qmin.template["method"] in ["mc-pdft", "xms-pdft", "cms-pdft"]:
                    keys = [f"KSDFT={qmin.template['functional']}"]
                    if qmin.requests["grad"]:
                        keys.append("GRAD")
                    else:
                        keys.append("noGrad")
                    if qmin.template["method"] in ["xms-pdft", "cms-pdft"]:
                        keys.append("MSPDFT")
                        keys.append("WJOB")
                    if qmin.template["method"] == "cms-pdft":
                        keys.append("CMMI=0")
                        keys.append("CMSS=Do_Rotate.txt")
                        keys.append("CMTH=1.0d-10")
                    tasks.append(["mcpdft", keys])
                    if qmin.template["method"] in ["xms-pdft", "cms-pdft"]:
                        tasks.append(["copy", "MOLCAS.JobIph", f"MOLCAS.{mult+1}.JobIph"])
            if not qmin.save["samestep"]:
                if qmin.template["method"] in ["caspt2", "ms-caspt2", "xms-caspt2"]:
                    tasks.append(["caspt2", mult + 1, states, qmin.template["method"]])
                    tasks.append(["copy", "MOLCAS.JobMix", f"MOLCAS.{mult+1}.JobIph"])

            # Gradients

            if self.QMin.requests["grad"]:
                for grad in qmin.maps["gradmap"]:
                    if grad[0] == mult + 1:
                        if qmin.template["roots"][mult] == 1:
                            # SS-CASSCF
                            if qmin.save["samestep"]:
                                tasks.append(["rasscf", mult + 1, qmin.template["roots"][mult], True, False])
                                if qmin.template["method"] == "mc-pdft":
                                    tasks.append(["mcpdft", [f"KSDFT={qmin.template['functional']}", "GRAD"]])
                                if qmin.template["method"] in ["ms-caspt2", "xms-caspt2"]:
                                    self.log.error("Single state gradient with MS/XMS-CASPT2")
                                    raise ValueError()
                            tasks.append(["alaska"])
                        else:
                            # SA-CASSCF
                            tasks.append(["link", f"MOLCAS.{mult+1}.JobIph", "JOBOLD"])

                            if qmin.template["method"] == "mc-pdft":
                                tasks.append(
                                    ["rasscf", mult + 1, qmin.template["roots"][mult], True, False, [f"RLXROOT={grad[1]}"]]
                                )
                                tasks.append(["mcpdft", [f"KSDFT={qmin.template['functional']}", "GRAD"]])
                            elif qmin.template["method"] == "cms-pdft":
                                if not qmin.save["init"]:
                                    tasks.append(
                                        ["copy", os.path.join(qmin.save["savedir"], f"Do_Rotate.{mult+1}.txt"), "Do_Rotate.txt"]
                                    )
                                tasks.append(
                                    [
                                        "rasscf",
                                        mult + 1,
                                        qmin.template["roots"][mult],
                                        True,
                                        False,
                                        [f"RLXROOT={grad[1]}", "CMSI"],
                                    ]
                                )
                                tasks.append(["mcpdft", [f"KSDFT={qmin.template['functional']}", "GRAD", "MSPDFT", "WJOB"]])
                                tasks.append(["alaska", grad[1]])
                            elif qmin.template["method"] == "casscf":
                                tasks.append(["rasscf", mult + 1, qmin.template["roots"][mult], True, False])
                                tasks.append(["mclr", qmin.template["gradaccudefault"], f"sala={grad[1]}"])
                            elif qmin.template["method"] in ["ms-caspt2", "xms-caspt2"]:
                                tasks.append(["rasscf", mult + 1, qmin.template["roots"][mult], True, False])
                                tasks.append(["caspt2", mult + 1, states, qmin.template["method"], f"GRDT\nrlxroot = {grad[1]}"])
                                tasks.append(["mclr", qmin.template["gradaccudefault"]])
                            if qmin.template["method"] not in ["caspt2", "xms-pdft", "cms-pdft"]:
                                tasks.append(["alaska"])

            # NACs
            if self.QMin.requests["nacdr"]:
                for nac in qmin.maps["nacmap"]:
                    if nac[0] == mult + 1:
                        tasks.append(["link", f"MOLCAS.{mult+1}.JobIph", "JOBOLD"])
                        if qmin.template["method"] == "casscf":
                            tasks.append(["rasscf", mult + 1, qmin["template"]["roots"][mult], True, False])
                            tasks.append(["mclr", qmin["template"]["gradaccudefault"], f"nac={nac[1]} {nac[3]}"])
                            tasks.append(["alaska"])
                        elif qmin.template["method"] == "cms-pdft":
                            if not qmin.save["init"]:
                                tasks.append(
                                    ["copy", os.path.join(qmin.save["savedir"], f"Do_Rotate.{mult+1}.txt"), "Do_Rotate.txt"]
                                )
                            tasks.append(["rasscf", mult + 1, qmin["template"]["roots"][mult], True, False, ["CMSI"]])
                            tasks.append(["mcpdft", [f"KSDFT={qmin.template['functional']}", "GRAD", "MSPDFT", "WJOB"]])
                            tasks.append(["mclr", qmin.template["gradaccudefault"], f"nac={nac[1]} {nac[3]}"])
                            tasks.append(["alaska"])
                        elif qmin.template["method"] in ["ms-caspt2", "xms-caspt2"]:
                            tasks.append(["rasscf", mult + 1, qmin["template"]["roots"][mult], True, False])
                            tasks.append(["caspt2", mult + 1, states, qmin.template["method"], f"GRDT\nnac = {nac[1]} {nac[3]}"])
                            tasks.append(["alaska", nac[1], nac[3]])

            # RASSI for overlaps
            if qmin.requests["overlap"]:
                tasks.append(
                    ["link", os.path.join(qmin.save["savedir"], f'MOLCAS.{mult+1}.JobIph.{qmin.save["step"]-1}'), "JOB001"]
                )
                tasks.append(["link", f"MOLCAS.{mult+1}.JobIph", "JOB002"])
                tasks.append(["rassi", "overlap", [states, states]])
                if qmin.requests["multipolar_fit"]:
                    for i in range(states):
                        for j in range(i + 1):
                            tasks.append(["copy", f"TRD2_{i+states+1:03d}_{j+states+1:03d}", f"TRD_{mult+1}_{i+1:03d}_{j+1:03d}"])

            # RASSI for dipole moments
            elif qmin.requests["dm"] or qmin.requests["ion"] or qmin.requests["multipolar_fit"]:
                tasks.append(["link", f"MOLCAS.{mult+1}.JobIph", "JOB001"])
                tasks.append(["rassi", "dm", [states]])
                if qmin.requests["multipolar_fit"]:
                    for i in range(states):
                        for j in range(i + 1):
                            tasks.append(["copy", f"TRD2_{i+1:03d}_{j+1:03d}", f"TRD_{mult+1}_{i+1:03d}_{j+1:03d}"])

        if qmin.requests["soc"]:
            i = 0
            roots = []
            for mult, states in enumerate(qmin.molecule["states"]):
                if states == 0:
                    continue
                i += 1
                roots.append(states)
                tasks.append(["link", f"MOLCAS.{mult+1}.JobIph", f"JOB{i:03d}"])
            tasks.append(["rassi", "soc", roots])

        self.log.debug(f"Generate tasklist\n{tasks}")
        return tasks

    def _write_geom(self, atoms: list[str], coords: list[list[float]] | np.ndarray) -> str:
        """
        Generate xyz file from coords
        """
        # TODO: qmmm
        geom_str = f"{len(atoms)}\n\n"
        for idx, (at, crd) in enumerate(zip(atoms, coords)):
            geom_str += f"{at}{idx+1}  {crd[0]*au2a:6f} {crd[1]*au2a:6f} {crd[2]*au2a:6f}\n"
        return geom_str

    def _create_aoovl(self) -> None:
        pass

    def dyson_orbitals_with_other(self, other) -> None:
        pass

    def getQMout(self) -> dict[str, np.ndarray]:
        pass

    def read_requests(self, requests_file: str = "QM.in") -> None:
        super().read_requests(requests_file)

        for req, val in self.QMin.requests.items():
            if val and req != "retain" and req not in all_features:
                self.log.error(f"Found unsupported request {req}.")
                raise ValueError()

        if self.QMin.template["method"] not in ["casscf", "ms-caspt2", "xms-caspt2"] and self.QMin.requests["nacdr"]:
            self.log.error("NACs are only possible with casscf, ms/xms-capt2!")
            raise ValueError()

    @staticmethod
    def get_molcas_version(path: str) -> tuple[int, int]:
        """
        Get version number of MOLCAS

        path:   Path to MOLCAS directory
        """

        with open(os.path.join(path, ".molcasversion"), "r", encoding="utf-8") as version_file:
            version = re.match(r"v(\d+)\.(\d+)", version_file.read())
            if not version:
                raise ValueError(f"No MOLCAS version found in {os.path.join(path, '.molcasversion')}")
        return int(version.group(1)), int(version.group(2))
