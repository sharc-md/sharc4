#!/usr/bin/env python3
import datetime
import os
import re
from io import TextIOWrapper
from textwrap import dedent

import numpy as np
from constants import au2a
from qmin import QMin
from SHARC_ABINITIO import SHARC_ABINITIO
from utils import convert_list, mkdir, writefile

__all__ = ["SHARC_QCHEM"]

AUTHORS = "Sascha Mausenberger, Sebastian Mai"
VERSION = "4.0"
VERSIONDATE = datetime.datetime(2024, 8, 27)
NAME = "QCHEM"
DESCRIPTION = "SHARC interface for QCHEM"

CHANGELOGSTRING = """
"""

all_features = set(
    [
        "h",
        "ion",
        # raw data request
    ]
)


class SHARC_QCHEM(SHARC_ABINITIO):
    """
    SHARC interface for QCHEM
    """

    _version = VERSION
    _versiondate = VERSIONDATE
    _authors = AUTHORS
    _changelogstring = CHANGELOGSTRING
    _name = NAME
    _description = DESCRIPTION

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Add template keys
        self.QMin.template.update({"basis": None})

        self.QMin.template.types.update({"basis": str})

        # Add resource keys
        self.QMin.resources.update({"qchemdir": None, "dry_run": False})
        self.QMin.resources.types.update({"qchemdir": str, "dry_run": bool})

        self._ao_labels = None

    @staticmethod
    def version() -> str:
        return SHARC_QCHEM._version

    @staticmethod
    def versiondate() -> datetime.datetime:
        return SHARC_QCHEM._versiondate

    @staticmethod
    def changelogstring() -> str:
        return SHARC_QCHEM._changelogstring

    @staticmethod
    def authors() -> str:
        return SHARC_QCHEM._authors

    @staticmethod
    def name() -> str:
        return SHARC_QCHEM._name

    @staticmethod
    def description() -> str:
        return SHARC_QCHEM._description

    @staticmethod
    def about() -> str:
        return f"{SHARC_QCHEM._name}\n{SHARC_QCHEM._description}"

    def get_features(self, KEYSTROKES: TextIOWrapper | None = None) -> set[str]:
        """return availble features

        ---
        Parameters:
        KEYSTROKES: object as returned by open() to be used with question()
        """
        return all_features

    def get_infos(self, INFOS: dict, KEYSTROKES: TextIOWrapper | None = None) -> dict:
        pass

    def prepare(self, INFOS: dict, dir_path: str):
        pass

    def read_template(self, template_file: str = "QCHEM.template", kw_whitelist: list[str] | None = None) -> None:
        return super().read_template(template_file, kw_whitelist)

    def read_resources(self, resources_file: str = "QCHEM.resources", kw_whitelist: list[str] | None = None) -> None:
        return super().read_resources(resources_file, kw_whitelist)

    def execute_from_qmin(self, workdir: str, qmin: QMin) -> tuple[int, datetime.timedelta]:
        # Create workdir
        self.log.debug(f"Create workdir {workdir}")
        mkdir(workdir)

        writefile(os.path.join(workdir, "qchem.inp"), self._generate_input())

        self.run_program(
            workdir,
            f"{os.path.join(self.QMin.resources['qchemdir'], 'bin', 'qchem')} -nt {self.QMin.resources['ncpu']} -archive qchem.inp",
            "qchem.out",
            "qchem.err",
        )

    def run(self) -> None:
        starttime = datetime.datetime.now()
        if not self.QMin.resources["dry_run"]:
            self.execute_from_qmin(self.QMin.resources["scratchdir"], self.QMin)
        self.QMout["runtime"] = datetime.datetime.now() - starttime

    def _generate_input(self) -> str:
        input_str = "$molecule\n0 1\n"
        for atom, coord in zip(self.QMin.molecule["elements"], self.QMin.coords["coords"]):
            input_str += f"{atom:3s} {coord[0]*au2a:16.12f} {coord[1]*au2a:16.12f} {coord[2]*au2a:16.12f}\n"
        input_str += "$end"

        input_str += dedent(
            f"""
        $rem
        SYM_IGNORE            true
        CORRELATION           CCSD
        BASIS                 {self.QMin.template["basis"]}
        PURECART              1111
        CCMAN2                true
        EE_SINGLETS           [{self.QMin.molecule["states"][0]-1}]
        EE_TRIPLETS           [{self.QMin.molecule["states"][2]}]
        IP_STATES             [{self.QMin.molecule["states"][1]}]
        CC_TRANS_PROP         true
        CC_DO_DYSON           true
        GUI                   2
        POINT_GROUP_SYMMETRY  FALSE
        $end
        """
        )
        return input_str

    def _get_dyson_norms(self, output: str) -> np.ndarray:
        # GS -> D norms
        # S1 -> D1, S1 -> D2, S2 -> D1, S2 -> D2
        if not (gs := re.findall(r"Reference -- EOM-IP-CCSD state  \d+.*?Right\D+(\d\.\d+)", output, re.DOTALL)):
            self.log.error("No ground state Dyson norms found!")
            raise ValueError

        # ES -> D norms
        if not (es := re.findall(r"EOM-EE-CCSD state \d+\/A -- EOM-IP-CCSD state \d+.*?Right\D+(\d\.\d+)", output, re.DOTALL)):
            self.log.error("No excited state Dyson norms found!")
            raise ValueError
        return np.asarray(gs + es, dtype=float)

    def _get_dets(self, output: str) -> list[dict[tuple[int, int], float]]:
        # Get orbital numbers
        frz_a, _ = convert_list(re.findall(r"Frozen occupied\s+(\d+)", output))
        occ_a, _ = convert_list(re.findall(r"Active occupied\s+(\d+)", output))
        vir_a, _ = convert_list(re.findall(r"Active virtual\s+(\d+)", output))

        self.log.debug(f"Frozens {frz_a}, Occupied {occ_a}, Virtuals {vir_a}")

        # Get transition blocks
        transition_blocks = re.findall(r"EOMEE transition(.*?)Sum", output, re.DOTALL)
        if not transition_blocks:
            self.log.error("No transitions found in output file!")
            raise ValueError

        # Cannot parse double excitations, not worth the effort
        if any(re.findall(r"(-?\d\.\d{4})\s+(\d+)\D+(\d+) \(A\) (A|B)\s+\->", block) for block in transition_blocks):
            self.log.error("Double excitations found in log file!")
            raise ValueError
        # Filter dets
        dets = [re.findall(r"(-?\d\.\d{4})\s+(\d+)\D+(\d+) \(A\) (A|B)", block) for block in transition_blocks]
        dets = [[(float(trans[0]), int(trans[1]), int(trans[2]), trans[3]) for trans in det] for det in dets]

        occ_str = [3] * (occ_a + frz_a) + [0] * vir_a
        eigenvectors = [{tuple(occ_str): 1.0}]  # S0
        for det in dets:  # Includes all exc dets for S and T
            tmp = {}
            for trans in det:
                key = occ_str[:]
                key[trans[1] - 1], key[trans[2] - 1] = (2, 1) if trans[3] == "A" else (1, 2)
                tmp[tuple(key)] = trans[0] if trans[3] == "A" else -trans[0]  # Weird symmetry problem in QCHEM
                self.log.debug(f"Amp {trans[0]:8.4f} {trans[1]:3d} -> {trans[2]:3d} {trans[3]}")
            eigenvectors.append(tmp)
            self.log.debug(f"Norm: {sum(x[0]**2 for x in det)**0.5}\n")
        return eigenvectors

    def _get_energy(self, output: str) -> np.ndarray:
        # Get GS energy
        if not (gs := re.findall(r"CCSD total energy\s+=\s+(-?\d+\.\d+)", output)):
            self.log.error("No ground state energy found in log file!")
            raise ValueError
        energies = np.zeros(self.QMin.molecule["nmstates"], dtype=float)
        energies[0] = float(gs[0])

        states = self.QMin.molecule["states"]
        # Get excited energies S+T
        if not (st_energies := re.findall(r"EOMEE.*\n Total energy = (\-\d+\.\d+) a\.u", output)):
            self.log.error("No singlet and/or triplet energies found!")
            raise ValueError
        st_energies = np.array(st_energies, dtype=float)

        # Add singlets
        energies[1 : states[0]] = st_energies[: states[0] - 1]

        # Get doublets
        if not (d_energies := re.findall(r"EOMIP.*\n Total energy = (\-\d+\.\d+) a\.u", output)):
            self.log.error("No doublet energies found!")
            raise ValueError
        d_energies = np.array(d_energies, dtype=float)
        energies[states[0] : states[0] + states[1] * 2] = np.tile(d_energies, 2)

        # Add triplets
        energies[states[0] + states[1] * 2 :] = np.tile(st_energies[states[0] - 1 :], 3)
        return energies

    def setup_interface(self) -> None:
        return super().setup_interface()


    def _create_aoovl(self) -> None:
        return super()._create_aoovl()

    def getQMout(self) -> dict[str, np.ndarray]:
        # Allocate matrices
        requests = set()
        for key, val in self.QMin.requests.items():
            if not val:
                continue
            requests.add(key)

        self.log.debug("Allocate space in QMout object")
        self.QMout.allocate(
            states=self.QMin.molecule["states"],
            natom=self.QMin.molecule["natom"],
            npc=self.QMin.molecule["npc"],
            requests=requests,
        )

        # Get content of logfile
        with open(os.path.join(self.QMin.resources["scratchdir"], "qchem.out"), "r", encoding="utf-8") as f:
            output = f.read()

        # Save determinant files for overlaps
        dets = self._get_dets(output)
        if (sing := self.QMin.molecule["states"][0]) > 0:
            writefile(os.path.join(self.QMin.save["savedir"], "dets.1"), self.format_ci_vectors(dets[:sing]))
        if len(self.QMin.molecule["states"]) > 1 and self.QMin.molecule["states"][2] > 0:
            writefile(os.path.join(self.QMin.save["savedir"], "dets.3"), self.format_ci_vectors(dets[sing:]))

        # Get energies
        np.einsum("ii->i", self.QMout["h"])[:] = self._get_energy(output)

        # Get Dyson norms
        if self.QMin.requests["ion"]:
            norms = self._get_dyson_norms(output).reshape(-1, self.QMin.molecule["states"][1])
            dyson_mat = np.zeros((self.QMin.molecule["nstates"], self.QMin.molecule["nstates"]))
            for i in range(self.QMin.molecule["states"][0]):
                dyson_mat[i, self.QMin.molecule["states"][0] : sum(self.QMin.molecule["states"][:2])] = norms[i, :]
            for i in range(self.QMin.molecule["states"][2]):
                dyson_mat[
                    i + sum(self.QMin.molecule["states"][:2]),
                    self.QMin.molecule["states"][0] : sum(self.QMin.molecule["states"][:2]),
                ] = norms[i + self.QMin.molecule["states"][0], :]
            dyson_mat += dyson_mat.T

            dyson_nmat = np.zeros((self.QMin.molecule["nmstates"], self.QMin.molecule["nmstates"]))
            for i in range(dyson_nmat.shape[0]):
                for j in range(i, dyson_nmat.shape[0]):
                    m1, s1, ms1 = tuple(self.QMin.maps["statemap"][i + 1])
                    m2, s2, ms2 = tuple(self.QMin.maps["statemap"][j + 1])
                    if not abs(ms1 - ms2) == 0.5:
                        continue
                    if m1 > m2:
                        s1, s2, m1, m2, ms1, ms2 = s2, s1, m2, m1, ms2, ms1
                    if ms1 < ms2:
                        factor = (ms1 + 1 + (m1 - 1) / 2) / m1
                    else:
                        factor = (-ms1 + 1 + (m1 - 1) / 2) / m1
                    x = sum(self.QMin.molecule["states"][: m1 - 1]) + s1 - 1
                    y = sum(self.QMin.molecule["states"][: m2 - 1]) + s2 - 1
                    dyson_nmat[i, j] = dyson_nmat[j, i] = dyson_mat[x, y] * factor
            self.QMout["prop2d"].append(("ion", dyson_nmat))
        return self.QMout


if __name__ == "__main__":
    SHARC_QCHEM().main()
