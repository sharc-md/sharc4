#!/usr/bin/env python3

# ******************************************
#
#    SHARC Program Suite
#
#    Copyright (c) 2019 University of Vienna
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

# IMPORTS
# external
import importlib.metadata

if int(importlib.metadata.version("schnetpack").split(".")[0]) > 1:
    raise ImportError("SchNarc requires schnetpack version < 2!")

import datetime
from io import TextIOWrapper

import numpy as np
import torch
from schnarc import calculators
# internal
from SHARC_FAST import SHARC_FAST
from utils import *

authors = "Maximilian Xaver Tiefenbacher"
version = "4.0"
versiondate = datetime.datetime(2023, 7, 15)

changelogstring = ""


class SHARC_SCHNARC(SHARC_FAST):

    _version = version
    _versiondate = versiondate
    _authors = authors
    _changelogstring = changelogstring

    @staticmethod
    def version():
        return SHARC_SCHNARC._version

    @staticmethod
    def versiondate():
        return SHARC_SCHNARC._versiondate

    @staticmethod
    def changelogstring():
        return SHARC_SCHNARC._changelogstring

    @staticmethod
    def authors():
        return "Maximilian Xaver Tiefenbacher"

    @staticmethod
    def name():
        return "SCHNARC"

    @staticmethod
    def description():
        return "SCHNARC model calculations"

    def get_features(self, KEYSTROKES: TextIOWrapper = None) -> set:
        return set(
            [
                "h",
                "soc",
                "dm",
                "grad",
                "nacdr",
                "point_charges",
                "grad_pc",
            ]
        )

    def read_template(self, template_filename="SCHNARC.template"):
        """reads the template file
        has to be called after setup_mol!"""

        kw_whitelist = {"modelpath","paddingstates"}
        QMin = self.QMin
        QMin.template.types = {"modelpath": str,"paddingstates":list}
        QMin.template.data = {"modelpath": "best_model","paddingstates":None}

        super().read_template(template_filename, kw_whitelist=kw_whitelist)
        return

    def read_resources(self, resources_filename="SCHNARC.resources"):
        super().read_resources(resources_filename)

    def get_infos(self, INFOS: dict, KEYSTROKES: TextIOWrapper | None = None) -> dict:
        self.log.info("=" * 80)
        self.log.info(f"{'||':<78}||")
        self.log.info(f"||{'SCHNARC interface setup':^76}||\n{'||':<78}||")
        self.log.info("=" * 80)
        self.log.info("\n")
        if os.path.isfile("SCHNARC.template"):
            self.log.info("Found SCHNARC.template in current directory")
            if question("Use this template file?", bool, KEYSTROKES=KEYSTROKES, default=True):
                self._template_file = "SCHNARC.template"
        else:
            self.log.info("Specify a path to a SCHNARC template file.")
            while not os.path.isfile(template_file := question("Template path:", str, KEYSTROKES=KEYSTROKES)):
                self.log.info(f"File {template_file} does not exist!")
            self._template_file = template_file

        return INFOS

    def prepare(self, INFOS: dict, dir_path: str):
        create_file = link if INFOS["link_files"] else shutil.copy
        if not self.resources_file:
            with open(os.path.join(dir_path, "SCHNARC.template"), "w", encoding="utf-8") as file:
                if "modelpath" in self.setupINFOS:
                    file.write(f"modelpath {self.setupINFOS['modelpath']}\n")
                else:
                    self.log.error("Modelpath not specified!")
                    raise ValueError
        else:
            create_file(expand_path(self.resources_file), os.path.join(dir_path, "SCHNARC.resources"))
        create_file(expand_path(self._template_file), os.path.join(dir_path, "SCHNARC.template"))

    ###################### run routines ######################

    def setup_interface(self):
        # param are some paramters for the neuralnetwork, which can be changed in the definition below
        # the SchNarculator is initialized with dummy coordinates and fieldschnet is enabled if point charges are found
        param = parameters()
        dummy_crd = torch.zeros(len(self.QMin.molecule["elements"]), 3)
        device = torch.device("cpu")
        if self.QMin.molecule["point_charges"]:
            self.models = calculators.SchNarculator(
                dummy_crd,
                self.QMin.molecule["elements"],
                self.QMin.template["model_file"],
                param=param,
                hessian=False,
                nac_approx=[1, None, None],
                adaptive=None,
                qmmm=True,
            )
        else:
            self.models = calculators.SchNarculator(
                dummy_crd,
                self.QMin.molecule["elements"],
                self.QMin.template["model_file"],
                param=param,
                hessian=False,
                nac_approx=[1, None, None],
                adaptive=None,
                qmmm=False,
            )

    def run(self):

        sharc_out = self.QMin.coords["coords"]
        if self.QMin.molecule["point_charges"]:
            # if point charges are included in the prediction we assume you are using filedschnet
            # therefore we will compute the electric field and add it to the input of the NN
            fields = self.get_pointfield()
            coords = self.QMin.coords["coords"]
            print("we have pc")
            sharc_out = {"positions": coords, "external_charges": self.QMin.coords["pccharge"], "electric_field": fields}
        else:
            fields = np.zeros((self.QMin.molecule["natom"], 3))
            sharc_out = {
                "positions": self.QMin.coords["coords"],
                "external_charges": self.QMin.coords["pccharge"],
                "electric_field": fields,
            }
        NN_out = self.models.calculate(sharc_out)
        keys = NN_out.keys()
        #'dm', 'nacdr', 'h', 'grad', 'dydf', 'pc_grad'

        self.log.debug(NN_out.keys())
        for key in NN_out.keys():
            match key:

                case "h":
                    self.log.debug(key)
                    print(NN_out["h"])
                    self.QMout.h = np.array(NN_out["h"])
                case "grad":
                    self.log.debug(key)
                    self.QMout.grad = np.array(NN_out["grad"])
                case "dydf":
                    if not self.QMin.molecule["point_charges"]:
                        # technically you could have an NN, which predicts this gradient even though there are no point charges in SHARC
                        continue
                    else:
                        # computing the gradient of each point charge with respect to each atom
                        # summing over the corresponding axis afterwards to get the correct dimensions
                        pc_grads = self.get_pc_grad(NN_out["dydf"])
                        self.QMout.grad += np.sum(pc_grads, axis=2) * (-1)
                        print(np.linalg.norm(np.sum(pc_grads, axis=2)))
                        print(np.linalg.norm(NN_out["grad"]))
                        self.QMout.grad_pc = np.sum(pc_grads, axis=1)  # *(-1)
                    self.log.debug(key)
                case "dm":
                    self.QMout.dm = np.array(NN_out["dm"])
                case "nacdr":
                    self.QMout.nacdr = np.array(NN_out["nacdr"])
                    self.log.debug(key)
                case "socdr":
                    self.QMout.socdr = np.array(NN_out["socdr"])
                case _:
                    self.log.warning(key, " is not implemented")
        return None

    def get_pointfield(self):
        field = np.zeros((len(self.QMin.coords["coords"]), 3))
        atom_nr = 0
        for atom in self.QMin.coords["coords"]:
            i = 0
            for point in self.QMin.coords["pccoords"]:
                dist_vec = atom - point
                field[atom_nr] += self.QMin.coords["pccharge"][i] * dist_vec / (np.linalg.norm(dist_vec, 2) ** 3)
                i += 1
            atom_nr += 1
        return field

    def getQMout(self):
        # everything is already
        self.QMout.states = self.QMin.molecule["states"]
        self.QMout.nstates = self.QMin.molecule["nstates"]
        self.QMout.nmstates = self.QMin.molecule["nmstates"]
        self.QMout.natom = self.QMin.molecule["natom"]
        self.QMout.npc = self.QMin.molecule["npc"]
        self.QMout.point_charges = False
        return self.QMout

    def create_restart_files(self):
        # x=open("restart/restart","w")
        # x.write(str(self.QMin.coords['coords']))
        # x.close()
        # return None
        pass

    def get_pc_grad(self, dEdF):
        # computes the gradients of the energy, which include the point charges
        # requires the change of energy with the electric field
        N_MM = self.QMin.coords["pccharge"].shape[0]
        N_QM = self.QMin.coords["coords"].shape[0]
        dF_dx = np.zeros((N_QM, N_MM, 3, 3))
        # loop over all qm atoms computing the change of the electric field with respect to the coordinates of atoms and point charges
        for qm_idx, qm_atom in enumerate(self.QMin.coords["coords"]):
            dists = np.subtract(qm_atom, self.QMin.coords["pccoords"])
            norms = np.linalg.norm(dists, axis=1).reshape(N_MM, 1, 1)
            mat = 3.0 * np.einsum("ij,il->ijl", dists, dists) / np.power(norms, 2) - np.identity(3)
            mat /= np.power(norms, 3)
            dF_dx[qm_idx] = mat * (self.QMin.coords["pccharge"].reshape(N_MM, 1, 1))
        return np.einsum("ijk,jlkm->ijlm", dEdF, dF_dx)


class parameters:
    cuda = False
    socs_mask = np.zeros(1)
    socmodel = None
    nacmodel = None
    emodel2 = None
    finish = False
    environment_provider = "simple"
    diss_tyro = None
    only_energy = True


if __name__ == "__main__":
    from logger import loglevel

    try:
        schnarc = SHARC_SCHNARC(loglevel=loglevel)
        schnarc.main()
    except KeyboardInterrupt:
        print("\nCTRL+C makes me a sad SHARC ;-(")
