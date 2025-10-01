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

# IMPORTS
# external

import os
import shutil
from io import TextIOWrapper

# internal
from SHARC_INTERFACE import SHARC_INTERFACE
from utils import expand_path, readfile


class SHARC_FAST(SHARC_INTERFACE):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.template_file = None
        self.resources_file = None
        self.extra_files = None
        self.savedict = {}

    def get_infos(self, INFOS: dict, KEYSTROKES: TextIOWrapper | None = None) -> dict:
        return INFOS

    def setup_interface(self):
        if self.persistent:
            # set last_step
            stepfile = os.path.join(self.QMin.save["savedir"], "STEP")
            last_step = None
            if os.path.isfile(stepfile):
                last_step = int(readfile(stepfile)[0])
            self.savedict["last_step"] = last_step


    def getQMout(self):
        return self.QMout

    def prepare(self, INFOS: dict, dir_path: str):
        if "link_files" in INFOS and INFOS["link_files"]:
            os.symlink(
                expand_path(self.template_file),
                os.path.join(dir_path, self.name() + ".template"),
            )
            if "resources_file" in self.__dict__ and self.resources_file:
                os.symlink(
                    expand_path(self.resources_file),
                    os.path.join(dir_path, self.name() + ".resources"),
                )
            if "extra_files" in self.__dict__ and self.extra_files:
                for file in self.extra_files:
                    os.symlink(
                        expand_path(file),
                        os.path.join(dir_path, os.path.split(file)[1]),
                    )
            return

        shutil.copy(self.template_file, os.path.join(dir_path, self.name() + ".template"))
        if "resources_file" in self.__dict__ and self.resources_file:
            shutil.copy(self.resources_file, os.path.join(dir_path, self.name() + ".resources"))
        if "extra_files" in self.__dict__ and self.extra_files:
            for file in self.extra_files:
                shutil.copy(expand_path(file), os.path.join(dir_path, os.path.split(file)[1]))


    def clean_savedir(self) -> None:
        if self.persistent:
            retain = self.QMin.requests["retain"]
            step = self.QMin.save["step"]
            if retain < 0:
                return
            to_be_deleted = set()
            for istep in self.savedict:
                if not isinstance(istep, int):
                    continue
                if istep < step - retain:
                    to_be_deleted.add(istep)
            for istep in to_be_deleted:
                del self.savedict[istep]
        else: 
            super().clean_savedir()

    def write_step_file(self) -> None:
        # write step file in every step only in non-persistent mode
        # TODO: This does not work, because of step_logic
        if not self.persistent:
            super().write_step_file()
        else:
            self.savedict["last_step"] = self.QMin.save['step']

    def create_restart_files(self):
        # write step file in persistent mode only at the very end
        if self.persistent:
            super().write_step_file()


