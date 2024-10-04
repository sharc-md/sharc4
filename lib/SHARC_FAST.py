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

import os
import shutil
from io import TextIOWrapper
from typing import Optional

# internal
from SHARC_INTERFACE import SHARC_INTERFACE
from utils import expand_path


class SHARC_FAST(SHARC_INTERFACE):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._threadsafe = True

    def get_infos(self, INFOS: dict, KEYSTROKES: Optional[TextIOWrapper] = None) -> dict:
        return INFOS

    def setup_interface(self):
        pass

    def getQMout(self):
        return self.QMout

    def prepare(self, INFOS: dict, dir_path: str):
        if "link_files" in INFOS and INFOS["link_files"]:
            os.symlink(
                expand_path(self.template_file),
                os.path.join(dir_path, self.name() + ".template"),
            )
            if "resources_file" in self.__dict__:
                os.symlink(
                    expand_path(self.resources_file),
                    os.path.join(dir_path, self.name() + ".resources"),
                )
            if "extra_files" in self.__dict__:
                for file in self.extra_files:
                    os.symlink(
                        expand_path(file),
                        os.path.join(dir_path, os.path.split(file)[1]),
                    )
            return

        shutil.copy(self.template_file, os.path.join(dir_path, self.name() + ".template"))
        if "resources_file" in self.__dict__:
            shutil.copy(self.resources_file, os.path.join(dir_path, self.name() + ".resources"))
        if "extra_files" in self.__dict__:
            for file in self.extra_files:
                shutil.copy(expand_path(file), os.path.join(dir_path, os.path.split(file)[1]))


    def clean_savedir(self) -> None:
        if not self.persistent:
            super().clean_savedir()



