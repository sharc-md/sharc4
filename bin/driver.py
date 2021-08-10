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
## EXTERNAL
import os, sys
import numpy as np
from SHARC_INTERFACE import INTERFACE

## INTERNAL
from sharc import sharc
from factory import factory

name = "lvc"


def main():
    args = sys.argv
    if len(args) == 0:
        print("call with path to input file for SHARC")
        exit(0)
    inp_file = args[0]
    param = args[0:-1]

    IRestart = sharc.setup_sharc(inp_file)
        

    interface = factory(name)

    i: INTERFACE = interface()

    i.printheader()


