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

# give the variables debug and print as singletons

class Debug(object):
    
    val = False

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Debug, cls).__new__(cls)
        return cls.instance

    def __bool__(self):
        return self.val
    
    def set(self, val):
        self.val = val


class Print(object):
    
    val = True

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Print, cls).__new__(cls)
        return cls.instance

    def __bool__(self):
        return self.val

    def set(self, val):
        self.val = val


DEBUG = Debug()

PRINT = Print()
