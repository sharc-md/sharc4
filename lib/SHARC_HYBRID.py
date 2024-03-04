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

import inspect
from collections.abc import Callable
from importlib import import_module

from SHARC_INTERFACE import SHARC_INTERFACE


class SHARC_HYBRID(SHARC_INTERFACE):
    """
    Abstract base class for SHARC hybrid interfaces
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Dict of child interfaces
        self._kindergarden = {}

    def instantiate_children(self, child_dict: dict[str, tuple[str, list, dict] | str]) -> None:
        """
        Populate kindergarden with instantiated child interfaces

        child_dict:     dictionary containing name of child and name of the interface or
                        a tuple with name of the interface and *args **kwargs
        """
        self.log.debug("Instantiace childs")

        for name, interface in child_dict.items():
            if name in self._kindergarden:
                self.log.error(f"{name} specified twice!")
                raise ValueError()

            if isinstance(interface, tuple):
                if len(interface) == 3 and isinstance(interface[1], list) and isinstance(interface[2], dict):
                    self._kindergarden[name] = self._load_interface(interface[0])(*interface[1], **interface[2])
                    self.log.debug(f"Assign instance of {interface[0]} to {name}")
                else:
                    self.log.error("Tuple must contain an interface name, an arg list and a kwarg dict!")
                    raise ValueError()
            else:
                self._kindergarden[name] = self._load_interface(interface)()
                self.log.debug(f"Assign instance of {interface} to {name}")

    def _load_interface(self, interface_name: str) -> Callable:
        """
        Dynamically loads interface from Python include path

        interface_name: Name of SHARC interface
        """

        interface_name = interface_name if interface_name.split("_")[0] == "SHARC" else f"SHARC_{interface_name}"
        try:
            module = import_module(interface_name)
        except (ModuleNotFoundError, ImportError, TypeError):
            self.log.error(f"{interface_name} could not be imported!")
            raise

        try:
            interface = getattr(module, interface_name)
            if not issubclass(interface, SHARC_INTERFACE):
                self.log.error(f"Class {interface_name} is not derived from SHARC_INTERFACE")
                raise ImportError()
            if inspect.isabstract(interface):
                self.log.error(f"{interface_name} is an abstract base class!")
                raise ImportError()
        except AttributeError as exc:
            self.log.error(f"Class {interface_name} not found in {module}")
            raise AttributeError from exc

        return interface


