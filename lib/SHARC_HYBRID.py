#!/usr/bin/env python3

# ******************************************
#
#    SHARC Program Suite
#
#    Copyright (c) 2024 University of Vienna
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
import os
import sys
from importlib import import_module
from multiprocessing import Manager, Process
from time import sleep

from pyscf.gto import Mole
from SHARC_INTERFACE import SHARC_INTERFACE


class SHARC_HYBRID(SHARC_INTERFACE):
    """
    Abstract base class for SHARC hybrid interfaces
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Dict of child interfaces
        self._kindergarden = {}

    @staticmethod
    def run_children(
        logger, children_dict: dict[str, SHARC_INTERFACE], ncpu: int, delay: float = 0.1, exit_on_failure: bool = True
    ) -> None:
        """
        Run all children in a parallel queue

        logger:             Logger object
        children_dict:      Dictionary of children that will be executed
        ncpu:               Maximal amount of CPUs used at once
        delay:              Delay time to check if a child can be added to queue, setting it to 0.0 will result
                            in high CPU ussage of the main Python process, but might be desired for very fast children
        exit_on_failure:    Kill all currently running jobs if one job in queue raises exception
        """
        manager = Manager()
        n_used_cpu = manager.Value("i", 0)
        qmins = manager.dict()
        qmouts = manager.dict()

        def run_a_child(label, n_used_cpu, QMins, QMouts):
            logger.info(f"Run child {label} on {os.uname()[1]} with pid: {os.getpid()}")
            try:
                children_dict[label]._step_logic()
                children_dict[label].QMout.mol = None
                children_dict[label].run()
                children_dict[label].getQMout()
                if children_dict[label].QMout.mol is not None:
                    children_dict[label].QMout.mol = Mole.pack(children_dict[label].QMout.mol)
                children_dict[label].clean_savedir()
                children_dict[label].write_step_file()
                QMins[label] = children_dict[label].QMin
                QMouts[label] = children_dict[label].QMout
            except:  # pylint: disable=bare-except
                logger.error(f"Some exception occured while running child {label}")
                sys.exit(1)  # Indicate failure of child process
            finally:
                n_used_cpu.value -= children_dict[label].QMin.resources["ncpu"]

        # Add jobs to queue until finished
        processes = []
        for label, child in children_dict.items():
            while True:
                if ncpu - n_used_cpu.value >= child.QMin.resources["ncpu"]:
                    processes.append(Process(target=run_a_child, args=(label, n_used_cpu, qmins, qmouts)))
                    n_used_cpu.value += child.QMin.resources["ncpu"]
                    processes[-1].start()
                    break
                sleep(delay)

        # Kill all processes if one fails
        while exit_on_failure:
            exit_on_failure = False
            for process in processes:
                if process.exitcode == 1:
                    for p in processes:
                        p.kill()
                    logger.error("A child process did not finish successfuly")
                    raise RuntimeError
                if process.is_alive():
                    exit_on_failure = True
            sleep(0.1)

        for process in processes:
            process.join()

        # Build mol objects if requested
        for label, child in children_dict.items():
            child.QMin = qmins[label]
            child.QMout = qmouts[label]
            if child.QMout.mol is not None:
                child.QMout.mol = Mole.unpack(child.QMout.mol)
                child.QMout.mol.build()

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

    def _load_interface(self, interface_name: str) -> SHARC_INTERFACE:
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

    def clean_savedir(self) -> None:
        """
        Clean save directory of all children in kindergarden
        """
        super().clean_savedir()
        for label, child in self._kindergarden.items():
            self.log.debug(f"Clean savedir from child {label}")
            child.clean_savedir()
