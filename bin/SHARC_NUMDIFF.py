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
import datetime
import os
import shutil
from copy import deepcopy
from io import TextIOWrapper
import numpy as np
from typing import Optional
import itertools

# internal
from constants import ATOMCHARGE, FROZENS
# from factory import factory
from SHARC_HYBRID import SHARC_HYBRID
from utils import InDir, mkdir, question, cleandir, writefile

version = '4.0'
versiondate = datetime.datetime(2025, 4, 1)

changelogstring = '''
'''
np.set_printoptions(linewidth=400)

numdiff_debug = False

# TODO: define __all__


def phase_correction(matrix):
    """
    Do a phase correction of a matrix.
    Follows algorithm from J. Chem. Theory Comput. 2020, 16, 2, 835-846 (https://doi.org/10.1021/acs.jctc.9b00952)
    """
    phases = np.ones(matrix.shape[-1])
    U = matrix.real.copy()
    det_U = np.linalg.det(U)
    if det_U < 0:
        U[:, 0] *= -1.0  # this row/column convention is correct
        phases[0] *= -1.0
    U_sq = U * U

    # sweeps
    length = len(U)
    sweeps = 0
    done = False
    while not done:
        done = True
        for j in range(length):
            for k in range(j + 1, length):
                delta = 3.0 * (U_sq[j, j] + U_sq[k, k])
                delta += 6.0 * U[j, k] * U[k, j]
                delta += 8.0 * (U[k, k] + U[j, j])
                delta -= 3.0 * (U[j, :] @ U[:, j] + U[k, :] @ U[:, k])

                # Test if delta < 0
                num_zero_thres = -1e-15  # needs proper threshold towards 0
                if delta < num_zero_thres:
                    U[:, j] *= -1.0  # this row/column convention is correct
                    U[:, k] *= -1.0  # this row/column convention is correct
                    phases[j] *= -1.0
                    phases[k] *= -1.0
                    done = False
        sweeps += 1
    
    if numdiff_debug:
        print(f"Finished phase correction after {sweeps} sweeps.")

    return U, phases


def loewdin_orthonormalization(A):
    """
    Do Loewdin orthonormalization of a matrix.
    """
    S = A.T @ A
    eigenvals, eigenvecs = np.linalg.eigh(S)
    idx = eigenvals > 1e-15
    S_sqrt = np.dot(eigenvecs[:, idx] / np.sqrt(eigenvals[idx]), eigenvecs[:, idx].conj().T)
    A_ortho = A @ S_sqrt
    
    # Normalize the matrix
    A_lo = A_ortho.T
    length = len(A_lo)
    A_lon = np.zeros((length, length))

    for i in range(length):
        norm_of_col = np.linalg.norm(A_lo[i])
        A_lon[i] = [e / (norm_of_col ** 0.5) for e in A_lo[i]]
    
    return A_lon.T
    

def post_process_overlap_matrix(overlap_matrix):
    """
    Process an overlap matrix to ensure that it has correct phases
    and in orthonormal.
    """
    # First fix phases
    phase_corrected_overlap, phases = phase_correction(overlap_matrix)

    # Do a Loewdin orthonormalization
    orthogonal_overlap = loewdin_orthonormalization(phase_corrected_overlap)

    # Extra phase correction (probably not needed)
    final, phases2 = phase_correction(orthogonal_overlap)
    return final, phases*phases2







class SHARC_NUMDIFF(SHARC_HYBRID):

    _version = version
    _versiondate = versiondate
    _changelogstring = changelogstring

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Update resource keys
        self.QMin.resources.update(
            {
                "use_all_cores_for_ref" : True,
            }
        )
        self.QMin.resources.types.update(
            {
                "use_all_cores_for_ref" : bool,
            }
        )

        # Add template keys
        self.QMin.template.update(
            {
                "qm-program"            :   None,
                "qm-dir"                :   None,             # that's where the QM template/resource are
                "numdiff_method"        :   "central-diff",   # or "central-quad"
                'numdiff_representation':   "adiabatic",      # or "diabatic"
                "numdiff_stepsize"      :   0.01,             # TODO: should be a list of displacements per-DOF
                "coord_type"            :   "cartesian",      # or 'displacement' -> 'normal_modes'
                "normal_modes_file"     :   None,
                "whitelist"             :   [],
            }
        )
        self.QMin.template.types.update(
            {
                "qm-program"            :   str,
                "qm-dir"                :   str,
                "numdiff_method"        :   str,
                "numdiff_representation":   str,
                "numdiff_stepsize"      :   float,
                "coord_type"            :   str,
                "normal_modes_file"     :   None,
                "whitelist"             :   list,
            }
        )

    @ staticmethod
    def authors() -> str:
        return 'Nicolai Machholdt Høyer and Sebastian Mai'

    @staticmethod
    def version():
        return SHARC_NUMDIFF._version
    
    @ staticmethod
    def versiondate():
        return SHARC_NUMDIFF._versiondate
    
    @staticmethod
    def name() -> str:
        return "NUMDIFF"
    
    @staticmethod
    def description():
        return "   HYBRID interface for numerical derivatives (grad, NACdr, SOCdr, DMdr)"
    
    @ staticmethod
    def changelogstring():
        return SHARC_NUMDIFF._changelogstring
    


# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------

    def get_features(self, KEYSTROKES: TextIOWrapper | None = None):
        """return availble features

        ---
        Parameters:
        KEYSTROKES: object as returned by open() to be used with question()
        """
        # TODO: if not self._read_template then ask for path and read the template
        # TODO: instantiate reference child and get features
        # check what differentiation is possible and add to features
        # remember which requests require differentiation and which are done by reference child

        # if in setup mode, ask for the template file, read it, and instantiate a child to get features
        # if not in setup mode, reference child is already there
        if not self._read_template:
            self.template_file =  question(
                "Please specify the path to your NUMDIFF.template file", 
                str, 
                KEYSTROKES=KEYSTROKES, 
                default="NUMDIFF.template"
            )
            self.read_template(self.template_file)
        if not hasattr(self,"ref_interface"):
            qm_program = self.QMin.template['qm-program']
            self.ref_interface = self._load_interface(qm_program)()
            # self.ref_interface.QMin.molecule['states'] = self.QMin.molecule['states']
            if isinstance(self.ref_interface, SHARC_HYBRID):
                self.log.error('Currently, Hybrid interfaces cannot be used as children of SHARC_NUMDIFF.py')
                raise NotImplementedError
        

        ref_features = self.ref_interface.get_features(KEYSTROKES=KEYSTROKES)
        needed = {'grad': set(['h']),
                  'socdr': set(['soc']),
                  'dmdr': set(['dm']),
                  'nacdr': set(['h','overlap']),
                  }
        if self.QMin.template["numdiff_representation"] == 'diabatic':
            for i in needed:
                needed[i].add('overlap')
        possible = set()
        for i in needed:
            if all([ j in ref_features for j in needed[i] ]):
                possible.add(i)
        qm_features = ref_features.union(possible)

        # NUMDIFF cannot displace point charges
        not_supported = {'point_charges'}
        qm_features -= not_supported

        # Make QM features into a set a return these
        self.log.debug(qm_features) # log features
        return set(qm_features)



    def get_infos(self, INFOS, KEYSTROKES: Optional[TextIOWrapper] = None) -> dict:
        """communicate requests from setup and asks for additional paths or info

        The `INFOS` dict holds all global informations like paths to programs
        and requests in `INFOS['needed_requests']`

        all interface specific information like additional files etc should be stored
        in the interface intance itself.

        use the `question()` function from the `utils` module and write the answers
        into `KEYSTROKES`

        Parameters:
        ---
        INFOS
            dict[str]: dictionary with all previously collected infos during setup
        KEYSTROKES
            str: object as returned by open() to be used with question()
        """
        # Setup some output to log
        self.log.info("=" * 80)
        self.log.info(f"{'||':<78}||")
        self.log.info(f"||{'NUMDIFF interface setup':^76}||\n{'||':<78}||")
        self.log.info("=" * 80)
        self.log.info("\n")

        # Interactive setting of file
        # TODO: add options
        if question("Do you have an NUMDIFF.resources file?", bool, KEYSTROKES=KEYSTROKES, autocomplete=False, default=False):
            self.resources_file = question("Specify path to NUMDIFF.resources", str, KEYSTROKES=KEYSTROKES, autocomplete=True)
        else:
            self.log.info(f"{'NUMDIFF Ressource usage':-^60}\n")
            self.log.info(
                """Please specify the number of CPUs to be used by EACH trajectory.
        """
            )
            self.setupINFOS["ncpu_numdiff"] = abs(question("Number of CPUs:", int, KEYSTROKES=KEYSTROKES)[0])
            self.setupINFOS["scratchdir_numdiff"] = question("Path to scratch directory:", str, KEYSTROKES=KEYSTROKES)
            self.setupINFOS["scratchdir_numdiff"] += '/$$/'

            # TODO: could use schedule scaling and Amdahl, but SHARC_HYBRID does not have it

        # if we need overlaps, we need to modify the INFOS['needed_requests'] to tell children to prepare for that
        needed_copy = deepcopy(INFOS['needed_requests'])
        if self.QMin.template["numdiff_representation"] == 'diabatic' or "nacdr" in INFOS['needed_requests']:
            INFOS['needed_requests'].add('overlap')
        
        # Get the infos from the child
        self.log.info(f"{' Setting up QM-interface ':=^80s}\n")
        self.ref_interface.get_infos(INFOS, KEYSTROKES=KEYSTROKES)

        # reset the needed requests
        INFOS['needed_requests'] = needed_copy

        return INFOS



    def prepare(self, INFOS: dict, dir_path: str):
        """
        prepares the folder for an interface calculation

        Parameters
        ----------
        INFOS
            dict[str]: dictionary with all infos from the setup script
        dir_path
            str: *relative* path to the directory to setup (can be appended to `scratchdir`)
        """
        # Copy files to the nummdiff dir
        shutil.copy(self.template_file, os.path.join(dir_path, self.name() + ".template"))
        # shutil.copy(self.template_file, os.path.join(dir_path, self.name() + ".resources"))

        # write resource file
        string = 'ncpu %i\nscratchdir %s\nuse_all_cores_for_ref True\n' % (self.setupINFOS['ncpu_numdiff'], self.setupINFOS["scratchdir_numdiff"])
        writefile(os.path.join(dir_path, self.name() + ".resources"), string)

        # Setup sub-dir for the QM calcs
        if self.QMin.template['qm-dir'] is None:
            raise ValueError("Keyword 'qm-dir' not found in template file!")
        qmdir = dir_path + f"/{self.QMin.template['qm-dir']}"
        mkdir(qmdir)

        # Make savedir and scratchdir for the reference interface
        if not self.QMin.save["savedir"]:
            self.log.warning(
                "savedir not specified in QM.in, setting savedir to current directory!"
            )
            self.QMin.save["savedir"] = os.getcwd()

        ref_savedir = os.path.join(dir_path, self.QMin.save["savedir"], 'QM_' + self.QMin.template["qm-program"].upper())
        self.log.debug(f"ref_savedir {ref_savedir}")
        # if not os.path.isdir(ref_savedir):
        #     mkdir(ref_savedir)
        
        ref_scratchdir = os.path.join(self.QMin.resources["scratchdir"],   'QM_' + self.QMin.template["qm-program"].upper())
        self.log.debug(f"ref_scratchdir {ref_scratchdir}")
        # if not os.path.isdir(ref_scratchdir):
        #     mkdir(ref_scratchdir)

        self.ref_interface.QMin.save["savedir"] = ref_savedir
        self.ref_interface.QMin.resources["scratchdir"] = ref_scratchdir

        # if we need overlaps, we need to modify the INFOS['needed_requests'] to tell children to prepare for that
        needed_copy = deepcopy(INFOS['needed_requests'])
        if self.QMin.template["numdiff_representation"] == 'diabatic' or "nacdr" in INFOS['needed_requests']:
            INFOS['needed_requests'].add('overlap')
        
        # Call prepare for the reference interface
        self.ref_interface.prepare(INFOS, qmdir)

        # reset the needed requests
        INFOS['needed_requests'] = needed_copy

        return



# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------







    def read_template(self, template_file="NUMDIFF.template") -> None:
        # Call the read_template() from the base (Simply reads all entries in the .template file and adds them to the self.Qmin.template)
        super().read_template(template_file)
        
        # If we use normal mode coordinates we read them in here
        if self.QMin.template["coord_type"] == "normal_modes":
            self.read_displacement_coordinates(os.path.abspath(self.QMin.template["normal_modes_file"]))
            # TODO: change to V0.txt format!

        # Sanitize (should be setup in another way I guess)
        # self.QMin.template["numdiff_stepsize"] = float(self.QMin.template["numdiff_stepsize"])

        # Set _read_template to True
        self._read_template = True
        return


    def read_resources(self, resources_file="NUMDIFF.resources") -> None:
        # TODO: only uses scratchdir currently. Anything else? ncpu/memory?
        # Do we need to throw a RunTimeError if the resource file is not there?
        # Search for resource file
        if not os.path.isfile(resources_file):
            err_str = f"File '{resources_file}' not found!"
            self.log.error(err_str)
            raise RuntimeError(err_str)
        
        # Call the read_resources() from the base
        super().read_resources(resources_file)
        self._read_resources = True
        
        return



    def read_displacement_coordinates(self, disp_coord_filename):
        # TODO: replace with V0.txt format
        # Need to decide what to do if the current geometry does not match with the reference geometry
        # I guess we should displace in Q and then transform...

        # Read the coord file
        disp_coords = []
        with open(disp_coord_filename, 'r') as f:
            for line in f:
                if "units" in line:
                    line = f.readline()
                if "normal modes" in line:
                    line = f.readline()
                    n_coords = int(line)
                    for i_coord in range(n_coords):
                        line = f.readline()
                        line = line.split()
                        disp_coords.append(np.resize(np.asarray(line, dtype=float), (self.QMin.molecule["natom"], 3)))

        self.QMin.disp_coords = disp_coords
        return




# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------




    def setup_interface(self) -> None:
        """
        Prepare the interface for calculations
        """
        # --- reference child ---

        # paths
        self.QMin.resources["scratchdir"] = os.path.abspath(os.path.expanduser(os.path.expandvars(self.QMin.resources["scratchdir"])))
        if self.QMin.template['qm-dir'] is None:
            raise ValueError("Keyword 'qm-dir' not found in template file!")
        self.qmdir = os.path.abspath(os.path.expanduser(os.path.expandvars(self.QMin.template['qm-dir'])))
        qm_program = self.QMin.template['qm-program']

        # Create reference child
        ref_logname = "Reference:%s" % qm_program
        pwd = os.path.join(self.QMin.resources["scratchdir"],'PWD','reference')
        mkdir(pwd)
        ref_logfile = os.path.join(pwd,'QM.log')
        self.ref_interface = self._load_interface(qm_program)(logfile = ref_logfile, logname=ref_logname, loglevel = self.log.level, persistent = False)
        if isinstance(self.ref_interface, SHARC_HYBRID):
            self.log.error('Currently, Hybrid interfaces cannot be used as children of SHARC_NUMDIFF.py')
            raise NotImplementedError

        # do setup molecule
        self.ref_interface.setup_mol(self.QMin)
        
        ## then do setup_mol/template/resources
        with InDir(self.qmdir):
            self.ref_interface.read_resources()
            self.ref_interface.read_template()
            ## reassign scratchdir and savedir
            scratchdir = os.path.join(self.QMin.resources["scratchdir"],'SCRA','reference')
            savedir = os.path.join(self.QMin.save["savedir"],'children','reference')
            mkdir(scratchdir)
            if not os.path.isdir(savedir):
                mkdir(savedir)
            self.ref_interface.QMin.resources['scratchdir'] = scratchdir
            self.ref_interface.QMin.save['savedir'] = savedir
            self.ref_interface.QMin.resources['pwd'] = pwd
            self.ref_interface.QMin.resources['cwd'] = pwd
            if self.QMin.resources["use_all_cores_for_ref"]:
                self.ref_interface.QMin.resources["ncpu"] = self.QMin.resources["ncpu"]
            self.ref_interface.setup_interface()
        
        # --- kindergarden ---

        # figure out how many children we have in the kindergarden and make labels
        labels = []
        match self.QMin.template["coord_type"]:
            case "cartesian":
                labels.append( ["cartesian"] )
                labels.append( [ i for i in range(self.QMin.molecule["natom"]) ] )
                labels.append( [ "x", "y", "z"] )
            case "normal_modes":
                labels.append( ["normal_modes"] )
                labels.append( [ i for i in range(len(self.QMin.disp_coords)) ] )
                raise NotImplementedError
            case _:
                raise RuntimeError(f"Input 'coord_type': {self.QMin.template['coord_type']} is not valid.")
        match self.QMin.template["numdiff_method"]:
            case 'central-diff':
                labels.append( ['p', 'n'] )
            case 'central-quad':
                labels.append( ['pp', 'p', 'n', 'nn'] )
            case _:
                raise RuntimeError(f"Input 'numdiff_method': {self.QMin.template['numdiff_method']} is not valid.") 
        # self.log.info(labels) 
        # make full labels as direct product of the labels:
        full_labels = list(itertools.product(*labels))
        # self.log.info(full_labels)


        # make child_dict: define logfiles
        child_dict = {}
        for label in full_labels:
            name = '_'.join(str(i) for i in label)
            pwd = os.path.join(self.QMin.resources["scratchdir"],'PWD',name)
            mkdir(pwd)
            logfile = os.path.join(pwd,'QM.log')
            logname = 'Displacement:%s:%s' % (qm_program,name)
            child_dict[label] = (qm_program, [], {"logfile": logfile, "logname": logname, 'loglevel': self.log.level, 'persistent': False})
        #self.log.info(child_dict)
        self.instantiate_children(child_dict)
        

        # do full setup for all children
        for label in self._kindergarden:
            child = self._kindergarden[label]
            # self.log.info(label)
            name = '_'.join(str(i) for i in label)
            child.setup_mol(self.QMin)
            with InDir(self.qmdir):
                child.read_resources()
                child.read_template()
                # scratch
                scratchdir = os.path.join(self.QMin.resources["scratchdir"],'SCRA',name)
                mkdir(scratchdir)
                child.QMin.resources['scratchdir'] = scratchdir
                # save
                savedir = os.path.join(self.QMin.save["savedir"],'children',name)
                mkdir(savedir)
                child.QMin.save['savedir'] = savedir
                # pwd
                pwd = os.path.join(self.QMin.resources["scratchdir"],'PWD',name)
                child.QMin.resources['pwd'] = pwd
                child.QMin.resources['cwd'] = pwd
                child.setup_interface()

        # --- feature setup ---

        # get possible features
        ref_features = self.ref_interface.get_features()
        own_features = self.get_features()
        self.log.info(ref_features)
        self.log.info(own_features)
        self.do_numerically = set()
        check_these = ['grad', 'socdr', 'dmdr', 'nacdr']
        for i in check_these:
            if i in self.QMin.template["whitelist"]:
                if i in ref_features:
                    self.log.info('Request %s white-listed and available from child, will be passed to reference' % i)
                else:
                    self.do_numerically.add(i)
                    self.log.info('Request %s white-listed but not available from child, will be done numerically' % i)
            elif i in own_features:
                self.do_numerically.add(i)
                self.log.info('Request %s will be done numerically' % i)
            else:
                self.log.info('Request %s not available' % i)

        return


# ----------------------------------------------------------------------------------------------

    # def _step_logic(self):
    #     super()._step_logic()
    #     self.ref_interface._step_logic()

    def write_step_file(self):
        super().write_step_file()
        self.ref_interface.write_step_file()
        
    # def update_step(self, step: int = None):
    #     super().update_step(step)
    #     self.ref_interface.update_step(step)

    def clean_savedir(self):
        super().clean_savedir()
        self.ref_interface.clean_savedir()

    def create_restart_files(self):
        self.ref_interface.create_restart_files()

# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------

    def run(self) -> None:
        """
        """

        # --- reference child calculation ---

        # coordinates
        self.ref_interface.QMin.coords["coords"] = self.QMin.coords["coords"].copy()
    
        # requests
        self.all_requests = {k: v for (k, v) in self.QMin.requests.items() if v is not None}
        self.ref_requests = {}
        self.num_requests = {}
        for key, value in self.all_requests.items():
            if key in self.do_numerically:
                if bool(value):
                    self.num_requests[key] = value
            else:
                self.ref_requests[key] = value
        for key, value in self.ref_requests.items():
            self.ref_interface.QMin.requests[key] = value

        # inherit current step
        self.ref_interface.QMin.save["step"] = self.QMin.save["step"]
        self.ref_interface._step_logic()
        self.ref_interface._request_logic()
        
        # mandatory requests
        self.ref_interface.QMin.requests['nooverlap'] = False
        self.ref_interface.QMin.requests['cleanup'] = False

        # run the child
        with InDir(self.ref_interface.QMin.resources['pwd']):
            self.ref_interface.run()
            self.ref_interface.getQMout()
            self.ref_interface.write_step_file()

        # --- displaced child calculations ---

        # figure out whether to do displacements
        do_numerically = bool(self.num_requests)

        # go ahead and set up children
        if do_numerically:
            self.log.info('Doing numerical differentiation for requests:')
            self.log.info(self.num_requests)

            # set coordinates of all kindergardners, based on their labels
            cart_directions = {"x": 0, "y": 1, "z": 2}
            displacements = {"pp": +2., "p": +1., "n": -1., "nn": -2.}          # for other differentiation than central, new labels are needed, e.g., "pp" or "nn"
            for label in self._kindergarden:
                match label[0]:
                    case "cartesian":
                        iatom, idir, idisp = label[1:4]
                        idir = cart_directions[idir]
                        idisp = displacements[idisp]
                        coords = self.QMin.coords["coords"].copy()
                        coords[iatom,idir] += idisp * self.QMin.template["numdiff_stepsize"]
                    case "normal_modes":
                        raise NotImplementedError
                        # # Normal coordinates => This is old code from Nikolai for whenever we implement normal mode displacements
                        # elif self.QMin.template["coord_type"] == "normal_modes":
                        #     dispmat = self.QMin.disp_coords[displacement_index]
                        #     if numdiff_debug:
                        #         print("Input structure:")
                        #         print(coords["coords"])
                        #         print("Displacement coordinate:")
                        #         print(dispmat)
                        #         print("Displacement step size:")
                        #         print(self.QMin.template["numdiff_stepsize"])
                        #     # Displace the coordinate
                        #     if   displacement_sign == '+':
                        #         coords["coords"] = coords["coords"] + self.QMin.template["numdiff_stepsize"] * dispmat
                        #     elif displacement_sign == '-':
                        #         coords["coords"] = coords["coords"] - self.QMin.template["numdiff_stepsize"] * dispmat
                        #     else:
                        #         raise RuntimeError("Argument 'displacement_sign' is invalid in call to 'displace_coordinates'. Argument must be '+' or '-'.")
                        #     if numdiff_debug:
                        #         print("Displaced structure:")
                        #         print(coords["coords"])
                self._kindergarden[label].QMin.coords["coords"] = coords
            
            # set requests
            child_requests = {}
            for req in self.num_requests:
                match req:
                    case "grad":
                        child_requests["h"] = True
                    case "socdr":
                        child_requests["soc"] = True
                    case "dmdr":
                        child_requests["dm"] = True
                    case "nacdr":
                        child_requests["h"] = True
                        child_requests["overlap"] = True
            if self.QMin.template["numdiff_representation"] == 'diabatic':
                child_requests["overlap"] = True
            for label in self._kindergarden:
                for key, value in child_requests.items():
                    self._kindergarden[label].QMin.requests[key] = value
            
            # take savedir of reference child and copy to all displaced children and set step
            for label in self._kindergarden:
                # delete savedir content
                cleandir(self._kindergarden[label].QMin.save['savedir'])
                # copy all files from reference child to displaced child
                ls = os.listdir(self.ref_interface.QMin.save['savedir'])
                for f in ls:
                    base, ext = os.path.splitext(f)
                    if not ext[1:].isdigit():
                        continue
                    if not int(ext[1:]) == self.ref_interface.QMin.save['step']:
                        continue
                    fromfile = os.path.join(self.ref_interface.QMin.save['savedir'],f)
                    tofile = os.path.join(self._kindergarden[label].QMin.save['savedir'],f)
                    shutil.copy(fromfile,tofile)
                # TODO: extra copy rules for LEGACY interface
                if self.QMin.template['qm-program'].upper() == "LEGACY":
                    for f in ["MOLPRO", "COLUMBUS", "ADF_AMS", "BAGEL"]:
                        fromdir = os.path.join(self.ref_interface.QMin.save['savedir'],f)
                        if os.path.isdir(fromdir):
                            self.log.info("Copying subdirectories for LEGACY interface")
                            todir = os.path.join(self._kindergarden[label].QMin.save['savedir'],f)
                            shutil.copytree(fromdir, todir)
                # set step for displaced child
                self._kindergarden[label].QMin.save['step'] = self.ref_interface.QMin.save['step'] + 1
                stepfile = os.path.join(self._kindergarden[label].QMin.save["savedir"], "STEP")
                writefile(stepfile, str(self.ref_interface.QMin.save['step']))
                self._kindergarden[label]._step_logic()
                self._kindergarden[label]._request_logic()
            
            # run the children
            t1 = datetime.datetime.now()
            self.log.info('\nSTART:\t%s' % (t1))
            self.run_children(self.log, 
                              self._kindergarden, 
                              self.QMin.resources['ncpu'])
            t2 = datetime.datetime.now()
            self.log.info('FINISH:\t%s\tRuntime: %s\n' % (t2, t2 - t1))
        

# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------

    def getQMout(self) -> dict[str, np.ndarray]:
        """
        Return QMout object
        """

        # create empty QMout
        self.QMout.allocate(
            self.QMin.molecule["states"],
            self.QMin.molecule["natom"],
            self.QMin.molecule["npc"],
            self.QMin.requests,
        )

        # Set QMout with stuff from the reference calculation
        for key,val in self.ref_interface.QMout.items():
            self.QMout[key] = deepcopy(val)
        self.QMout.charges = [0 for i in self.QMin.molecule["states"]]  # TODO: remove later

        # do all the numerical requests
        do_numerically = bool(self.num_requests)
        if do_numerically:

            # make phase corrections if overlaps are present
            any_child = self._kindergarden[next(iter(self._kindergarden))]
            if any_child.QMin.requests['overlap']:
                self.log.info('Doing phase correction ...')
                for label in self._kindergarden:
                    self._kindergarden[label].QMout['overlap'], phases = post_process_overlap_matrix(self._kindergarden[label].QMout['overlap'])
                    phases2 = phases[:,None] * phases[None,:]
                    if self._kindergarden[label].QMout['h'] is not None:
                        self._kindergarden[label].QMout['h'] *= phases2
                    if self._kindergarden[label].QMout['dm'] is not None:
                        self._kindergarden[label].QMout['dm'] *= phases2[None,:,:]
            
            # preparation 
            cart_directions = {"x": 0, "y": 1, "z": 2}
            displacements = {"pp": +2., "p": +1., "n": -1., "nn": -2.} 
            stepsize = self.QMin.template["numdiff_stepsize"]
            nstates = self.QMin.molecule['nmstates']

            # compute derivatives
            match self.QMin.template['coord_type']:
                case "cartesian":
                    for iatom in range(self.QMin.molecule["natom"]):
                        for idir in ["x","y","z"]:

                            # pick the involved children for this direction
                            # the run() function would decide how many children per direction, depending on numdiff_method
                            # and here we only pick those that correspond to the current direction
                            children = {}
                            for label in self._kindergarden:
                                if ("cartesian", iatom, idir) == label[:-1]:
                                    children[label[-1]] = self._kindergarden[label]

                            # the trafo matrices for this direction
                            match self.QMin.template["numdiff_representation"]:
                                case "adiabatic":
                                    S = {}
                                    for label in children:
                                        S[label] = np.identity(nstates)
                                case "diabatic":
                                    S= {}
                                    for label in children:
                                        S[label] = children[label].QMout["overlap"]

                            # go through the requests for this direction
                            for request in self.num_requests:

                                # pick quantity for this request
                                match request:
                                    case "grad": # differentiate the energies, i.e., the diagonal elements of the Hamiltonian
                                        A = {}
                                        for label in children:
                                            A[label] = np.diag(np.diag(children[label].QMout["h"]))
                                    case "socdr": # differentiate the off-diagonal elements of the Hamiltonian
                                        A = {}
                                        for label in children:
                                            A[label] = children[label].QMout["h"] - np.diag(np.diag(children[label].QMout["h"]))
                                    case "dmdr": # differentiate the dipole moment matrix
                                        A = {}
                                        for label in children:
                                            A[label] = children[label].QMout["dm"]
                                    case "nacdr": # NACs are a bit more complicated
                                        match self.QMin.template["numdiff_representation"]:
                                            case "adiabatic": # differentiate the overlap matrix elements 
                                                A = {}
                                                for label in children:
                                                    A[label] = children[label].QMout["overlap"]
                                            case "diabatic": # differentiate the diabatized diagonal of the Hamiltonian
                                                A = {}
                                                for label in children:
                                                    A[label] = np.diag(np.diag(children[label].QMout["h"]))

                                # make the transformation and differentiation for this request and direction
                                # if other differentiation schemes will be implemented, here one can simply 
                                # use them based on their labels. In this way, this is the only block that depends on numdiff_method
                                match self.QMin.template['numdiff_method']:
                                    case "central-diff":
                                        numerator = S['p'].T@A['p']@S['p'] - S['n'].T@A['n']@S['n']
                                        denomimator = stepsize * (displacements['p'] - displacements['n'])
                                        result = numerator / denomimator
                                    case "central-quad":
                                        numerator = -S['pp'].T@A['pp']@S['pp'] + 8.*S['p'].T@A['p']@S['p'] - 8.*S['n'].T@A['n']@S['n'] + S['nn'].T@A['nn']@S['nn']
                                        denomimator = 6. * stepsize * (displacements['p'] - displacements['n'])
                                        result = numerator / denomimator
                                    case _:
                                        raise NotImplementedError("Only central differences implemented")
                                    
                                # assign correctly the resulting request elements
                                match request:
                                    case "grad":  # QMout['grad'] has shape (nstates, natom, 3)
                                        self.QMout['grad'][:,iatom,cart_directions[idir]] = np.diag(result)
                                    case "socdr": # QMout['socdr'] has shape (nstates, nstates, natom, 3)
                                        self.QMout['socdr'][:,:,iatom,cart_directions[idir]] = result
                                    case "dmdr":  # QMout['socdr'] has shape (3, nstates, nstates, natom, 3)
                                        self.QMout['dmdr'][:,:,:,iatom,cart_directions[idir]] = result
                                    case "nacdr": # QMout['socdr'] has shape (nstates, nstates, natom, 3)
                                        match self.QMin.template["numdiff_representation"]:
                                            case "adiabatic":
                                                # we only make properly anti-Hermitian
                                                self.QMout['nacdr'][:,:,iatom,cart_directions[idir]] = (result - result.T)/2.
                                            case "diabatic": 
                                                # result contains dH/dR. To get the NAC, we need to scale by the energy gaps
                                                result = (result + result.T)/2. # maybe unnecessary
                                                E = np.diag(self.QMout['h'])
                                                denominator = E[:, None] - E[None, :]
                                                denominator[np.diag_indices_from(denominator)] = np.inf
                                                denominator[denominator == 0.] = np.inf
                                                self.QMout['nacdr'][:,:,iatom,cart_directions[idir]] = result / denominator
                case "normal_modes":
                    raise NotImplementedError("Normal mode displacements not allowed")
                    # TODO: probably the same as for Cartesian, but afterwards we have to do a coordinate transformation of all derivatives
                    # because the interface should always return Cartesian derivatives
                case _:
                    raise NotImplementedError("Only Cartesian displacements allowed")
        
        self.QMout.runtime = self.clock.measuretime()
        return self.QMout

# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------










if __name__ == "__main__":
    from logger import loglevel
    try:
        num_diff = SHARC_NUMDIFF(loglevel=loglevel)
        num_diff.main()
    except KeyboardInterrupt:
        print("\nCTRL+C makes me a sad SHARC ;-(")
