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
import datetime
import os
import shutil
from copy import deepcopy
from io import TextIOWrapper
import numpy as np

# internal
from constants import ATOMCHARGE, FROZENS
from factory import factory
from SHARC_HYBRID import SHARC_HYBRID
from SHARC_INTERFACE import SHARC_INTERFACE
from utils import ATOM, InDir, itnmstates, mkdir, question, readfile, expand_path
from qmout import formatcomplexmatrix

version = '1.0'
versiondate = datetime.datetime(2023, 11, 3)

changelogstring = '''
'''
np.set_printoptions(linewidth=400)

numdiff_debug = False


def phase_correction(matrix):
    """
    Do a phase correction of a matrix.
    Follows algorithm from J. Chem. Theory Comput. 2020, 16, 2, 835-846 (https://doi.org/10.1021/acs.jctc.9b00952)
    """
    U = matrix.real.copy()
    det_U = np.linalg.det(U)
    if det_U < 0:
        U[:, 0] *= -1.0  # this row/column convention is correct
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
                    done = False
        sweeps += 1
    
    if numdiff_debug:
        print(f"Finished phase correction after {sweeps} sweeps.")

    return U


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
    phase_corrected_overlap = phase_correction(overlap_matrix)

    # Do a Loewdin orthonormalization
    orthogonal_overlap = loewdin_orthonormalization(phase_corrected_overlap)

    # Extra phase correction (probably not needed)
    return phase_correction(orthogonal_overlap)


class SHARC_NUMDIFF(SHARC_HYBRID):

    _version = version
    _versiondate = versiondate
    _changelogstring = changelogstring

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Update resource keys
        self.QMin.resources.update(
            {
            }
        )
        self.QMin.resources.types.update(
            {
            }
        )

        # Add template keys
        self.QMin.template.update(
            {
                "qm-program"            :   None,
                "numdiff-method"        :   "central-diff",
                "numdiff-stepsize"      :   0.01,
                "coord-type"            :   "cartesian",
                "displacement-coords"   :   None,
                "properties"            :   None,
                "qm-dir"                :   None,
            }
        )
        self.QMin.template.types.update(
            {
                "qm-program"            :   str,
                "numdiff-method"        :   str,
                "numdiff-stepsize"      :   float,
                "coord-type"            :   str,
                "displacement-coords"   :   str,
                "properties"            :   list,
                "qm-dir"                :   str,
            }
        )

    @ staticmethod
    def authors() -> str:
        return 'Nicolai Machholdt Høyer'

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
        return "Hybrid interface for numerical differentiation."
    
    @ staticmethod
    def changelogstring():
        return SHARC_NUMDIFF._changelogstring
    
    def get_features(self, KEYSTROKES: TextIOWrapper):
        """return availble features

        ---
        Parameters:
        KEYSTROKES: object as returned by open() to be used with question()
        """
        # Some interactive setting of files?
        self.template_file = question(
            "Please specify the path to your NUMDIFF.template file",
            str,
            KEYSTROKES=KEYSTROKES,
            default="NUMDIFF.template")
        
        # Read in info on the NUMDIFF and QM calculations
        self.read_template(self.template_file)

        # Load features from QM calc
        qm_features = self.ref_interface.get_features()

        # Make QM features into a set a return these
        self.log.debug(qm_features) # log features
        return set(qm_features)

    def get_infos(self, INFOS, KEYSTROKES: [TextIOWrapper] = None) -> dict:
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
        self.log.info(f"||{'NUMDIFF interface setup':=^76}||\n{'||':<78}||")
        self.log.info("=" * 80)
        self.log.info("\n")

        # Interactive setting of file
        if question("Do you have an NUMDIFF.resources file?", bool, KEYSTROKES=KEYSTROKES, autocomplete=False, default=False):
            self.resources_file = question("Specify path to NUMDIFF.resources", str, KEYSTROKES=KEYSTROKES, autocomplete=True)

        # Get the QM infos
        self.log.info(f"{' Setting up QM-interface ':=^80s}\n")
        self.ref_interface.get_infos(INFOS, KEYSTROKES=KEYSTROKES)

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
        shutil.copy(self.template_file, os.path.join(dir_path, self.name() + ".resources"))

        # Setup sub-dir for the QM calcs
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
        if not os.path.isdir(ref_savedir):
            mkdir(ref_savedir)
        
        ref_scratchdir = os.path.join(self.QMin.resources["scratchdir"],   'QM_' + self.QMin.template["qm-program"].upper())
        self.log.debug(f"ref_scratchdir {ref_scratchdir}")
        if not os.path.isdir(ref_scratchdir):
            mkdir(ref_scratchdir)

        self.ref_interface.QMin.save["savedir"] = ref_savedir
        self.ref_interface.QMin.resources["scratchdir"] = ref_scratchdir

        # Call prepare for the reference interface
        self.ref_interface.prepare(INFOS, qmdir)

        return


    def create_restart_files(self) -> None:
        """
        Create restart files
        """
        return


    def read_resources(self, resources_file="NUMDIFF.resources") -> None:
        # Search for resource file
        if not os.path.isfile(resources_file):
            err_str = f"File '{resources_file}' not found!"
            self.log.error(err_str)
            raise RuntimeError(err_str)
        
        # Call the read_resources() from the base
        super().read_resources(resources_file)
        self._read_resources = True
        
        return


    def read_template(self, template_file="NUMDIFF.template") -> None:
        # Call the read_template() from the base (Simply reads all entries in the .template file and adds them to the self.Qmin.template)
        super().read_template(template_file)
        
        # If we use displacement coordinates we read them in here
        if self.QMin.template["coord-type"] == "displacement":
            self.read_displacement_coordinates(os.path.abspath(self.QMin.template["displacement-coords"]))

        # Sanitize (should be setup in another way I guess)
        self.QMin.template["numdiff-stepsize"] = float(self.QMin.template["numdiff-stepsize"])

        # Set _read_template to True
        self._read_template = True
        return


    def read_displacement_coordinates(self, disp_coord_filename):
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


    def setup_interface(self) -> None:
        """
        Prepare the interface for calculations
        """
        # Set the number of degrees of freedom (only cartesian for now)
        if self.QMin.template["coord-type"] == "cartesian":
            self.QMin.molecule["nDoF"] = 3 * self.QMin.molecule["natom"]
        elif self.QMin.template["coord-type"] == "displacement":
            self.QMin.molecule["nDoF"] = len(self.QMin.disp_coords)
        else:
            raise RuntimeError(f"Input 'coord-type': {self.QMin.template['coord-type']} is not a valid coordinate type to use with SHARC_NUMDIFF")


        # Create statemap
        self.QMin.maps["statemap"] = {
            i + 1: [*v]
            for i, v in enumerate(itnmstates(self.QMin.molecule['states']))
        }
        
        # Create instance of the QM interface to use as the reference
        ref_logname = f"Reference_structure_{self.QMin.template['qm-program']}"
        self.ref_interface = self.construct_qm_interface(ref_logname)

        return


    def read_requests(self, requests_file: str = "QM.in") -> None:
        """
        """
        super().read_requests(requests_file)
        # Also read requests for ref_interface
        self.ref_interface.read_requests(requests_file)
        # Save the request filename for use with the displaced interfaces
        self.requests_file = requests_file

        # Check that ref_interface calculates the neccesary properties and update the QMin.request to include derivatives of the numdiff properties
        for property in self.QMin.template["properties"]:
            if property == "h" or property == "soc":
                self.QMin.requests["socdr"] = True
            elif property == "dm":
                self.QMin.requests["dmdr"] = True
            elif property == "overlap":
                self.QMin.requests["nacdr"] = [i + 1 for i in range(self.QMin.molecule["nmstates"])]

            if not self.ref_interface.QMin.requests[property] and property != "overlap":
                error_string = f"The property {property} is not requested in the QM calculation."
                self.log.error(error_string)
                raise RuntimeError(error_string)
        return


    def run(self) -> None:
        """
        Do request & other logic and calculations here
        """
        # Run reference SP
        self.ref_interface.run()
        # Get reference value
        self.ref_interface.getQMout()

        # Create displaced SPs
        self.displaced_interfaces = self.create_displaced_interfaces()

        # Run all displaced SPs
        for i_coord_interfaces in self.displaced_interfaces:
            for displaced_interface in i_coord_interfaces:
                displaced_interface.run()
                displaced_interface.getQMout()

                # We need to ensure correct phases and orthonormality of the overlap matrices
                if "overlap" in self.QMin.template["properties"]:
                    displaced_interface.QMout.overlap = post_process_overlap_matrix(displaced_interface.QMout.overlap)


        # Create map to hold derivatives for each requested property
        self.derivatives = dict()

        # Loop over properties to differentiate
        for property in self.QMin.template["properties"]:
            # Do numerical differentiation
            if numdiff_debug:
                print(f"Doing NUMDIFF for {property}")
            self.do_numerical_diff(property)

        return


    def getQMout(self) -> dict[str, np.ndarray]:
        """
        Return QMout object
        """
        # Set QMout with stuff from the reference calculation
        self.QMout = self.ref_interface.QMout

        # Set calculated derivatives
        if "h" in self.QMin.template["properties"]:
            self.QMout["socdr"] = self.derivatives["h"] # The socs are included as off-diagonals if the electronic structure has calculated these
        if "dm" in self.QMin.template["properties"]:
            self.QMout["dmdr"] = self.derivatives["dm"]
        if "overlap" in self.QMin.template["properties"]:
            self.QMout["nacdr"] = self.derivatives["overlap"]
            
        return


    def formatQMout(self) -> str:
        """
        Output calculated properties from the reference QM calculations
        and all calculated derivatives.
        """

        # First format QMout data from the reference interface
        QMout_format_string = self.ref_interface.QMout.formatQMout(self.ref_interface.QMin, DEBUG=self._DEBUG)
        """
        # Now add numerical differentiation results
        QMout_format_string += "==> Numerical derivatives:\n\n"

        # Loop over all properties
        for property in self.QMin.template["properties"]:
            QMout_format_string += f"=> Derivative {property}\n"
            # Loop over degrees of freedom
            for i_coord in range(self.QMin.molecule["nDoF"]):
                # Add header
                QMout_format_string += f"displacement coordinate {i_coord}:\n"
                # Add matrix
                QMout_format_string += formatcomplexmatrix(self.QMout.derivatives[property][i_coord], self.QMin.molecule["states"])
                QMout_format_string += "\n"
        """

        return QMout_format_string


    def do_numerical_diff(self, property) -> None:
        """
        # Options (controlled by self.QMin.template["numdiff-method"]): 
            # Central diff:
                dO/dR = (O_{+} - O_{-})/(2 DelR)
            # "Diabatic" central diff:
                dO/dR = (S_{+}^\dagger O_{+} S_{+} - S_{-}^\dagger O_{-} S_{-})/(2 DelR)
        """
        # Bool indicating whether property has cartesian components
        has_components = (property == "dm")

        # Create array to hold derivatives for all degrees of freedom
        n_states = self.QMin.molecule["nmstates"]
        if self.QMin.template["coord-type"] == "cartesian":
            n_atoms = self.QMin.molecule["natom"]
            if has_components:
                derivatives = np.zeros((3, n_states, n_states, n_atoms, 3))
            else:
                derivatives = np.zeros((n_states, n_states, n_atoms, 3))
        elif self.QMin.template["coord-type"] == "displacement":
            n_disp = len(self.displaced_interfaces)
            if has_components:
                derivatives = np.zeros((3, n_states, n_states, n_disp))
            else:
                derivatives = np.zeros((n_states, n_states, n_disp))
        else:
            err = f"coord-type: {self.QMin.template['coord-type']} is not recognized as a valid coordinate type"
            self.log.error(err)
            raise RuntimeError(err)

        # Loop over the degrees of freedom
        for i_coord in range(len(self.displaced_interfaces)):
            # Extract overlap matrices if they are needed
            if self.QMin.template["numdiff-method"] == "diabatic-central-diff":
                overlap_plus  = self.displaced_interfaces[i_coord][0].QMout["overlap"]
                overlap_minus = self.displaced_interfaces[i_coord][1].QMout["overlap"]
            
            # Ensure we loop over the cartesian coordinates if the current property has cartesian components
            if has_components:
                n_comp = 3
            else:
                n_comp = 1
            
            for i_xyz in range(n_comp):
                # Get displaced values
                if has_components:
                    disp_plus_val  = self.displaced_interfaces[i_coord][0].QMout[property][i_xyz]
                    disp_minus_val = self.displaced_interfaces[i_coord][1].QMout[property][i_xyz]
                else:
                    disp_plus_val  = self.displaced_interfaces[i_coord][0].QMout[property]
                    disp_minus_val = self.displaced_interfaces[i_coord][1].QMout[property]

                if self.QMin.template["numdiff-method"] == "diabatic-central-diff":
                    # Transform values with the overlap matrices
                    disp_plus_val  = overlap_plus.T  @ disp_plus_val  @ overlap_plus
                    disp_minus_val = overlap_minus.T @ disp_minus_val @ overlap_minus
                
                # Calculate the derivative
                if self.QMin.template["numdiff-method"] == "central-diff" or self.QMin.template["numdiff-method"] == "diabatic-central-diff":
                    deriv = self.do_central_diff(disp_plus_val, disp_minus_val, 2*float(self.QMin.template["numdiff-stepsize"]))
                else:
                    raise RuntimeError("Other methods than central-diff has not been implemented for numdiff")

                # Save derivative in format that is compatible with other saved derivatives
                # Save format: deriv[i_state][j_state][i_atom][i_xyz]
                # Get a reference to the derivatives array for this component
                if has_components:
                    deriv_containter = derivatives[i_xyz]
                else:
                    deriv_containter = derivatives
                # Fill in values
                n_states = self.QMin.molecule["nmstates"]
                for i_state in range(n_states):
                    for j_state in range(n_states):
                        if self.QMin.template["coord-type"] == "cartesian":
                            i_atom = int(i_coord / 3)
                            i_xyz = i_coord % 3
                            deriv_containter[i_state][j_state][i_atom][i_xyz] = deriv[i_state][j_state]
                        elif self.QMin.template["coord-type"] == "displacement":
                            deriv_containter[i_state][j_state][i_coord] = deriv[i_state][j_state]         

        # Assign the derivatives
        self.derivatives[property] = derivatives
        return


    def do_central_diff(self, disp_plus, disp_minus, distance):
        """
        Calculate central difference
        (O_{-} - O_{+})/(distance)
        """
        if numdiff_debug:
            for i_state in range(self.QMin.molecule["nmstates"]):
                for j_state in range(self.QMin.molecule["nmstates"]):
                    print(f"central diff for states {i_state},{j_state}")
                    print(f"disp_plus  = {disp_plus[i_state][j_state]}")
                    print(f"disp_minus = {disp_minus[i_state][j_state]}")
                    print(f"distance   = {distance}")
                    print(f"(disp_plus - disp_minus)/distance   = {((disp_plus - disp_minus)/distance)[i_state][j_state]}")
        return (np.abs(disp_plus) - np.abs(disp_minus))/distance


    def create_displaced_interfaces(self):
        """
        Create a set of displaced QM calculations
        """
        displaced_interfaces = []

        # Loop over degrees of freedom
        for i_coord in range(self.QMin.molecule["nDoF"]):
            # Append a pair of displaced interfaces
            displaced_interfaces.append([self.spawn_displaced_interface(i_coord, '+'), self.spawn_displaced_interface(i_coord, '-')])
        
        return displaced_interfaces


    def spawn_displaced_interface(self, displacement_index, displacement_sign):
        """
        """
        displacement_letter = "p" if displacement_sign == '+' else "m"
        disp_name = f"disp_{displacement_index}_{displacement_letter}_{self.QMin.template['qm-program']}"
        displaced_interface = self.construct_qm_interface(disp_name)

        # Displace the coordinates
        displaced_interface.QMin.coords = self.displace_coordinates(displaced_interface.QMin.coords, displacement_index, displacement_sign)

        # Make savedir and scratchdir for the displaced interface
        savedir = os.path.join(self.ref_interface.QMin.save["savedir"], "displacements", disp_name)
        if not os.path.isdir(savedir):
            mkdir(savedir)
        scratchdir = os.path.join(self.ref_interface.QMin.resources["scratchdir"], "displacements", disp_name)
        if not os.path.isdir(scratchdir):
            mkdir(scratchdir)
        # Set the dirs
        displaced_interface.QMin.save["savedir"] = savedir
        displaced_interface.QMin.resources["scratchdir"] = scratchdir

        # We read the requests
        displaced_interface.read_requests(self.requests_file)

        # If we want to do num diff for overlaps we need to change the setup of the overlap files
        if "overlap" in self.QMin.template["properties"]:

            # Make sure that the displaced interface requests an overlap
            displaced_interface.QMin.requests["overlap"] = True
            
            # In SHARC_ABINITIO the files for wfoverlap is found using step=self.QMin.save["step"] and step-1.
            # We thus increment the step of the current interface
            displaced_interface.QMin.save["step"] = displaced_interface.QMin.save["step"] + 1

            # And we copy the mos and det files from the ref dir to the savedir of the current interface
            ref_savedir  = self.ref_interface.QMin.save["savedir"]
            disp_savedir = displaced_interface.QMin.save["savedir"]
            ref_files = os.listdir(ref_savedir)
            for ref_file in ref_files:
                if os.path.isfile(os.path.join(ref_savedir,  ref_file)):
                    shutil.copy(os.path.join(ref_savedir,  ref_file),
                                os.path.join(disp_savedir, ref_file))

        return displaced_interface


    def construct_qm_interface(self, interface_logname):
        """
        """
        # Create an interface
        qm_interface: SHARC_INTERFACE = factory(self.QMin.template['qm-program'])(logname=interface_logname, loglevel=self.log.level)

        # Setup the molecule and coordinate data for the reference QM calculation
        qm_interface.QMin.molecule = self.QMin.molecule 
        qm_interface._setup_mol = True
        qm_interface.QMin.coords = self.QMin.coords

        # Copy statemap into new interface
        qm_interface.QMin.maps["statemap"] = self.QMin.maps["statemap"]

        # Read in resources and template data
        qm_interface_template  = self.QMin.template["qm-program"].upper() + ".template"
        qm_interface_resources = self.QMin.template["qm-program"].upper() + ".resources"
        qm_interface.read_resources(qm_interface_resources)
        qm_interface.read_template(qm_interface_template)

        # Setup the interface
        qm_interface.setup_interface()
        
        return qm_interface


    def displace_coordinates(self, interface_coords, displacement_index, displacement_sign):
        """
        Take an interface coordinate and return a copy where the coordintates have been displaced 
        with a displacement corresponding the the displacement_index'th degree of freedom.
        """
        # Make a deepcopy of the input coordinates
        coords = deepcopy(interface_coords)

        # Make displacement of coord dependent on the coord-type
        # Cartesian
        if self.QMin.template["coord-type"] == "cartesian":
            # Determine atom and xyz index in coord from displacement_index
            i_atom = int(displacement_index / 3)
            i_cart = displacement_index % 3
            # Displace the coordinate
            if   displacement_sign == '+':
                coords["coords"][i_atom][i_cart] = coords["coords"][i_atom][i_cart] + self.QMin.template["numdiff-stepsize"]
            elif displacement_sign == '-':
                coords["coords"][i_atom][i_cart] = coords["coords"][i_atom][i_cart] - self.QMin.template["numdiff-stepsize"]
            else:
                raise RuntimeError("Argument 'displacement_sign' is invalid in call to 'displace_coordinates'. Argument must be '+' or '-'.")
        # Normal coordinates
        elif self.QMin.template["coord-type"] == "displacement":
            dispmat = self.QMin.disp_coords[displacement_index]
            if numdiff_debug:
                print("Input structure:")
                print(coords["coords"])
                print("Displacement coordinate:")
                print(dispmat)
                print("Displacement step size:")
                print(self.QMin.template["numdiff-stepsize"])
            # Displace the coordinate
            if   displacement_sign == '+':
                coords["coords"] = coords["coords"] + self.QMin.template["numdiff-stepsize"] * dispmat
            elif displacement_sign == '-':
                coords["coords"] = coords["coords"] - self.QMin.template["numdiff-stepsize"] * dispmat
            else:
                raise RuntimeError("Argument 'displacement_sign' is invalid in call to 'displace_coordinates'. Argument must be '+' or '-'.")
            if numdiff_debug:
                print("Displaced structure:")
                print(coords["coords"])
        # Unknown displacement
        else:
            err = f"coord-type: {self.QMin.template['coord-type']} is not recognized as a valid coordinate type"
            self.log.error(err)
            raise RuntimeError(err)
        
        return coords


if __name__ == "__main__":
    from logger import loglevel
    try:
        num_diff = SHARC_NUMDIFF(loglevel=loglevel)
        num_diff.main()
    except KeyboardInterrupt:
        print("\nCTRL+C makes me a sad SHARC ;-(")
