#!/usr/bin/env python3                                                                                                                                  

# ******************************************
#
#    SHARC Program Suite
#
#    Copyright (c) 2026 University of Vienna
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

# Interactive script for the extraction of EM-fields from FDTD simulations to a laser input file for SHARC
#
# usage: python extract_fields_fdtd.py


import numpy as np 
import os
import sys
import datetime
import argparse
import time
from logger import log
from utils import question                                 
import shutil
# from SHARC_INTERFACE import SHARC_INTERFACE                
# =========================================================
sharcversion='4.0'  # QA -> Take from SHARC

version = '1.0'                                                                                                                                
versionneeded = [1.0, float(version)]                                                                                           
versiondate = datetime.date(2023, 8, 24)                                                                                                       
global KEYSTROKES                                                                                                                              
old_question = question

# UNIT FACTORS
spat_unit_fac = 1E-6  # Conversion input unit to SI
temp_unit_fac = 1E-15  # Conversion input unit to SI
stepsize = 0.5  # Length of the nuclear dynamics time steps in fs: QA -> take from SHARC
nsubsteps = 25  # Number of substeps for the integration of the electronic EOM: QA -> take from SHARC


progress_width = 50
posresponse = ['y', 'yes', 'true', 't', 'ja', 'si', 'yea', 'yeah', 'aye', 'sure', 'definitely'] 
negresponse = ['n', 'no', 'false', 'f', 'nein', 'nope']                                         


def open_keystrokes():
    global KEYSTROKES
    KEYSTROKES = open('KEYSTROKES.tmp', 'w')


def close_keystrokes():
    KEYSTROKES.close()
    shutil.move('KEYSTROKES.tmp', 'KEYSTROKES.extract_laser_fields_rotation')


def custom_formatter(val: float):
    """
    Formats the laser fields files' values in defined scientific notation
    Args:
        x (int): 

    Returns:
       Formatted laser fields files' values 
    """
    assert isinstance(val, float), "val must be a float!"
    if val!=0.0:
        if np.abs(val)<1E-99:
            val=0.0
    val_form = '{:.8e}'.format(val)  # Format with 3 digits for the exponent
    mantissa, exponent = val_form.split('e')
    sign = '  ' if float(mantissa) >= 0 else ' '  # Check if positive
    return f'{sign}{mantissa}E{exponent[0]}{exponent[1:].zfill(2)}'


def displaywelcome():
    log.info('Script for extraction of laser fields from FDTD simulation output and creation of a laser field file started...\n')
    string = '\n'
    string += '  ' + '=' * 80 + '\n'
    input = [' ',
             'Align laser field polarization for SHARC dynamics',
             ' ',
             'Authors: Lorenz Grünewald',
             ' ',
             'Version: %s' % (version),
             'Date: %s' % (versiondate.strftime("%d.%m.%y")),
             ' ']
    for inp in input:
        string += '||{:^80}||\n'.format(inp)
    string += '  ' + '=' * 80 + '\n\n'
    string += '''
This script automatizes the alignment of laser fields from SHARC laser files for SHARC dynamics.
  '''
    log.info(string)


def get_general(head, INFOS):
    '''This routine questions from the user some general information:
    - FDTD simulation output file
    - temporal stepsize (for interpolation)
    - spatial (3D) stepsize (for interpolation)
    - spatial (3D) point at which the fields should be extracted'''

    #log.info(f'{"Laser file":-^60s}' + '\n')
    # open the initconds file
    try:
        np.loadtxt(INFOS["laser_file_path"], comments=("!", "#"))  
    except IOError:
        log.info('Could not open: {laser_file_path}')
    with open(INFOS["laser_file_path"], 'r') as version_test:
        if not isinstance(version_test.readline().strip()[0], float):
            with open(INFOS["laser_file_path"], 'r') as file:    
                for line_no, line in enumerate(file, start=0):
                    if len(line.split())>0:
                        if line.strip(" ").startswith("!") or line.strip(" ").startswith("#"):
                            head.append("  "+line)
    #rot_mat_file_path = os.path.expanduser(os.path.expandvars(rot_mat_file_path))
    if INFOS["rot_mat_file_path"] is not None:
        try:
            np.loadtxt(INFOS["rot_mat_file_path"], comments = ("!", "#"))  
        except IOError:
            log.info('Could not open: {rot_mat_file_path}')
    return head


def check_field_keywords(laser_file_path, INFOS):
    INFOS["e_field"] = False
    INFOS["b_field"] = False
    INFOS["e_field_gradients"] = False
    INFOS["b_field_gradients"] = False
    with open(laser_file_path, 'r') as file:
        if not isinstance(file.readline().strip()[0], float):
            for line_no, line in enumerate(file, start=1):
                if "e-field " in line:
                    INFOS["e_field"]=True if line.split()[2] in posresponse else False 
                if "b-field " in line:
                    INFOS["b_field"] = True if line.split()[2] in posresponse else False
                if "e-field_grad" in line:
                    INFOS["e_field_gradients"] = True if line.split()[2] in posresponse else False 
                if "b-field_grad " in line:
                    INFOS["b_field_gradients"] = True if line.split()[2] in posresponse else False

def gen_rot_matrix(INFOS):
    rot_z_mat = np.array([[np.cos(2*np.pi*INFOS["random_numbers"][0]), np.sin(2*np.pi*INFOS["random_numbers"][0]), 0],
                          [-np.sin(2*np.pi*INFOS["random_numbers"][0]), np.cos(2*np.pi*INFOS["random_numbers"][0]), 0],
                          [0, 0, 1]])
    householder_vec = np.array([np.cos(2*np.pi*INFOS["random_numbers"][1])*np.sqrt(INFOS["random_numbers"][2]),
                                np.sin(2*np.pi*INFOS["random_numbers"][1])*np.sqrt(INFOS["random_numbers"][2]),
                                np.sqrt(1-INFOS["random_numbers"][2])])
    householder_mat = np.eye(3, 3)-2*np.outer(householder_vec, householder_vec)
    return -householder_mat @ rot_z_mat

def rotate_matrix(laser_file_path, rot_mat_file_path, INFOS):
    laser_file = np.loadtxt(laser_file_path, comments = ("!", "#"))  
    rot_laser_fields = []
    rot_mat = INFOS["rot_matrix"]
    is_square = rot_mat.shape[0] == rot_mat.shape[1]
    is_orthogonal = np.allclose(np.dot(rot_mat, rot_mat.T), np.eye(3))
    determinant = np.linalg.det(rot_mat)
    if not all([is_square, is_orthogonal, (np.isclose(determinant, 1) or np.isclose(determinant, -1))]):
        log.info(f'No valid rotational matrix!')
        raise IOError
        
    # b_write_shift = e_write_shift+6*int(INFOS["e_field"])             
    # egrad_write_shift = b_write_shift+6*int(INFOS["b_field"])  
    # bgrad_write_shift = egrad_write_shift+18*int(INFOS["e_field_gradients"])
    # no_of_columns = bgrad_write_shift+18*int(INFOS["b_field_gradients"])    
    for line_no, line in enumerate(laser_file):
        done = int(line_no * progress_width // laser_file.shape[0])
        write_shift = int(0)                                             
        #log.info(done,laser_file.shape[0])
        #log.info(line_no)
        sys.stdout.write("\rTransformation progress: [" + "=" * done + " " * (progress_width - done) + "] %3i%% " % (done * 100 // progress_width))
        sys.stdout.flush()
        result=[line[0]]
        if str(INFOS["e_field"]).lower() in posresponse:
            efield_real = np.matmul(line[1:6:2], rot_mat) 
            efield_imag = np.matmul(line[2:7:2], rot_mat)
            result=result+[efield_real[0], efield_imag[0], 
                           efield_real[1], efield_imag[1],
                           efield_real[2], efield_imag[2]]
            write_shift += 6
        if str(INFOS["b_field"]).lower() in posresponse:
            bfield_real = np.matmul(line[1+write_shift:6+write_shift:2], rot_mat) 
            bfield_imag = np.matmul(line[2+write_shift:7+write_shift:2], rot_mat) 
            result=result+[bfield_real[0], bfield_imag[0],
                           bfield_real[1], bfield_imag[1],
                           bfield_real[2], bfield_imag[2]]
            write_shift += 6
        if str(INFOS["e_field_gradients"]).lower() in posresponse:
            efield_grad_real =  (rot_mat @ np.reshape(line[1+write_shift:18+write_shift:2], (3, 3)) @ rot_mat.T).flatten() 
            efield_grad_imag =  (rot_mat @ np.reshape(line[2+write_shift:19+write_shift:2], (3, 3)) @ rot_mat.T).flatten() 
            result=result+[item for pair in zip(efield_grad_real, efield_grad_imag) for item in pair]
            write_shift += 18
        result = result+[freq for freq in line[1+write_shift:]]
        rot_laser_fields.append(result)
        #print(rot_laser_fields, line_no)
    
    return rot_laser_fields

def rotate_matrix_old(laser_file_path, rot_mat_file_path, INFOS):
    laser_file = np.loadtxt(laser_file_path, comments = ("!", "#"))  
    if laser_file.shape[1]<8:
        log.info(f'Number of columns too small!')
    rot_laser_fields = []
    rot_mat = INFOS["rot_matrix"]
    is_square = rot_mat.shape[0] == rot_mat.shape[1]
    is_orthogonal = np.allclose(np.dot(rot_mat, rot_mat.T), np.eye(3))
    determinant = np.linalg.det(rot_mat)
    if not all([is_square, is_orthogonal, (np.isclose(determinant, 1) or np.isclose(determinant, -1))]):
        log.info(f'No valid rotational matrix!')
        raise IOError
    for line_no, line in enumerate(laser_file):
        done = int(line_no * progress_width // laser_file.shape[0])
        #log.info(done,laser_file.shape[0])
        #log.info(line_no)
        sys.stdout.write("\rTransformation progress: [" + "=" * done + " " * (progress_width - done) + "] %3i%% " % (done * 100 // progress_width))
        sys.stdout.flush()
        result=[line[0]]
        efield_real = np.matmul(line[1:6:2], rot_mat) 
        efield_imag = np.matmul(line[2:7:2], rot_mat)
        result=result+[efield_real[0], efield_imag[0], 
                       efield_real[1], efield_imag[1],
                       efield_real[2], efield_imag[2]]+list(line[7:])
        rot_laser_fields.append(result)
    log.info(rot_laser_fields)
    return rot_laser_fields


def main():
    '''Main routine'''

    usage = '''
    python extract_fields_fdtd.py 
    Interactive script for the extraction of EM-fields from FDTD simulations to a laser input file for SHARC
    As input it takes an FDTD output (.hdf5), the spatial position of the fields to be extracted and the time step to be interpolated
    '''
    open_keystrokes()
    head = []
    INFOS = {}
    # description = ''
    # parser = OptionParser(usage=usage, description=description)
    #print(INFOS)

    parser = argparse.ArgumentParser(description='Process input for laser file alignment.')
    parser.add_argument('--laser_file','-lf', type=str, help='Path to laser file')
    parser.add_argument('--rot_matrix','-rm', type=str, help='Path to rotation matrix')
    parser.add_argument('--output_name','-of', type=str, help='Path to output file')
    parser.add_argument('--random', '-r', type=int, help='Perform rotation based on random rotation matrix') 
    args = parser.parse_args()
    
    if args.laser_file is None:
        log.info(f'Laser file path is empty!')
        raise IOError
    elif not os.path.exists(args.laser_file):
        log.info(f'Laser file path "{args.laser_file}" does not exist!')
        raise IOError
    #laser_file_path = os.path.expanduser(os.path.expandvars(laser_file_path))
    if all(v is not None for v in [args.rot_matrix, args.random]) or all(v is None for v in [args.rot_matrix, args.random]):
        log.info(args.rot_matrix)
        log.info(args.random)
        log.info(f'Either calculation is performed with rotation matrix file or random rotation matrix')
        raise IOError
    if args.rot_matrix is not None:
        if not os.path.isfile(args.rot_matrix):
            log.info(f'Rotation matrix path "{args.rot_matrix}" does not exist!')
            raise IOError
    #if args.random is not None:
    #    if len(str(args.random)) != 3:
    #        log.info(f'Random seed "{args.random}" is not of format NNN!')
    #        raise IOError
    #    else:
    #        INFOS["random_numbers"] = args.random#[int(str(args.random)[i]) for i in range(len(str(args.random)))]

    INFOS["laser_file_path"] = args.laser_file
    INFOS["rot_mat_file_path"] = args.rot_matrix
    INFOS["output_file_path"] = args.output_name
    np.random.seed(args.random)
    INFOS["random_numbers"] = np.random.rand(3)#[int(str(args.random)[i]) for i in range(len(str(args.random)))]
   
    displaywelcome()
    #open_keystrokes()

    head = get_general(head, INFOS)
    check_field_keywords(INFOS["laser_file_path"], INFOS)
    if INFOS["rot_mat_file_path"] is not None:
        INFOS["rot_matrix"] = np.loadtxt(INFOS["rot_mat_file_path"])
    else:
        INFOS["rot_matrix"] = gen_rot_matrix(INFOS)
    for item in INFOS:
        log.info(f"{item:25} {INFOS[item]}")  
    if any([INFOS["e_field"], INFOS["b_field"], INFOS["e_field_gradients"], INFOS["b_field_gradients"]]):
        rot_laser_fields = rotate_matrix(INFOS["laser_file_path"], INFOS["rot_mat_file_path"], INFOS) 
    else:
        rot_laser_fields = rotate_matrix_old(INFOS["laser_file_path"], INFOS["rot_mat_file_path"], INFOS)
    sys.stdout.write("\rTransformation progress: [" + "=" * progress_width + " " * (0) + "] %3i%% \n" % (100))                                   
    sys.stdout.flush()
    formatted_laser_file = np.array([[" "]+[custom_formatter(val) for val in row] for row in rot_laser_fields], dtype=str)
    head=''.join(head)
    np.savetxt(INFOS["output_file_path"], formatted_laser_file, fmt="%s", delimiter="", header=head, comments='')
    close_keystrokes()
## ======================================================================================================================
#
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        log.info('\nCtrl+C makes me a sad SHARC ;-(\n')
        quit(0)
