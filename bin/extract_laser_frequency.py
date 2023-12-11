#!/usr/bin/env python3  

# ******************************************                                                               
#                                                                                                          
#    SHARC Program Suite                                                                                   
#                                                                                                          
#    Copyright (c) 2023 University of Vienna                                                               
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
# usage: python extract_laser_frequency.py                                                                     

import numpy as np                                                                                                                                  
import datetime 
import scipy.constants as const  # SHOULD THIS BE WRITTEN in the constants library?                                                                 
import os 
import shutil                                                                                                                                       
import matplotlib.pyplot as plt                                                                                                                     
                                                                                                                                                    
from logger import log                                                                                                                              
from scipy import fft, signal, ndimage                                                                                                              
from utils import question                                                                                                                          
# from SHARC_INTERFACE import SHARC_INTERFACE                                                                                                       
# =========================================================                                                                                         
sharcversion='4.0'  # QA -> Take from SHARC                                                                                                         
version = '1.0'                                                                                                                                     
versionneeded = [1.0, float(version)]                                                                                                               
versiondate = datetime.date(2023, 8, 24)                                                                                                            
global KEYSTROKES                                                                                                                                   
old_question = question                                                                                                                             
            
# UNIT FACTORS                                                                                                                                      
# spat_unit_fac = 1E-6  # Conversion input unit to SI                                                                                                 
temp_unit_fac = 1E-15  # Conversion input unit to SI                                                                                                
# stepsize = 0.5  # Length of the nuclear dynamics time steps in fs: QA -> take from SHARC                                                            
# nsubsteps = 25  # Number of substeps for the integration of the electronic EOM: QA -> take from SHARC                                               
efield_au_to_v_per_m = const.physical_constants["Hartree energy"][0]/const.e/const.physical_constants["Bohr radius"][0]                             
bfield_au_to_t = const.electron_mass*const.physical_constants["Hartree energy"][0]/(const.e*const.physical_constants["reduced Planck constant"][0]) 
# efield_grad_au_to_v_per_m2 = efield_au_to_v_per_m*const.physical_constants["Bohr radius"][0]                                                        
# bfield_grad_au_to_t_per_m =  bfield_au_to_t*const.physical_constants["Bohr radius"][0]                                                              


def question(question, typefunc, default=None, autocomplete=True, ranges=False):
    return old_question(question=question, typefunc=typefunc, KEYSTROKES=KEYSTROKES, default=default, autocomplete=autocomplete, ranges=ranges)


def try_read(word, index, typefunc, default):                                                                           
    try:                                                                                                                
        return typefunc(word[index])                                                                                    
    except IndexError:                                                                                                  
        return typefunc(default)                                                                                        
    except ValueError:                                                                                                  
        log.info('Could not initialize object!')                                                                        


def custom_formatter(val: float):
    """
    Formats the laser fields files' values in defined scientific notation
    Args:
        x (int): 

    Returns:
       Formatted laser fields files' values 
    """
    assert isinstance(val, float), "val must be a float!"
    if val==0:
        val=0.0
    elif np.log(val)<=-99:
        val=0.0
    elif np.isnan(val):
        return f'NaN'
    val_form = '{:.6e}'.format(val)  # Format with 3 digits for the exponent
    mantissa, exponent = val_form.split('e')
    sign = '+' if float(mantissa) >= 0 else ''  # Check if positive
    return f'{sign}{mantissa}E{exponent[0]}{exponent[1:].zfill(3)}'


def displaywelcome():
    log.info('Script for extraction of laser fields from FDTD simulation output and creation of a laser field file started...\n')
    string = '\n'
    string += '  ' + '=' * 80 + '\n'
    input = [' ',
             'Setup laser fields file for SHARC dynamics',
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
This script automatizes the extraction of laser fields from a MEEP FDTD simulation at one spatial point and creates a laser field file
for SHARC dynamics.
  '''
    log.info(string)


def open_keystrokes():
    global KEYSTROKES
    KEYSTROKES = open('KEYSTROKES.tmp', 'w')


def close_keystrokes():
    KEYSTROKES.close()
    shutil.move('KEYSTROKES.tmp', 'KEYSTROKES.extract_laser_fields')


def check_laser_file_version(laser_file_path):
    """
    Checks version of laser fields file to attain compatibility with old laser field files without extraction of field gradients
    Requested laser fields file header: "Laser fields file, version 1.0"
    Args:
        string (): 

    Returns:
       True/False (True, if Gradients are provided) 
    """
    with open(laser_file_path, "r") as laser_file:
        for line in laser_file:
            if 'version' not in line.split():
                return True
            else:
                pass
    return False


def get_general(INFOS):
    '''This routine questions from the user some general information:
    - laser file
    '''

    log.info(f'{"Laser file":-^60s}' + '\n')
    # open the initconds file
    try:
        laser_file_path = 'laser'
        if os.path.exists(laser_file_path):
            log.info('Laser file "laser" detected. Do you want to use this?')
            if not question('Use file "laser"?', bool, True):
                raise IOError
        else:
            raise IOError
    except IOError:
        log.info('\nIf you do not have a laser file, prepare one!\n')
        log.info('Please enter the path of your laser file.')
        while True:
            laser_file_path = question('Laser file path:', str)
            laser_file_path = os.path.expanduser(os.path.expandvars(laser_file_path))
            if os.path.isdir(laser_file_path):
                log.info(f'Is a directory: {laser_file_path}')
                continue
            if not os.path.isfile(laser_file_path):
                log.info(f'File does not exist: {laser_file_path}')
                continue
            try:
                np.loadtxt(laser_file_path, comments="!")
            except IOError:
                log.info('Could not open: {laser_file_path}')
            break

    INFOS["laser_file_path"] = laser_file_path
    return INFOS


def fft_calc(field, time_arr):                                                              
    freq_signal = fft.fft(field)[1:len(time_arr)//2]                                               
    freq = fft.fftfreq(len(time_arr), d=(time_arr[1]-time_arr[0]))[1:len(time_arr)//2]             
    central_freq = np.sum(np.abs(freq_signal)*np.abs(freq))/np.sum(np.abs(freq_signal))            
    # plt.plot(freq, np.abs(freq_signal), label="abs")                                               
    # plt.vlines(central_freq, 0, 2.5E-5, color="red", label="central_freq")                         
    # plt.legend()                                                                                   
    # plt.show()                                                                                     
    return central_freq


def wavelet_calc(field, time_arr):
    dt = time_arr[1]-time_arr[0]                                                                   
    freq_step = 1/dt                                                                               
    w=8*np.pi                                                                                      
    freq_arr = np.linspace(1, freq_step/2, 100)
    widths=w*freq_step/(2*freq_arr*np.pi)                                                          
    cwtm = np.abs(signal.cwt(field, signal.morlet2, widths, w=w))                                  
    # T_arr, F_arr = np.meshgrid(time_arr, freq_arr)                                                
    # total_weight = np.sum(cwtm, axis=1)                                                           
    # weighted_sum_f = np.sum(cwtm * F_arr)                                                         
    # weighted_sum_t = np.sum(cwtm * T_arr)                                                         
    # center_f = weighted_sum_f / total_weight                                                      
    # center_t = weighted_sum_t / total_weight                                                      
    # print(center_f, center_t*1E15)                                                                
    com_freq = [np.sum(cwtm[:, t_i]*freq_arr)/np.sum(cwtm[:, t_i]) for t_i in range(len(time_arr))]
    # print(com_freq)
    # plt.pcolormesh(time_arr*1E15, freq_arr, np.abs(cwtm), cmap="viridis", shading="gourad")        
    # plt.plot(time_arr*1E15, com_freq, "r-", linewidth=3, label="COM")                              
    # plt.plot(time_arr*1E15, freq_arr[np.argmax(cwtm, axis=0)], "b-", linewidth=3, label="MAX")     
    # plt.legend()                                                                                   
    # plt.show()                                                                                     
    return com_freq                                                                                       


def read_fields(INFOS):
    laser_file = np.loadtxt(INFOS["laser_file_path"], comments="!")
    return laser_file[:, 1:13]  # 2*3*2 fields


def read_time(INFOS):
    laser_file = np.loadtxt(INFOS["laser_file_path"], comments="!")
    return laser_file[:, 0]*temp_unit_fac


def main():
    '''Main routine'''
    usage = '''
    python extract_laser_frequency.py 
    Interactive script for the extraction of a central frequency from laser input file for SHARC.
    As input it takes a laser input file (laser)
    '''
    displaywelcome()
    open_keystrokes()
    INFOS = {}
    INFOS['cwd'] = os.getcwd()
    INFOS = get_general(INFOS)
    for item in INFOS:
        log.info(f"{item:<25} {INFOS[item]}") 
    time_arr = read_time(INFOS)
    header = f'''\
        ! Laser frequency file SHARC {sharcversion}
        ! version 1.0 
        ! laser file path = {INFOS["laser_file_path"]} 
        '''
    header = '\n'.join(line.lstrip() for line in header.split('\n'))
    laser_freq_file = np.nan*np.ones((len(time_arr), 2))  # tsteps, (f_exr, f_eyr, f_ezr or f_bxr, f_byr, f_bzr) #3*2 Exyz (real, imag), #3*2 Bxyz (real, imag), #3*3*2 Grad Exyz (real, imag), #3*3*2 Grad Bxyz (real, imag)
    laser_freq_file[:, 0] = time_arr*1E15  # SAVE timesteps in fs
    if check_laser_file_version(INFOS["laser_file_path"]):
        while True:
            fft_field = question("Do you want to obtain perform FFT (0) or WT (1)?", int, [0])
            em_fields= read_fields(INFOS)
            e_fields = em_fields[:, :6]
            b_fields = em_fields[:, 6:]/const.c
            e_fields_max = [np.max(e_fields[:, field_idx]) for field_idx in range(6)]
            b_fields_max = [np.max(b_fields[:, field_idx]) for field_idx in range(6)]
            em_fields_max = np.append(e_fields_max, b_fields_max)
            em_fields = np.hstack((e_fields, b_fields))
            if fft_field[0]==0:
                central_fft_freq = [fft_calc(em_fields[:, field_idx], time_arr) for field_idx in range(len(em_fields[0, :]))]
                combined_central_fft_freq = np.nansum(em_fields_max*central_fft_freq)/np.nansum(em_fields_max)
                # print(combined_central_fft_freq/(const.c/527.5E-9))
                laser_freq_file[:, 1] = np.ones_like(time_arr)*combined_central_fft_freq
                break
            elif fft_field[0]==1:
                central_fft_freq = [wavelet_calc(em_fields[:, field_idx], time_arr) for field_idx in range(len(em_fields[0, :]))]
                combined_central_fft_freq = np.nansum(em_fields.T**2*central_fft_freq, axis=0)/np.nansum(em_fields.T**2, axis=0)  
                laser_freq_file[:, 1] = combined_central_fft_freq
                # QA: Should a treshold be implemented such that the frequency is assigned NaN, if the corresponding fields are within noise/lower than a treshold value?
                # print(combined_central_fft_freq)#/(const.c/527.5E-9))
                # for i in range(12):
                #     plt.plot(range(161), central_fft_freq[i], label=i)
                # plt.plot(range(161), combined_central_fft_freq, linestyle="dashed", label="combined")
                # plt.legend()
                # plt.show()
                
                break
            else:
                log.info(f"Did not understand input: {fft_field}!")
                continue
    else:
        log.info("Laser file version not implemented currently!")
        raise IOError
    log.info("Writing frequencies to file: laser_freq")
    formatted_laser_freq_file = np.array([[custom_formatter(val) for val in row] for row in laser_freq_file], dtype=str)
    np.savetxt("laser_freq", formatted_laser_freq_file, fmt="%s", delimiter="  ", header=header, comments='')
        
    close_keystrokes()
    
# ======================================================================================================================

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        log.info('\nCtrl+C makes me a sad SHARC ;-(\n')
        quit(0)
