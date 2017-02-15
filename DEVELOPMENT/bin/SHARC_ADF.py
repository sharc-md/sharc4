#!/usr/bin/env python2

#    ====================================================================
#||                                                                       ||
#||                General Remarks                                        ||
#||                                                                       ||
#    ====================================================================
#
# This script uses several different specification for the electronic states under consideration.
# Generally, the input specs are like "3 Singlets and 3 Triplets"
# Based on this information, the states may be denoted by different systems.
#
# The most comprehensive denotation is: (mult, state, MS).
# In this case, states of higher multiplicity show up several times and the total number of states may be quite large.
# This total number of states is called nmstates in the script (equals to 12 for the above example)
#
# Since the MS components of a multiplet often share several expectation values, these need to be calculated only once.
# In this case, states can be safely denoted by (mult,state).
# The total number of these states is called nstates.
#
# In both systems, the states can be given indices. In this script and in SHARC, one first counts up the state quantum number,
# then the MS quantum number, and finally the multiplicity, like in this code snippet:
#
# i=0
# for mult in range(len(states)):
#     for MS in range(mult):
#         for state in range(states[mult]):
#             i+=1
#             print i, mult+1, state+1, MS-i/2
#
# more about this below in the docstrings of the iterator functions

# ======================================================================= #

# IMPLEMENTATION OF ADDITIONAL TASKS KEYWORDS, JOBS, ETC:
#

# ======================================================================= #
# Modules:
# Operating system, isfile and related routines, move files, create directories
import os
import shutil
# External Calls to MOLCAS
import subprocess as sp
# Command line arguments
import sys
# Regular expressions
import re
# debug print for dicts and arrays
import pprint
# sqrt and other math
import math
import cmath
# runtime measurement
import datetime
# copy of arrays of arrays
from copy import deepcopy
# parallel calculations
from multiprocessing import Pool
import time
from socket import gethostname

try:
  import numpy
except ImportError:
  print 'The kf module required to read ADF binary files needs numpy. Please install numpy and then try again'
  sys.exit(11)

# =========================================================0
# compatibility stuff

if sys.version_info[0]!=2:
    print 'This is a script for Python 2!'
    sys.exit(0)

if sys.version_info[1]<5:
    def any(iterable):
        for element in iterable:
            if element:
                return True
        return False

    def all(iterable):
        for element in iterable:
            if not element:
                return False
        return True

# ======================================================================= #

version='1.1'
versiondate=datetime.date(2016,9,15)



changelogstring='''
30.09.2016:
- PARTIAL REWORK
- added capabilities to run unrestricted SHARC ADF dynamics
- Enabled single multiplicity restricted runs

03.10.2016:
-Fixed instances of duplicate key words
-Removed key words that are not relevant to ADF interface
-Fixed number of core use for gradients when GS gradient is not explicitly calculated
-Changed link routine to same as new SHARC_MOLPRO interface
-Simplified frozen core vairable reading

05.10.2016:
-Fixed minor issue with number of excitations when running unrestricted calculations of doublet or higher than triplet multiplicity

14.10.2016:
-Fixed a minor issue with the CreateQMout subroutine

17.10.2016:
-Fixed the unrestricted multiplicity checking routine
-Added internal checks for charge and multiplicity
-Fixed for singlet runs only that it only calculate singlet excitations in the TD-DFT
-For doublet, quartet etc, fixes charge relative to the atomic charge
-Added atomic charge library

18.10.2016
-Fixed some issues with regards to only singlet runs

28.10.2016
-Added a keyword allowing the use to choose the number of padding states for the TD-DFT
-Modified the gradient routine to initiate a gradient calculation in the initial TD-DFT
-Fixed an issue with regards to using multiple basis sets during the AO overlap calculation

14.11.2016
-Fixed routine for creating the cicoef files so that in an unrestircted case both alpha and beta orbitals are frozen

17.01.2017
-Fixed an indexing error in the get_cicoef module that meant that the CI vector was sometimes truncated too early
-Fixed an indexing error in the overlaps routine for writing QMout

09.02.2017
-Rewrote the routine for get_cicoef so that it is correctly done for the unrestricted case.
'''

# ======================================================================= #
# holds the system time when the script was started
starttime=datetime.datetime.now()

# global variables for printing (PRINT gives formatted output, DEBUG gives raw output)
DEBUG=False
PRINT=True

# hash table for conversion of multiplicity to the keywords used in MOLCAS
IToMult={
                 1: 'Singlet', 
                 2: 'Doublet', 
                 3: 'Triplet', 
                 4: 'Quartet', 
                 5: 'Quintet', 
                 6: 'Sextet', 
                 7: 'Septet', 
                 8: 'Octet', 
                 'Singlet': 1, 
                 'Doublet': 2, 
                 'Triplet': 3, 
                 'Quartet': 4, 
                 'Quintet': 5, 
                 'Sextet': 6, 
                 'Septet': 7, 
                 'Octet': 8
                 }

# hash table for conversion of polarisations to the keywords used in MOLCAS
IToPol={
                0: 'X', 
                1: 'Y', 
                2: 'Z', 
                'X': 0, 
                'Y': 1, 
                'Z': 2
                }

#Number of frozen core orbitals
FROZENS = {'H':  0, 'He': 0,
'Li': 1, 'Be': 1, 'B':  1, 'C':  1,  'N': 1,  'O': 1, 'F':  1, 'Ne':1,
'Na':1, 'Mg':1, 'Al':5, 'Si':5,  'P':5,  'S':5, 'Cl':5, 'Ar':5,
'K': 5, 'Ca':5,
'Sc':5, 'Ti':5, 'V': 5, 'Cr':5, 'Mn':5, 'Fe':5, 'Co':5, 'Ni':5, 'Cu':5, 'Zn':5,
'Ga':9, 'Ge':9, 'As':9, 'Se':9, 'Br':9, 'Kr':9,
'Rb':9, 'Sr':9,
'Y':14,  'Zr':14, 'Nb':14, 'Mo':14, 'Tc':14, 'Ru':14, 'Rh':14, 'Pd':14, 'Ag':14, 'Cd':14,
'In':18, 'Sn':18, 'Sb':18, 'Te':18,  'I':18, 'Xe':18,
'Cs':18, 'Ba':18,
'La':23, 'Lu':23, 'Hf':23, 'Ta':23,  'W':23, 'Re':23, 'Os':23, 'Ir':23, 'Pt':23, 'Au':23, 'Hg':23,
'Tl':23, 'Pb':23, 'Bi':23, 'Po':23, 'At':23, 'Rn':23
}

ELEMENTS = {'H':'h', 'He':'he',
'Li':'li', 'Be':'be', 'B':'b', 'C':'c',  'N':'n',  'O':'o', 'F':'f', 'Ne':'ne',
'Na':'na', 'Mg':'mg', 'Al':'al', 'Si':'si',  'P':'p',  'S':'s', 'Cl':'cl', 'Ar':'ar',
'K':'k', 'Ca':'ca',
'Sc':'sc', 'Ti':'ti', 'V':'v', 'Cr':'cr', 'Mn':'mn', 'Fe':'fe', 'Co':'co', 'Ni':'ni', 'Cu':'cu', 'Zn':'zn',
'Ga':'ga', 'Ge':'ge', 'As':'as', 'Se':'se', 'Br':'br', 'Kr':'kr',
'Rb':'rb', 'Sr':'sr',
'Y':'y',  'Zr':'zr', 'Nb':'nb', 'Mo':'mo', 'Tc':'tc', 'Ru':'ru', 'Rh':'rh', 'Pd':'pd', 'Ag':'ag', 'Cd':'cd',
'In':'in', 'Sn':'sn', 'Sb':'sb', 'Te':'te',  'I':'i', 'Xe':'xe',
'Cs':'cs', 'Ba':'ba',
'La':'la', 'Lu':'lu', 'Hf':'hf', 'Ta':'ta',  'W':'w', 'Re':'re', 'Os':'os', 'Ir':'ir', 'Pt':'pt', 'Au':'au', 'Hg':'hg',
'Tl':'tl', 'Pb':'pb', 'Bi':'bi', 'Po':'po', 'At':'at', 'Rn':'rn'
}


ATOMCHARGE = {'H':1, 'He':2,
'Li':3, 'Be':4, 'B':5, 'C':6,  'N':7,  'O':8, 'F':9, 'Ne':10,
'Na':11, 'Mg':12, 'Al':13, 'Si':14,  'P':15,  'S':16, 'Cl':17, 'Ar':18,
'K':19, 'Ca':20,
'Sc':21, 'Ti':22, 'V':23, 'Cr':24, 'Mn':25, 'Fe':26, 'Co':27, 'Ni':28, 'Cu':29, 'Zn':30,
'Ga':31, 'Ge':32, 'As':33, 'Se':34, 'Br':35, 'Kr':36,
'Rb':37, 'Sr':38,
'Y':39,  'Zr':40, 'Nb':41, 'Mo':42, 'Tc':43, 'Ru':44, 'Rh':45, 'Pd':46, 'Ag':47, 'Cd':48,
'In':49, 'Sn':50, 'Sb':51, 'Te':52,  'I':53, 'Xe':54,
'Cs':55, 'Ba':56,
'La':57, 'Lu':71, 'Hf':72, 'Ta':73,  'W':74, 'Re':75, 'Os':76, 'Ir':77, 'Pt':78, 'Au':79, 'Hg':80,
'Tl':81, 'Pb':82, 'Bi':83, 'Po':84, 'At':85, 'Rn':86
}

# conversion factors
au2a=0.529177211
rcm_to_Eh=4.556335e-6
a2au=1.889726124

# =============================================================================================== #
# =============================================================================================== #
# =========================================== general routines ================================== #
# =============================================================================================== #
# =============================================================================================== #

# ======================================================================= #
def readfile(filename):
  try:
    f=open(filename)
    out=f.readlines()
    f.close()
  except IOError:
    print 'File %s does not exist!' % (filename)
    sys.exit(13)
  return out

# ======================================================================= #
def writefile(filename,content):
  # content can be either a string or a list of strings
  try:
    f=open(filename,'w')
    if isinstance(content,list):
      for line in content:
        f.write(line)
    elif isinstance(content,str):
      f.write(content)
    else:
      print 'Content %s cannot be written to file!' % (content)
    f.close()
  except IOError:
    print 'Could not write to file %s!' % (filename)
    sys.exit(14)

# ======================================================================= #
def isbinary(path):
  return (re.search(r':.* text',sp.Popen(["file", '-L', path], stdout=sp.PIPE).stdout.read())is None)

# ======================================================================= #
def eformat(f, prec, exp_digits):
    '''Formats a float f into scientific notation with prec number of decimals and exp_digits number of exponent digits.

    String looks like:
    [ -][0-9]\.[0-9]*E[+-][0-9]*

    Arguments:
    1 float: Number to format
    2 integer: Number of decimals
    3 integer: Number of exponent digits

    Returns:
    1 string: formatted number'''

    s = "% .*e"%(prec, f)
    mantissa, exp = s.split('e')
    return "%sE%+0*d"%(mantissa, exp_digits+1, int(exp))

# ======================================================================= #
def measuretime():
    '''Calculates the time difference between global variable starttime and the time of the call of measuretime.

    Prints the Runtime, if PRINT or DEBUG are enabled.

    Arguments:
    none

    Returns:
    1 float: runtime in seconds'''

    endtime=datetime.datetime.now()
    runtime=endtime-starttime
    if PRINT or DEBUG:
        hours=runtime.seconds/3600
        minutes=runtime.seconds/60-hours*60
        seconds=runtime.seconds%60
        print '==> Runtime:\n%i Days\t%i Hours\t%i Minutes\t%i Seconds\n\n' % (runtime.days,hours,minutes,seconds)
    total_seconds=runtime.days*24*3600+runtime.seconds+runtime.microseconds/1.e6
    return total_seconds

# ======================================================================= #
def removekey(d,key):
    '''Removes an entry from a dictionary and returns the dictionary.

    Arguments:
    1 dictionary
    2 anything which can be a dictionary keyword

    Returns:
    1 dictionary'''

    if key in d:
        r = dict(d)
        del r[key]
        return r
    return d

# ======================================================================= #         OK
def containsstring(string,line):
    '''Takes a string (regular expression) and another string. Returns True if the first string is contained in the second string.

    Arguments:
    1 string: Look for this string
    2 string: within this string

    Returns:
    1 boolean'''

    a=re.search(string,line)
    if a:
        return True
    else:
        return False



# =============================================================================================== #
# =============================================================================================== #
# ============================= iterator routines  ============================================== #
# =============================================================================================== #
# =============================================================================================== #

# ======================================================================= #
def itmult(states):

    for i in range(len(states)):
        if states[i]<1:
            continue
        yield i+1
    return

# ======================================================================= #
def itnmstates(states):

    for i in range(len(states)):
        if states[i]<1:
            continue
        for k in range(i+1):
            for j in range(states[i]):
                yield i+1,j+1,k-i/2.
    return

# =============================================================================================== #
# =============================================================================================== #
# =========================================== print routines ==================================== #
# =============================================================================================== #
# =============================================================================================== #

# ======================================================================= #
def printheader():
    '''Prints the formatted header of the log file. Prints version number and version date

    Takes nothing, returns nothing.'''

    print starttime,gethostname(),os.getcwd()
    if not PRINT:
        return
    string='\n'
    string+='  '+'='*80+'\n'
    string+='||'+' '*80+'||\n'
    string+='||'+' '*28+'SHARC - ADF - Interface'+' '*28+'||\n'
    string+='||'+' '*80+'||\n'
    string+='||'+' '*28+'Authors: Andrew Atkins'+' '*29+'||\n'
    string+='||'+' '*80+'||\n'
    string+='||'+' '*(36-(len(version)+1)/2)+'Version: %s' % (version)+' '*(35-(len(version))/2)+'||\n'
    lens=len(versiondate.strftime("%d.%m.%y"))
    string+='||'+' '*(37-lens/2)+'Date: %s' % (versiondate.strftime("%d.%m.%y"))+' '*(37-(lens+1)/2)+'||\n'
    string+='||'+' '*80+'||\n'
    string+='  '+'='*80+'\n\n'
    print string
    if DEBUG:
        print changelogstring

# ======================================================================= #
def printQMin(QMin):

  if DEBUG:
    pprint.pprint(QMin)
  if not PRINT:
    return
  print '==> QMin Job description for:\n%s' % (QMin['comment'])

  string='Tasks:  '
  if 'h' in QMin:
    string+='\tH'
  if 'soc' in QMin:
    string+='\tSOC'
  if 'dm' in QMin:
    string+='\tDM'
  if 'grad' in QMin:
    string+='\tGrad'
  if 'nacdr' in QMin:
    string+='\tNac(ddr)'
  if 'nacdt' in QMin:
    string+='\tNac(ddt)'
  if 'overlap' in QMin:
    string+='\tOverlaps'
  if 'angular' in QMin:
    string+='\tAngular'
  if 'ion' in QMin:
    string+='\tDyson norms'
  if 'dmdr' in QMin:
    string+='\tDM-Grad'
  if 'socdr' in QMin:
    string+='\tSOC-Grad'
  print string

  string='States: '
  for i in itmult(QMin['states']):
    string+='\t%i %s' % (QMin['states'][i-1],IToMult[i])
  print string

  string='Method: \t'
  string+='DFT '
  for i in range(0,2): 
      string+=QMin['template']['xc'][1][i].upper()
      if i==0:
         string+=' '
  string+='(%i)/%s' % (int(QMin['template']['excitation'][2][1]),QMin['template']['basis'][1][1])
  parts=[]
  if 'sopert' in QMin['template']:
    parts.append('Spin orbit coupling (perturbative)')
  if len(parts)>0:
      string+='\t('
      string+=','.join(parts)
      string+=')'
  print string

  string='Found Geo'
  if 'veloc' in QMin:
    string+=' and Veloc! '
  else:
    string+='! '
  string+='NAtom is %i.\n' % (QMin['natom'])
  print string

  string='\nGeometry in Bohrs:\n'
  for i in range(QMin['natom']):
    string+='%s ' % (QMin['geo'][i][0])
    for j in range(3):
      string+='% 7.4f ' % (QMin['geo'][i][j+1])
    string+='\n'
  print string

  if 'veloc' in QMin:
    string=''
    for i in range(QMin['natom']):
      string+='%s ' % (QMin['geo'][i][0])
      for j in range(3):
        string+='% 7.4f ' % (QMin['veloc'][i][j])
      string+='\n'
    print string

  if 'grad' in QMin:
    string='Gradients:   '
    for i in range(1,QMin['nmstates']+1):
      if i in QMin['grad']:
        string+='X '
      else:
        string+='. '
    string+='\n'
    print string

  if 'overlap' in QMin:
    string='Overlaps:\n'
    for i in range(1,QMin['nmstates']+1):
      for j in range(1,QMin['nmstates']+1):
        if [i,j] in QMin['overlap'] or [j,i] in QMin['overlap']:
          string+='X '
        else:
          string+='. '
      string+='\n'
    print string

  for i in QMin:
    if not any( [i==j for j in ['h','dm','soc','dmdr','socdr','geo','veloc','states','comment','LD_LIBRARY_PATH', 'grad','nacdr','ion','overlap','template'] ] ):
      if not any( [i==j for j in ['ionlist','ionmap'] ] ) or DEBUG:
        string=i+': '
        string+=str(QMin[i])
        print string
  print '\n'
  sys.stdout.flush()

# ======================================================================= #
def printtasks(tasks):
    '''If PRINT, prints a formatted table of the tasks in the tasks list.

    Arguments:
    1 list of lists: tasks list (see gettasks for specs)'''

    if DEBUG:
        pprint.pprint(tasks)
    if not PRINT:
        return
    print '==> Task Queue:\n'
    for i in range(len(tasks)):
      task=tasks[i]
      if task[0]=='movetoold':
        print 'Move SAVE to old'
      if task[0]=='backupdata':
        print 'Backup data\t%s' % (task[1])
      elif task[0]=='cleanup':
        print 'Clean directory\t%s' % (task[1])
      elif task[0]=='mkdir':
        print 'Make directory\t%s' % (task[1])
      elif task[0]=='link':
        print 'Link\t\t%s\n\t--> \t%s' % (task[2],task[1])
      elif task[0]=='writeinput':
        print 'Writes the ADF .run files for the TD-DFT, Gradient and AO overlap calculations'
      elif task[0]=='td-dft':
        print 'Runs the first TD-DFT calculation needed for each step'
      elif task[0]=='gradientcalculation':
        print 'Runs all the needed gradient calculations'
      elif task[0]=='get_ADF_out':
        print 'Extracts the needed information from the .out and .t21 files'
      elif task[0]=='check_supergeom':
        print 'Checks that the supergeometry needed for the AO overlap can be ran'
      elif task[0]=='ADF_AOcalc':
        print 'Runs the supergeometry AO overlap calculation'
      elif task[0]=='get_Overlap_mat':
        print 'Retrieves the AO overlap matrix for the supergeometry'
      elif task[0]=='getmocoef':
        print 'Retrieves the needed MO coefficients from the current and previous step calculation'
      elif task[0]=='get_CIcoef':
        print 'Retrieves and constructs the CI coefficients from the eigenvectors'
      elif task[0]=='run_WFOverlap':
        print 'Runs the wave function overlap calculation for the NACV\'s'
      elif task[0]=='createQMout':
        print 'Creates the QMout file needed by SHARC'
      else:
        print task
    print '\n'
    sys.stdout.flush()

# ======================================================================= #
def printcomplexmatrix(matrix,states):
    '''Prints a formatted matrix. Zero elements are not printed, blocks of different mult and MS are delimited by dashes. Also prints a matrix with the imaginary parts, of any one element has non-zero imaginary part.

    Arguments:
    1 list of list of complex: the matrix
    2 list of integers: states specs'''

    nmstates=0
    for i in range(len(states)):
        nmstates+=states[i]*(i+1)
    string='Real Part:\n'
    string+='-'*(11*nmstates+nmstates/3)
    string+='\n'
    istate=0
    for imult,i,ms in itnmstates(states):
        jstate=0
        string+='|'
        for jmult,j,ms2 in itnmstates(states):
            if matrix[istate][jstate].real==0.:
                string+=' '*11
            else:
                string+='% .3e ' % (matrix[istate][jstate].real)
            if j==states[jmult-1]:
                string+='|'
            jstate+=1
        string+='\n'
        if i==states[imult-1]:
            string+='-'*(11*nmstates+nmstates/3)
            string+='\n'
        istate+=1
    print string
    imag=False
    string='Imaginary Part:\n'
    string+='-'*(11*nmstates+nmstates/3)
    string+='\n'
    istate=0
    for imult,i,ms in itnmstates(states):
        jstate=0
        string+='|'
        for jmult,j,ms2 in itnmstates(states):
            if matrix[istate][jstate].imag==0.:
                string+=' '*11
            else:
                imag=True
                string+='% .3e ' % (matrix[istate][jstate].imag)
            if j==states[jmult-1]:
                string+='|'
            jstate+=1
        string+='\n'
        if i==states[imult-1]:
            string+='-'*(11*nmstates+nmstates/3)
            string+='\n'
        istate+=1
    string+='\n'
    if imag:
        print string

# ======================================================================= #
def printgrad(grad,natom,geo):
    '''Prints a gradient or nac vector. Also prints the atom elements. If the gradient is identical zero, just prints one line.

    Arguments:
    1 list of list of float: gradient
    2 integer: natom
    3 list of list: geometry specs'''

    string=''
    iszero=True
    for atom in range(natom):
        string+='%i\t%s\t' % (atom+1,geo[atom][0])
        for xyz in range(3):
            if grad[atom][xyz]!=0:
                iszero=False
            g=grad[atom][xyz]
            if isinstance(g,float):
                string+='% .5f\t' % (g)
            elif isinstance(g,complex):
                string+='% .5f\t% .5f\t\t' % (g.real,g.imag)
        string+='\n'
    if iszero:
        print '\t\t...is identical zero...\n'
    else:
        print string

# ======================================================================= #
def printQMout(QMin,QMout):
    '''If PRINT, prints a summary of all requested QM output values. Matrices are formatted using printcomplexmatrix, vectors using printgrad. 

    Arguments:
    1 dictionary: QMin
    2 dictionary: QMout'''

    #if DEBUG:
        #pprint.pprint(QMout)
    if not PRINT:
        return
    states=QMin['states']
    nstates=QMin['nstates']
    nmstates=QMin['nmstates']
    natom=QMin['natom']
    print '===> Results:\n'
    # Hamiltonian matrix, real or complex
    if 'h' in QMin or 'soc' in QMin:
        eshift=math.ceil(QMout['h'][0][0].real)
        print '=> Hamiltonian Matrix:\nDiagonal Shift: %9.2f' % (eshift)
        matrix=deepcopy(QMout['h'])
        for i in range(nmstates):
            matrix[i][i]-=eshift
        printcomplexmatrix(matrix,states)
    # Dipole moment matrices
    if 'dm' in QMin:
        print '=> Dipole Moment Matrices:\n'
        for xyz in range(3):
            print 'Polarisation %s:' % (IToPol[xyz])
            matrix=QMout['dm'][xyz]
            printcomplexmatrix(matrix,states)
    # Gradients
    if 'grad' in QMin:
        print '=> Gradient Vectors:\n'
        istate=0
        for imult,i,ms in itnmstates(states):
            print '%s\t%i\tMs= % .1f:' % (IToMult[imult],i,ms)
            printgrad(QMout['grad'][istate],natom,QMin['geo'])
            istate+=1
    # Overlaps
    if 'overlap' in QMin:
        print '=> Overlap matrix:\n'
        matrix=QMout['overlap']
        printcomplexmatrix(matrix,states)
        if 'phases' in QMout:
            print '=> Wavefunction Phases:\n'
            for i in range(nmstates):
                print '% 3.1f % 3.1f' % (QMout['phases'][i].real,QMout['phases'][i].imag)
            print '\n'
    # Spin-orbit coupling derivatives
    if 'socdr' in QMin:
        print '=> Spin-Orbit Gradient Vectors:\n'
        istate=0
        for imult,i,ims in itnmstates(states):
            jstate=0
            for jmult,j,jms in itnmstates(states):
                print '%s\t%i\tMs= % .1f -- %s\t%i\tMs= % .1f:' % (IToMult[imult],i,ims,IToMult[jmult],j,jms)
                printgrad(QMout['socdr'][istate][jstate],natom,QMin['geo'])
                jstate+=1
            istate+=1
    # Dipole moment derivatives
    if 'dmdr' in QMin:
        print '=> Dipole moment derivative vectors:\n'
        istate=0
        for imult,i,msi in itnmstates(states):
            jstate=0
            for jmult,j,msj in itnmstates(states):
                if imult==jmult and msi==msj:
                    for ipol in range(3):
                        print '%s\tStates %i - %i\tMs= % .1f\tPolarization %s:' % (IToMult[imult],i,j,msi,IToPol[ipol])
                        printgrad(QMout['dmdr'][ipol][istate][jstate],natom,QMin['geo'])
                jstate+=1
            istate+=1
    sys.stdout.flush()


# =============================================================================================== #
# =============================================================================================== #
# ======================================= Matrix initialization ================================= #
# =============================================================================================== #
# =============================================================================================== #

# ======================================================================= #         OK
def makecmatrix(a,b):
    '''Initialises a complex axb matrix.

    Arguments:
    1 integer: first dimension
    2 integer: second dimension

    Returns;
    1 list of list of complex'''

    mat=[ [ complex(0.,0.) for i in range(a) ] for j in range(b) ]
    return mat

# ======================================================================= #         OK
def makermatrix(a,b):
    '''Initialises a real axb matrix.

    Arguments:
    1 integer: first dimension
    2 integer: second dimension

    Returns;
    1 list of list of real'''

    mat=[ [ 0. for i in range(a) ] for j in range(b) ]
    return mat


# =============================================================================================== #
# =============================================================================================== #
# =========================================== QMout writing ===================================== #
# =============================================================================================== #
# =============================================================================================== #


# ======================================================================= #
def writeQMout(QMin,QMout,QMinfilename):
    '''Writes the requested quantities to the file which SHARC reads in. The filename is QMinfilename with everything after the first dot replaced by "out". 

    Arguments:
    1 dictionary: QMin
    2 dictionary: QMout
    3 string: QMinfilename'''

    k=QMinfilename.find('.')
    if k==-1:
        outfilename=QMinfilename+'.out'
    else:
        outfilename=QMinfilename[:k]+'.out'
    if PRINT:
        print '===> Writing output to file %s in SHARC Format\n' % (outfilename)
    string=''
    if 'h' in QMin or 'soc' in QMin:
        string+=writeQMoutsoc(QMin,QMout)
    if 'dm' in QMin:
        string+=writeQMoutdm(QMin,QMout)
    if 'grad' in QMin:
        string+=writeQMoutgrad(QMin,QMout)
    if 'overlap' in QMin:
        string+=writeQMoutnacsmat(QMin,QMout)
    if 'socdr' in QMin:
        string+=writeQMoutsocdr(QMin,QMout)
    if 'dmdr' in QMin:
        string+=writeQMoutdmdr(QMin,QMout)
    string+=writeQMouttime(QMin,QMout)
    outfile=os.path.join(QMin['pwd'],outfilename)
    writefile(outfile,string)
    return

# ======================================================================= #
def writeQMoutsoc(QMin,QMout):
    '''Generates a string with the Spin-Orbit Hamiltonian in SHARC format.

    The string starts with a ! followed by a flag specifying the type of data. In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements. Blocks are separated by a blank line.

    Arguments:
    1 dictionary: QMin
    2 dictionary: QMout

    Returns:
    1 string: multiline string with the SOC matrix'''

    states=QMin['states']
    nstates=QMin['nstates']
    nmstates=QMin['nmstates']
    natom=QMin['natom']
    string=''
    string+='! %i Hamiltonian Matrix (%ix%i, complex)\n' % (1,nmstates,nmstates)
    string+='%i %i\n' % (nmstates,nmstates)
    for i in range(nmstates):
        for j in range(nmstates):
            string+='%s %s ' % (eformat(QMout['h'][i][j].real,9,3),eformat(QMout['h'][i][j].imag,9,3))
        string+='\n'
    string+='\n'
    return string

# ======================================================================= #
def writeQMoutdm(QMin,QMout):
    '''Generates a string with the Dipole moment matrices in SHARC format.

    The string starts with a ! followed by a flag specifying the type of data. In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements. Blocks are separated by a blank line. The string contains three such matrices.

    Arguments:
    1 dictionary: QMin
    2 dictionary: QMout

    Returns:
    1 string: multiline string with the DM matrices'''

    states=QMin['states']
    nstates=QMin['nstates']
    nmstates=QMin['nmstates']
    natom=QMin['natom']
    string=''
    string+='! %i Dipole Moment Matrices (3x%ix%i, complex)\n' % (2,nmstates,nmstates)
    for xyz in range(3):
        string+='%i %i\n' % (nmstates,nmstates)
        for i in range(nmstates):
            for j in range(nmstates):
                string+='%s %s ' % (eformat(QMout['dm'][xyz][i][j].real,9,3),eformat(QMout['dm'][xyz][i][j].imag,9,3))
            string+='\n'
        #string+='\n'
    return string

# ======================================================================= #
def writeQMoutgrad(QMin,QMout):
    '''Generates a string with the Gradient vectors in SHARC format.

    The string starts with a ! followed by a flag specifying the type of data. On the next line, natom and 3 are written, followed by the gradient, with one line per atom and a blank line at the end. Each MS component shows up (nmstates gradients are written).

    Arguments:
    1 dictionary: QMin
    2 dictionary: QMout

    Returns:
    1 string: multiline string with the Gradient vectors'''

    states=QMin['states']
    nstates=QMin['nstates']
    nmstates=QMin['nmstates']
    natom=QMin['natom']
    string=''
    string+='! %i Gradient Vectors (%ix%ix3, real)\n' % (3,nmstates,natom)
    i=0
    for imult,istate,ims in itnmstates(states):
        string+='%i %i ! %i %i %i\n' % (natom,3,imult,istate,ims)
        for atom in range(natom):
            for xyz in range(3):
                string+='%s ' % (eformat(QMout['grad'][i][atom][xyz],9,3))
            string+='\n'
        #string+='\n'
        i+=1
    return string

# ======================================================================= #
def writeQMoutnacsmat(QMin,QMout):
    '''Generates a string with the adiabatic-diabatic transformation matrix in SHARC format.

    The string starts with a ! followed by a flag specifying the type of data. In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements. Blocks are separated by a blank line.

    Arguments:
    1 dictionary: QMin
    2 dictionary: QMout

    Returns:
    1 string: multiline string with the transformation matrix'''

    states=QMin['states']
    nstates=QMin['nstates']
    nmstates=QMin['nmstates']
    natom=QMin['natom']
    string=''
    string+='! %i Overlap matrix (%ix%i, complex)\n' % (6,nmstates,nmstates)
    string+='%i %i\n' % (nmstates,nmstates)
    for j in range(nmstates):
        for i in range(nmstates):
            string+='%s %s ' % (eformat(QMout['overlap'][j][i].real,9,3),eformat(QMout['overlap'][j][i].imag,9,3))
        string+='\n'
    string+='\n'
    return string

# ======================================================================= #
def writeQMoutdmdr(QMin,QMout):

  states=QMin['states']
  nmstates=QMin['nmstates']
  natom=QMin['natom']
  string=''
  string+='! %i Dipole moment derivatives (%ix%ix3x%ix3, real)\n' % (12,nmstates,nmstates,natom)
  i=0
  for imult,istate,ims in itnmstates(states):
    j=0
    for jmult,jstate,jms in itnmstates(states):
      for ipol in range(3):
        string+='%i %i ! m1 %i s1 %i ms1 %i   m2 %i s2 %i ms2 %i   pol %i\n' % (natom,3,imult,istate,ims,jmult,jstate,jms,ipol)
        for atom in range(natom):
          for xyz in range(3):
            string+='%s ' % (eformat(QMout['dmdr'][ipol][i][j][atom][xyz],12,3))
          string+='\n'
        string+=''
      j+=1
    i+=1
  return string

# ======================================================================= #
def writeQMoutsocdr(QMin,QMout):

  states=QMin['states']
  nmstates=QMin['nmstates']
  natom=QMin['natom']
  string=''
  string+='! %i Spin-Orbit coupling derivatives (%ix%ix3x%ix3, complex)\n' % (13,nmstates,nmstates,natom)
  i=0
  for imult,istate,ims in itnmstates(states):
    j=0
    for jmult,jstate,jms in itnmstates(states):
        string+='%i %i ! m1 %i s1 %i ms1 %i   m2 %i s2 %i ms2 %i\n' % (natom,3,imult,istate,ims,jmult,jstate,jms)
        for atom in range(natom):
            for xyz in range(3):
                string+='%s %s ' % (eformat(QMout['socdr'][i][j][atom][xyz].real,12,3),eformat(QMout['socdr'][i][j][atom][xyz].imag,12,3))
        string+='\n'
        string+=''
        j+=1
    i+=1
  return string

# ======================================================================= #
def writeQMouttime(QMin,QMout):
    '''Generates a string with the quantum mechanics total runtime in SHARC format.

    The string starts with a ! followed by a flag specifying the type of data. In the next line, the runtime is given

    Arguments:
    1 dictionary: QMin
    2 dictionary: QMout

    Returns:
    1 string: multiline string with the runtime'''

    string='! 8 Runtime\n%s\n' % (eformat(QMout['runtime'],9,3))
    return string


# =============================================================================================== #
# =============================================================================================== #
# =========================================== SUBROUTINES TO readQMin =========================== #
# =============================================================================================== #
# =============================================================================================== #

# ======================================================================= #
def checkscratch(SCRATCHDIR):
    '''Checks whether SCRATCHDIR is a file or directory. If a file, it quits with exit code 1, if its a directory, it passes. If SCRATCHDIR does not exist, tries to create it.

    Arguments:
    1 string: path to SCRATCHDIR'''

    exist=os.path.exists(SCRATCHDIR)
    if exist:
        isfile=os.path.isfile(SCRATCHDIR)
        if isfile:
            print '$SCRATCHDIR=%s exists and is a file!' % (SCRATCHDIR)
            sys.exit(15)
    else:
        try:
            os.makedirs(SCRATCHDIR)
        except OSError:
            print 'Can not create SCRATCHDIR=%s\n' % (SCRATCHDIR)
            sys.exit(16)

# ======================================================================= #
def removequotes(string):
  if string.startswith("'") and string.endswith("'"):
    return string[1:-1]
  elif string.startswith('"') and string.endswith('"'):
    return string[1:-1]
  else:
    return string

# ======================================================================= #
def getsh2ADFkey(sh2ADF,key):
  i=-1
  while True:
    i+=1
    try:
      line=re.sub('#.*$','',sh2ADF[i])
    except IndexError:
      break
    line=line.split(None,1)
    if line==[]:
      continue
    if key.lower() in line[0].lower():
      return line
  return ['','']

# ======================================================================= #
def get_sh2ADF_environ(sh2ADF,key,environ=True,crucial=True):
  line=getsh2ADFkey(sh2ADF,key)
  if line[0]:
    LINE=line[1]
    LINE=removequotes(LINE).strip()
  else:
    if environ:
      LINE=os.getenv(key.upper())
      if not LINE:
        if crucial:
          print 'Either set $%s or give path to %s in SH2ADF.inp!' % (key.upper(),key.upper())
          sys.exit(17)
        else:
          return None
    else:
      if crucial:
        print 'Give path to %s in SH2adf.inp!' % (key.upper())
        sys.exit(18)
      else:
        return None
  LINE=os.path.expandvars(LINE)
  LINE=os.path.expanduser(LINE)
  if containsstring(';',LINE):
    print "$%s contains a semicolon. Do you probably want to execute another command after %s? I can't do that for you..." % (key.upper(),key.upper())
    sys.exit(19)
  return LINE

# ======================================================================= #
def get_pairs(QMinlines,i):
  nacpairs=[]
  while True:
    i+=1
    try:
      line=QMinlines[i].lower()
    except IndexError:
      print '"keyword select" has to be completed with an "end" on another line!'
      sys.exit(20)
    if 'end' in line:
      break
    fields=line.split()
    try:
      nacpairs.append([int(fields[0]),int(fields[1])])
    except ValueError:
      print '"nacdr select" is followed by pairs of state indices, each pair on a new line!'
      sys.exit(21)
  return nacpairs,i

# ======================================================================= #         OK
def readQMin(QMinfilename):
    '''Reads the time-step dependent information from QMinfilename. This file contains all information from the current SHARC job: geometry, velocity, number of states, requested quantities along with additional information. The routine also checks this input and obtains a number of environment variables necessary to run MOLCAS.

    Steps are:
    - open and read QMinfilename
    - Obtain natom, comment, geometry (, velocity)
    - parse remaining keywords from QMinfile
    - check keywords for consistency, calculate nstates, nmstates
    - obtain environment variables for path to MOLCAS and scratch directory, and for error handling

    Arguments:
    1 string: name of the QMin file

    Returns:
    1 dictionary: QMin'''

    # read QMinfile
    QMinlines=readfile(QMinfilename)
    QMin={}

    # Get natom
    try:
        natom=int(QMinlines[0])
    except ValueError:
        print 'first line must contain the number of atoms!'
        sys.exit(22)
    QMin['natom']=natom
    if len(QMinlines)<natom+4:
        print 'Input file must contain at least:\nnatom\ncomment\ngeometry\nkeyword "states"\nat least one task'
        sys.exit(23)

    # Save Comment line
    QMin['comment']=QMinlines[1]

    # Get geometry and possibly velocity (for backup-analytical non-adiabatic couplings)
    QMin['geo']=[]
    QMin['veloc']=[]
    hasveloc=True
    QMin['frozcore']=0
    QMin['Atomcharge']=0
    for i in range(2,natom+2):
        if not containsstring('[a-zA-Z][a-zA-Z]?[0-9]*.*[-]?[0-9]+[.][0-9]*.*[-]?[0-9]+[.][0-9]*.*[-]?[0-9]+[.][0-9]*', QMinlines[i]):
            print 'Input file does not comply to xyz file format! Maybe natom is just wrong.'
            sys.exit(24)
        fields=QMinlines[i].split()
        symb = fields[0].lower().title()
        QMin['frozcore']+=FROZENS[symb]
        QMin['Atomcharge']+=ATOMCHARGE[symb]
        for j in range(1,4):
            fields[j]=float(fields[j])
        QMin['geo'].append(fields[0:4])
        if len(fields)>=7:
            for j in range(4,7):
                fields[j]=float(fields[j])
            QMin['veloc'].append(fields[4:7])
        else:
            hasveloc=False
    if not hasveloc:
        QMin=removekey(QMin,'veloc')


    # Parse remaining file
    i=natom+1
    while i+1<len(QMinlines):
        i+=1
        line=QMinlines[i]
        line=re.sub('#.*$','',line)
        if len(line.split())==0:
            continue
        key=line.lower().split()[0]
        if 'savedir' in key:
            args=line.split()[1:]
        else:
            args=line.lower().split()[1:]
        if key in QMin:
            print 'Repeated keyword %s in line %i in input file! Check your input!' % (key,i+1)
            continue  # only first instance of key in QM.in takes effect
        if len(args)>=1 and 'select' in args[0]:
            pairs,i=get_pairs(QMinlines,i)
            QMin[key]=pairs
        else:
            QMin[key]=args

    if 'unit' in QMin:
        if QMin['unit'][0]=='angstrom':
            factor=1./au2a
        elif QMin['unit'][0]=='bohr':
            factor=1.
        else:
            print 'Dont know input unit %s!' % (QMin['unit'][0])
            sys.exit(25)
    else:
        factor=1./au2a

    for iatom in range(len(QMin['geo'])):
        for ixyz in range(3):
            QMin['geo'][iatom][ixyz+1]*=factor


    if not 'states' in QMin:
        print 'Keyword "states" not given!'
        sys.exit(26)
    # Calculate states, nstates, nmstates
    for i in range(len(QMin['states'])):
        QMin['states'][i]=int(QMin['states'][i])
    nstates=0
    nmstates=0
    for i in range(len(QMin['states'])):
        nstates+=QMin['states'][i]
        nmstates+=QMin['states'][i]*(i+1)
    QMin['nstates']=nstates
    QMin['nmstates']=nmstates

#    while len(QMin['states']) < 3:
#       QMin['states'].append(0)

    # Various logical checks
    if not 'states' in QMin:
        print 'Number of states not given in QM input file %s!' % (QMinfilename)
        sys.exit(27)

    possibletasks=['h','soc','dm','grad','overlap','dmdr','socdr','ion']
    if not any([i in QMin for i in possibletasks]):
        print 'No tasks found! Tasks are "h", "soc", "dm", "grad","dmdr", "socdr" and "overlap".'
        sys.exit(28)

    if 'samestep' in QMin and 'init' in QMin:
        print '"Init" and "Samestep" cannot be both present in QM.in!'
        sys.exit(29)

    if 'overlap' in QMin and 'init' in QMin:
        print '"overlap" cannot be calculated in the first timestep! Delete either "overlap" or "init"'
        sys.exit(30)

    if not 'init' in QMin and not 'samestep' in QMin:
        QMin['newstep']=[]

    if not any([i in QMin for i in ['h','soc','dm','grad']]) and 'overlap' in QMin:
        QMin['h']=[]

    restrmult=[0,2]
    restr=True
    nunr=0
    nmults=0
    for mult,nstate in enumerate(QMin['states']):
        if nstate>0:
            nmults+=1
#            if not mult in restrmult:
#                nunr+=1
#                restr=False
    if nmults>1:
        bla=[ (n!=0 and not mult in restrmult) for mult,n in enumerate(QMin['states']) ]
        #print bla
        if any( bla ):
            print "Only single unrestricted multiplicity runs are supported"
            sys.exit(31)
        else:
            restr=True
    if nmults==1:
        if QMin['states'][0]!=0:
            restr=True
        else:
            restr=False
    if restr:
        print "Running restricted calculation."
        QMin['unr']='no'
    else:
        print "Running unrestricted calculation."
        QMin['unr']='yes'

#    for i in range(len(QMin['states'])):
#        numberstates=QMin['states'][i]
#        numberunrmult=0
#        if i == 1 or i>=3:
#          if numberstates!=0:
#             numberunrmult=numberunrmult+1
#             QMin['unr']='yes'
#        if i == 2:
#           if QMin['states'][0]==0 and QMin['states'][1]==0:
#              numberunrmult=numberunrmult+1
#        if numberunrmult >= 2:
#           print "Only single unrestricted multiplicity runs are supported"
#           sys.exit(31)
#        if i == 0 or i==2:
#           if numberstates!=0 and numberunrmult >= 2:
#              print "Mixed restricted and unrestricted multiplicities are not currently supported"
#              sys.exit(32)
#
#    if QMin['states'][0] !=0 and QMin['states'][2]!=0:
#       print "Will run as a restricted calculation"
#       QMin['unr']='no'
#    if QMin['states'][0] ==0 and QMin['states'][2]!=0:
# #      if 'restricted' in QMin:
# #         print 'Will run a restricted triplet calculation and Number of singlet states is set to 1 (S0)'
# #         QMin['unr']='no'
# #         QMin['states'][0]=1        
# #      else:
#       print "Will run unrestricted triplet calculation"
#       QMin['unr']='yes'
#    if QMin['states'][0] !=0 and QMin['states'][2]==0:
#       print "Will run restricted singlet calculation"
#       QMin['unr']='no'
           
        
#    if len(QMin['states'])>3:
#        print 'Higher multiplicities than triplets are not supported!'
#        sys.exit(31)
#
#    if QMin['states'][1]!=0:
#        print 'Doublets are not supported'
#        sys.exit(32)

    if QMin['unr']=='yes' and 'soc' in QMin:
       QMin=removekey(QMin,'soc')
       QMin['h']=[]

    if QMin['unr']=='no' and QMin['states'][0]>1:
       if len(QMin['states'])<=2:
          QMin=removekey(QMin,'soc')
          QMin['h']=[]
       elif QMin['states'][2]==0:
          QMin=removekey(QMin,'soc')
          QMin['h']=[]

    if 'h' in QMin and 'soc' in QMin:
        QMin=removekey(QMin,'h')

    if 'nacdt' in QMin or 'nacdr' in QMin:
        print 'Within the SHARC-ADF interface couplings can only be calculated via the overlap method. "nacdr" and "nacdt" are not supported.'
        sys.exit(33)

    if 'ion' in QMin:
        print 'Ionization probabilities not implemented!'
        sys.exit(34)
     
    if 'dmdr' in QMin:
        print 'Dipole derivatives not currently supported'
        sys.exit(35)

    if 'socdr' in QMin:
        print 'Spin-orbit coupling derivatives are not implemented'
        sys.exit(36)

    # Check for correct gradient list
    if 'grad' in QMin:
        if len(QMin['grad'])==0 or QMin['grad'][0]=='all':
            QMin['grad']=[ i+1 for i in range(nmstates)]
            #pass
        else:
            for i in range(len(QMin['grad'])):
                try:
                    QMin['grad'][i]=int(QMin['grad'][i])
                except ValueError:
                    print 'Arguments to keyword "grad" must be "all" or a list of integers!'
                    sys.exit(37)
                if QMin['grad'][i]>nmstates:
                    print 'State for requested gradient does not correspond to any state in QM input file state list!'
                    sys.exit(38)

    # Process the overlap requests
    # identically to the nac requests
    if 'overlap' in QMin:
        if len(QMin['overlap'])>=1:
            nacpairs=QMin['overlap']
            for i in range(len(nacpairs)):
                if nacpairs[i][0]>nmstates or nacpairs[i][1]>nmstates:
                    print 'State for requested non-adiabatic couplings does not correspond to any state in QM input file state list!'
                    sys.exit(39)
        else:
            QMin['overlap']=[ [j+1,i+1] for i in range(nmstates) for j in range(i+1)]

    # obtain the statemap 
    statemap={}
    i=1
    for imult,istate,ims in itnmstates(QMin['states']):
        statemap[i]=[imult,istate,ims]
        i+=1
    QMin['statemap']=statemap

    # get the set of states for which gradients actually need to be calculated
    gradmap=set()
    if 'grad' in QMin:
        for i in QMin['grad']:
            gradmap.add( tuple(statemap[i][0:2]) )
    gradmap=list(gradmap)
    gradmap.sort()
    QMin['gradmap']=gradmap








    # open SH2ADF.inp
    sh2ADF=readfile('SH2ADF.inp')

    QMin['pwd']=os.getcwd()

    QMin['ADFHOME']=get_sh2ADF_environ(sh2ADF,'adfhome')
    os.environ['ADFHOME']=QMin['ADFHOME']
    QMin['scmlicense']=get_sh2ADF_environ(sh2ADF,'scmlicense')
    os.environ['SCMLICENSE']=QMin['scmlicense']
    os.environ['ADFBIN']=QMin['ADFHOME']+'/bin'
    os.environ['ADFRESOURCES']=QMin['ADFHOME']+'/atomicdata'
    os.environ['PATH']='$ADFBIN:'+os.environ['PATH']
    SCMTEMPDIR=get_sh2ADF_environ(sh2ADF,'scmtmpdir',True,False)
    if os.environ.get('PBS_JOBID') != None: 
       os.unsetenv('PBS_JOBID')
       if os.environ.get('PBS_ENVIRONMENT') != None:
          os.unsetenv('PBS_ENVIRONMENT')
    elif os.environ.get('SLURM_JOB_ID') != None:
       os.unsetenv('SLURM_JOB_ID')
       if os.environ.get('SLURM_JOBID') != None:
          os.unsetenv('SLURM_JOBID')
    elif os.environ.get('LSB_JOBID') != None:
       os.unsetenv.get('LSB_JOBID')
    elif os.environ.get('SGE_JOB_SPOOL_DIR') != None:
       os.unsetenv('SGE_JOB_SPOOL_DIR')
       if os.environ.get('PE_HOSTFILE')!= None:
          os.unsetenv('PE_HOSTFILE')
    elif os.environ.get('SCM_MACHINEFILE') != None:
       os.unsetenv('SCM_MACHINEFILE')
    elif SCMTEMPDIR != None:
       os.environ['SCM_TMPDIR']=SCMTEMPDIR

    padstates=get_sh2ADF_environ(sh2ADF,'paddingstates',False,False)

    numfrozencore=get_sh2ADF_environ(sh2ADF,'numfrozcore',False,False)
    if numfrozencore!= None:
       numfroz=int(numfrozencore)
       if numfroz==0:
          QMin['frozcore']=0 
       elif numfroz>0:
          numfrozcore=int(numfrozencore)
          QMin['frozcore']=int(numfrozcore) 

    if 'overlap' in QMin:
      QMin['wfoverlap']=get_sh2ADF_environ(sh2ADF,'wfoverlap',False,False)
      if QMin['wfoverlap']==None:
        ciopath=os.path.join(os.path.expandvars(os.path.expanduser('$SHARC')),'wfoverlap.x')
        if os.path.isfile(ciopath):
          QMin['wfoverlap']=ciopath
        else:
          print 'Give path to wfoverlap.x in SH2ADF.inp!'
          sys.exit(40)
    QMin['wfthres']=0.99 
    valthresh=get_sh2ADF_environ(sh2ADF,'wfthres',False,False)
    if valthresh != None:
       QMin['wfthres']= float(valthresh)


    # Set up scratchdir
    line=get_sh2ADF_environ(sh2ADF,'scratchdir',False,False)
    if line==None:
        line=QMin['pwd']+'/SCRATCHDIR/'
    line=os.path.expandvars(line)
    line=os.path.expanduser(line)
    line=os.path.abspath(line)
    #checkscratch(line)
    QMin['scratchdir']=line


    # Set up savedir
    if 'savedir' in QMin:
        # savedir may be read from QM.in file
        line=QMin['savedir'][0]
    else:
        line=get_sh2ADF_environ(sh2ADF,'savedir',False,False)
        if line==None:
            line=QMin['pwd']+'/SAVEDIR/'
    line=os.path.expandvars(line)
    line=os.path.expanduser(line)
    line=os.path.abspath(line)
    if 'init' in QMin:
        checkscratch(line)
    QMin['savedir']=line


    line=getsh2ADFkey(sh2ADF,'debug')
    if line[0]:
        if len(line)<=1 or 'true' in line[1].lower():
            global DEBUG
            DEBUG=True

    QMin['ncpu']=1
    line=getsh2ADFkey(sh2ADF,'ncpu')
    if line[0]:
        try:
            QMin['ncpu']=int(line[1])
        except ValueError:
            print 'Number of CPUs does not evaluate to numerical value!'
            sys.exit(41)

    NSLOTS=os.environ.get('NSLOTS')
    SLURM=os.environ.get('SLURM_NTASKS_PER_NODE')
    SHARC_NCPU=os.environ.get('SHARC_NCPU')
    if NSLOTS != None:
       QMin['ncpu']=int(NSLOTS)
    elif SHARC_NCPU!=None:
       QMin['ncpu']=int(SHARC_NCPU)
    elif SLURM != None:
       QMin['ncpu']=int(SLURM)

    if QMin['ncpu'] !=1:
      if QMin['ncpu']%2!=0:
        print 'Number of cpus is an odd number! The interface has reduced the number to be used by 1'
        QMin['ncpu']=int(QMin['ncpu'])-1

#    QMin['delay']=0.0
#    line=getsh2ADFkey(sh2ADF,'delay')
#    if line[0]:
#        try:
#            QMin['delay']=float(line[1])
#        except ValueError:
#            print 'Submit delay does not evaluate to numerical value!'
#            sys.exit(42)

#    line=getsh2ADFkey(sh2ADF,'always_orb_init')
#    if line[0]:
#        QMin['always_orb_init']=[]
#    line=getsh2ADFkey(sh2ADF,'always_guess')
#    if line[0]:
#        QMin['always_guess']=[]
#    if 'always_orb_init' in QMin and 'always_guess' in QMin:
#        print 'Keywords "always_orb_init" and "always_guess" cannot be used together!'
#        sys.exit(43)

    # open template
    template=readfile('ADF.template')
    QMin['tda']=False
    QMin['template']={}
    integers=[]
    strings =['save','print','relativistic','symmetry']
    keystrings=['sopert','stofit','exactdensity','gscorr','nosharedarrays','tda','unrestricted']
    floats=[]
    blocks=['basis','scf','xc','excitation','geometry','beckegrid','zlmfit','atoms','excitedgo','cosmo']
    
    a=-1
    for line in template:
        a=a+1
        line=re.sub('#.*$','',line).lower().split(None,1)
        if len(line)==0:
            continue
        elif 'charge' in line[0]:
            data=line[1].split()
            values = len(data)
#            if values == 2:
            if QMin['statemap'][1][0]%2==0:
               if QMin['Atomcharge']%2==0:
                  totnelec=float(data[0])+float(QMin['Atomcharge'])
                  if totnelec%2==0:
                     newcharge=float(data[0])+1.0
                     QMin['template']['charge']=float(newcharge)
               else:
                  QMin['template']['charge']=float(data[0])
            else:
               if QMin['Atomcharge']%2==0:
                  if QMin['unr']=='no':
                     totnelec=float(data[0])+float(QMin['Atomcharge'])
                     if totnelec%2==0:
                        QMin['template']['charge']=float(data[0])
                     else:
                        unel=totnelec%2
                        newcharge=float(data[0])-unel
                        print 'The charge on the system leads to %i unpaired electrons' % (int(unel))
                        print 'Charge of the system is set to %i' % (newcharge)
                        QMin['template']['charge']=float(newcharge)
                  else:
                     QMin['template']['charge']=float(data[0])
            if values == 2:
               if QMin['unr']=='no':
                  QMin['template']['unpelec']=float(0.0)
               else:
                  unpel=int(QMin['statemap'][1][0])-1 
                  QMin['template']['unpelec']=float(unpel)
#            else:
#               QMin['template']['charge']=float(data[0])
#        elif 'lowest' in line[0]
#            QMin['template']['nrexci']=int(line[1])
        elif line[0] in strings:
            QMin['template'][line[0]]=line[1]
        elif line[0] in keystrings:
            QMin['template'][line[0]]=line[0]
            if 'tda' in line[0]:
               QMin['tda']=True
        elif line[0] in blocks:
           if 'grad' in QMin and line[0]=='cosmo':
               continue
           else:
              l = template[a]
              l=re.sub('#.*$','',l).lower().split(None,1)
              block = [[]]
              block[0].append(l)
              i=0
              while l[0] != 'end':
                i=i+1
                l=template[a+i]
                l1=re.sub('#.*$','',l).split(None,1)
                l=re.sub('#.*$','',l).lower().split(None,1)
                if l[0] == 'lowest':
                   exci=QMin['states']
                   if QMin['unr']=='yes':
                      nstates=QMin['nstates']
                      l[1]=str(int(nstates)+int(padstates))+'\n'
                   else:
                      l[1]=str(int(exci[0]-1)+int(padstates))+'\n'
                      if len(exci)>=3 and exci[2]>exci[0]-1:
                          l[1]=str(int(exci[2])+int(padstates))+'\n'               
                   #elif int(exci[2])>=int(exci[0]-1):
                   #   l[1]=str(int(exci[2])+3)+'\n'
                   #else:
                   #   l[1]=str(int(exci[0]-1)+3)+'\n'
                   #print l[1]
                if str(l1[0]) in ELEMENTS:
                   block[0].append(l1)
                else:
                   block[0].append(l)
              QMin['template'][line[0]]=block[0]

    necessary=['basis','xc','excitation','save']
    for i in necessary:
        if not i in QMin['template']:
            print 'Key %s missing in template file!' % (i)
            sys.exit(44)

    

    QMin['gradmode']=1
    QMin['ncpu']=max(1,QMin['ncpu'])

    # Check the save directory
    try:
        ls=os.listdir(QMin['savedir'])
        err=0
    except OSError:
        err=1
    if 'init' in QMin:
        err=0
    elif 'samestep' in QMin:
        if not 'ADF.t21' in ls:
            print 'File "ADF.t21" missing in SAVEDIR!'
            err+=1
        if 'overlap' in QMin:
            if not 'ADF.t21.old' in ls:
                print 'File "ADF.t21.old" missing in SAVEDIR!' 
                err+=1
            if QMin['step'] > 1:
               if not mos.a.old or mos.b.old in ls:
                  print 'File "mos.a/b.old" missing in SAVEDIR'
                  err+=1
               if not det.a.old or det.b.old in ls:
                  print 'File "det.a/b.old" missing in SAVEDIR'
                  err+=1
    elif 'overlap' in QMin:
            if not 'ADF.t21' in ls:
                print 'File "ADF.t21" missing in SAVEDIR!'
                err+=1
    if err>0:
        print '%i files missing in SAVEDIR=%s' % (err,QMin['savedir'])
        sys.exit(47)


    if 'backup' in QMin:
      backupdir=QMin['savedir']+'/backup'
      backupdir1=backupdir
      i=0
      while os.path.isdir(backupdir1):
        i+=1
        if 'step' in QMin:
          backupdir1=backupdir+'/step%s_%i' % (QMin['step'][0],i)
        else:
          backupdir1=backupdir+'/calc_%i' % (i)
      QMin['backup']=backupdir

    if PRINT:
        printQMin(QMin)

    return QMin

# =============================================================================================== #
# =============================================================================================== #
# =========================================== gettasks and setup routines ======================= #
# =============================================================================================== #
# =============================================================================================== #

def gettasks(QMin):

    states=QMin['states']
    nstates=QMin['nstates']
    nmstates=QMin['nmstates']
    # Currently implemented keywords: soc, dm, grad, 
    tasks=[]
    #During initalization create all temporary directories and link them appropriately
    tasks.append(['mkdir',QMin['scratchdir']])
    tasks.append(['link',QMin['scratchdir'],QMin['pwd']+'/SCRATCH',False])
    tasks.append(['mkdir',QMin['scratchdir']+'/ADF'])
    tasks.append(['mkdir',QMin['scratchdir']+'/OVERLAP'])
    tasks.append(['mkdir',QMin['scratchdir']+'/GRAD']) 

    if 'init' in QMin:
      tasks.append(['mkdir',QMin['savedir']])
      tasks.append(['link', QMin['savedir'],QMin['pwd']+'/SAVE',False])

    if not 'samestep' in QMin and not 'init' in QMin:
      tasks.append(['movetoold'])
  
    if 'backup' in QMin:
      tasks.append(['mkdir',QMin['savedir']+'/backup/'])
      tasks.append(['mkdir',QMin['backup']])

    #Initial calculation of each step
    tasks.append(['writeinput'])
    tasks.append(['td-dft',QMin['template']])
    tasks.append(['getmocoef'])
    tasks.append(['get_CIcoef'])

    #Then calculate the needed gradients for the dynamics
    if 'grad'in QMin:
        tasks.append(['gradientcalculation',QMin['template']])

    #Runs the overlap calculations for the NACV's
    if not 'init' in QMin and 'overlap' in QMin:
        tasks.append(['check_supergeom'])
      #  tasks.append(['mkdir', QMin['scratchdir']+'/OVERLAP/AO_OVERL'])
        tasks.append(['ADF_AOcalc'])
        tasks.append(['get_Overlap_mat'])
        tasks.append(['run_WFOverlap'])
        tasks.append(['get_WFOverlap_out'])

    tasks.append(['get_ADF_out'])
    tasks.append(['createQMout'])
    if 'backup' in QMin:
        tasks.append(['backup_data',QMin['backup']])
    if 'overlap' in QMin:
        tasks.append(['cleanup',QMin['scratchdir']+'/OVERLAP'])
    if 'grad' in QMin:
        tasks.append(['cleanup',QMin['scratchdir']+'/GRAD'])
    tasks.append(['cleanup',QMin['scratchdir']+'/ADF'])

    if 'cleanup' in QMin:
        tasks.append(['cleanup',QMin['savedir']])
        tasks.append(['cleanup',QMin['scratchdir']])



    if DEBUG:
        printtasks(tasks)


    return tasks

# ======================================================================= #

def runeverything(tasks, QMin):

  if PRINT or DEBUG:
    print '=============> Entering RUN section <=============\n\n'
  
  QMout={}
  #states=QMin['states']
  #nstates=QMin['nstates']
  #nmstates=QMin['nmstates']
  for i in range(0,len(tasks)):
    task = tasks[i]
    if DEBUG:
      print task
    if task[0]=='movetoold':
      movetoold(QMin)
    if task[0]=='mkdir':
      mkdir(task[1])
    if task[0]=='link':
      if len(task)==4:
        link(task[1],task[2],task[3])
      else:
        link(task[1],task[2])
    if task[0]=='backup_data':
      backupdata(task[1],QMin)
    if task[0]=='writeinput':
      type = 0 
      write_ADFinput(type,QMin)
    if task[0]=='td-dft':
      run_tddft(QMin)
    if task[0]=='gradientcalculation':
      if len(QMin['gradmap'])>1:
         run_gradients(QMin)
    if task[0]=='get_ADF_out':
      get_adf_out(QMin,QMout)
    if task[0]=='check_supergeom':
      check_overlgeom(QMin)
    if task[0]=='ADF_AOcalc':
      type = 1
      write_ADFinput(type,QMin)
      run_smat(QMin)
    if task[0]=='get_Overlap_mat':
      get_smat(QMin)
    if task[0]=='getmocoef':
      get_mocoef(QMin)
    if task[0]=='get_CIcoef':
      get_cicoef(QMin)
    if task[0]=='run_WFOverlap':
      run_wfoverlap(QMin)
    if task[0]=='get_WFOverlap_out':
      get_wfoverlap(QMin,QMout)
#    if task[0]=='save_step_data':        
#      get_step_data(QMin)
    if task[0]=='createQMout':
      QMout=CreateQMout(QMin,QMout)
#    if task[0]=='cleanup':
#      cleandir(task[1])

  return QMout

# =============================================================================================== #
# =============================================================================================== #
# =========================================== SUBROUTINES TO RUNEVERYTING ======================= #
# =============================================================================================== #
# =============================================================================================== #

def mkdir(PATH):
  if os.path.exists(PATH):
    if os.path.isfile(PATH):
      print '%s exists and is a file!' % (PATH)
      sys.exit(48)
    else:
      ls=os.listdir(PATH)
      if not ls==[]:
        print 'INFO: %s exists and is a non-empty directory!' % (PATH)
        #sys.exit(49)
  else:
    os.makedirs(PATH)

# ======================================================================= #
def check_overlgeom(QMin):

    path=QMin['savedir']
    os.chdir(path)
    import kf
    oldgeom=kf.kffile('ADF.t21.old')
    newgeom=kf.kffile('ADF.t21')
    old = oldgeom.read("General","Input")
    new = newgeom.read("General",'Input')
    old = old[1:]
    new = new[1:]
    oldxyz = []
    newxyz = []
    supergeom=[]
    l=0
    while not 'end' in old[l]:
       oldxyz.append(old[l])
       newxyz.append(new[l])
       l=l+1
    for xyz in range(0,int(len(oldxyz))):
         supergeom.append(oldxyz[xyz])
    for xyz in range(0,int(len(oldxyz))):
        coords1=oldxyz[xyz].split()
        coords2=newxyz[xyz].split()
        for i in range(1,4):
            same=cmp(coords1[i],coords2[i])
            if same == 0:
               coords2[i]=float(coords2[i])+0.00000001
        string = str(coords2[0])+'.1  '+str(coords2[1])+'  '+str(coords2[2])+'  '+str(coords2[3])
        supergeom.append(string)
    QMin['supergeom']=supergeom

    return QMin

# ======================================================================= #

def movetoold(QMin):
  # rename all eivectors, mocoef, ADF.run
  saveable=['ADF.t21','cicoef','cicoef_S','cicoef_T','mocoef']
  savedir=QMin['savedir']
  ls=os.listdir(savedir)
  if ls==[]:
    return
  for f in ls:
    f2=savedir+'/'+f
    if os.path.isfile(f2):
      if any( [ i in f for i in saveable ] ):
        if not 'old' in f:
          fdest=f2+'.old'
          shutil.copy(f2,fdest)

# ======================================================================= #

#def link(PATH, NAME,crucial=True,force=False):
#  # do not create broken links
#  if not os.path.exists(PATH):
#    print 'Source %s does not exist, cannot create link!' % (PATH)
#    sys.exit(50)
#  # do ln -f only if NAME is already a link
#  if os.path.exists(NAME):
#    if os.path.islink(NAME) or force:
#      os.remove(NAME)
#    else:
#      print '%s exists, cannot create a link of the same name!' % (NAME)
#      if crucial:
#        sys.exit(51)
#      else:
#        return
#  if not os.path.exists(os.path.realpath(NAME)):
#    if os.path.islink(NAME):
#    # NAME is already a broken link
#      os.remove(NAME)
#    else:
#      return
#  os.symlink(PATH, NAME)

def link(PATH,NAME,crucial=True,force=True):
  # do not create broken links
  if not os.path.exists(PATH):
    print 'Source %s does not exist, cannot create link!' % (PATH)
    sys.exit(52)
  if os.path.islink(NAME):
    if not os.path.exists(NAME):
      # NAME is a broken link, remove it so that a new link can be made
      os.remove(NAME)
    else:
      # NAME is a symlink pointing to a valid file
      if force:
        # remove the link if forced to
        os.remove(NAME)
      else:
        print '%s exists, cannot create a link of the same name!' % (NAME)
        if crucial:
          sys.exit(53)
        else:
          return
  elif os.path.exists(NAME):
    # NAME is not a link. The interface will not overwrite files/directories with links, even with force=True
    print '%s exists, cannot create a link of the same name!' % (NAME)
    if crucial:
      sys.exit(54)
    else:
      return
  os.symlink(PATH, NAME)

# ======================================================================= #
def cleandir(directory):
  if DEBUG:
    print '===> Cleaning up directory %s\n' % (directory)
  for data in os.listdir(directory):
    path=directory+'/'+data
    if os.path.isfile(path) or os.path.islink(path):
      if DEBUG:
        print 'rm %s' % (path)
      try:
        os.remove(path)
      except OSError:
        print 'Could not remove file from directory: %s' % (path)
    else:
      if DEBUG:
        print ''
      cleandir(path)
      os.rmdir(path)
      if DEBUG:
        print 'rm %s' % (path)
  if DEBUG:
    print '\n'

# ======================================================================= #
def backupdata(backupdir,QMin):
  # save all files in savedir, except which have 'old' in their name
  ls=os.listdir(QMin['savedir'])
  for f in ls:
    ff=QMin['savedir']+'/'+f
    if os.path.isfile(ff) and not 'old' in ff:
      step = int(QMin['step'][0])
      fdest=backupdir+'/'+f+'.stp'+str(step)
      shutil.copy(ff,fdest)
#  # save molden files
#  if 'molden' in QMin:
#    for job in QMin['joblist']:
#      ff=os.path.join(QMin['savedir'],'MOLDEN',job,'step_%s.molden' % (QMin['step'][0]))
#      fdest=os.path.join(backupdir,'step_%s.%s.molden' % (QMin['step'][0],job.replace('/','_')))
#      shutil.copy(ff,fdest)

# ======================================================================= #
def write_ADFinput(type,QMin):

    if type == 1:
       path = QMin['scratchdir']+'/OVERLAP'
       os.chdir(path)
       outfile = open('ADF_AO_overlap.run','w')
       geom = QMin['supergeom']
       outfile.write('atoms \n')
       for l in range(0,int(len(geom))):
           outfile.write('%s\n'% (geom[l]))
       outfile.write('end\n\n')
       outfile.write('basis\n')
       for l in range(1,len(QMin['template']['basis'])-1):
           for i in range(0,len(QMin['template']['basis'][l])):
               if l!=1 and i != 1:
                  outfile.write(' %s'%(QMin['template']['basis'][l][i]))
               elif QMin['template']['basis'][l][0] in ELEMENTS:
                  outfile.write(' %s'%(QMin['template']['basis'][l][i]))
                  outfile.write('%s' %((QMin['template']['basis'][l][0])))
                  outfile.write('.1')
                  if i != 0:
                     outfile.write(' %s'%(QMin['template']['basis'][l][i]))
               else:
                  basisset = QMin['template']['basis'][l][i].upper()
                  outfile.write(' '+basisset)
       outfile.write('end\n\n')
       outfile.write('xc\n')
       for l in range(1,len(QMin['template']['xc'])-1):
           for i in range(0,len(QMin['template']['xc'][l])):
               outfile.write(' %s'%(QMin['template']['xc'][l][i]))
       outfile.write('end\n\n')
       outfile.write('save TAPE15\n\n')
       outfile.write('SYMMETRY NOSYM\n\n')
       outfile.write('beckegrid\nquality %s\nend\n\n'%(QMin['template']['beckegrid'][1][1]))
       outfile.write('SHARCOVERLAP \n\n')
       if 'nosharedarrays' in QMin['template']:
           outfile.write('nosharedarrays\n\n')
       if 'relativistic' in QMin['template']:
           outfile.write('relativistic %s\n'%(QMin['template']['relativistic']))
       if 'charge' in QMin['template']:
           char=float(QMin['template']['charge'])*2.0
           outfile.write('CHARGE %2.1f' %(float(char)))
           if 'unpelec' in QMin['template']:
               unp=float(QMin['template']['unpelec'])*2.0
               outfile.write(' %2.1f \n\n'%(float(unp)))
           else:
               outfile.write('\n')
       if QMin['unr']=='yes': 
#'unrestricted' in QMin['template']:
           outfile.write('unrestricted\n\n')
       if 'zlmfit' in QMin['template']:
           outfile.write('zlmfit\nquality %send\n\n'%(QMin['template']['zlmfit'][1][1]))
       elif 'stofit' in QMin['template']:
           outfile.write('STOFIT\n\n')
       elif 'exactdensity' in QMin['template']:
           outfile.write('EXACTDENSITY\n\n')
       if not DEBUG:
          outfile.write('NOPRINT LOGFILE')
    elif type == 0:
       geom = QMin['geo']
       for l in range(0,int(len(geom))):
           for i in range(1,4):
               geom[l][i] = float(geom[l][i])*float(au2a)
       path = QMin['scratchdir']+'/ADF'
       os.chdir(path)
       outfile = open('ADF_tddft.run','w')
       outfile.write('atoms \n' )
       for l in range(0,int(len(geom))):
           outfile.write('%s'% (geom[l][0]))
           for i in range(1,4):
               outfile.write('  %s  '% (geom[l][i]))
               if i == 3:
                  outfile.write(' \n')
       outfile.write('end\n\n')
       outfile.write('basis\n')
       for l in range(1,len(QMin['template']['basis'])-1):
           for i in range(0,len(QMin['template']['basis'][l])):
               if l!=1 and i != 1:
                  outfile.write(' %s'%(QMin['template']['basis'][l][i]))
               elif QMin['template']['basis'][l][0] in ELEMENTS:
                  outfile.write(' %s'%(QMin['template']['basis'][l][i]))
               else:
                  basisset = QMin['template']['basis'][l][i].upper()
                  outfile.write(' '+basisset)
       outfile.write('end\n\n')
       outfile.write('xc\n')
       for l in range(1,len(QMin['template']['xc'])-1):
           for i in range(0,len(QMin['template']['xc'][l])):
               outfile.write(' %s'%(QMin['template']['xc'][l][i]))
       outfile.write('end\n\n')
       if 'tda' in QMin['template']:
           outfile.write('tda\n\n')
       outfile.write('save TAPE21\n\n')
       outfile.write('SYMMETRY NOSYM\n\n')
       outfile.write('DEPENDENCY\n\n')
       outfile.write('beckegrid\nquality %s\nend\n\n'%(QMin['template']['beckegrid'][1][1]))
       if 'relativistic' in QMin['template']:
           outfile.write('relativistic %s \n\n'%(QMin['template']['relativistic']))
       if 'charge' in QMin['template']:
           outfile.write('CHARGE %2.1f' %(float(QMin['template']['charge'])))
           if 'unpelec' in QMin['template']:
               outfile.write(' %2.1f \n\n'%(float(QMin['template']['unpelec'])))
           else:
               outfile.write('\n\n')
       if QMin['unr']=='yes':
# or 'unrestricted' in QMin['template']:
           outfile.write('unrestricted\n\n')
       if 'zlmfit' in QMin['template']:
           outfile.write('zlmfit\nquality %send\n\n'%(QMin['template']['zlmfit'][1][1]))
       elif 'stofit' in QMin['template']:
           outfile.write('STOFIT\n\n')
       elif 'exactdensity' in QMin['template']:
           outfile.write('EXACTDENSITY\n\n')
       outfile.write('excitation\n')
       for l in range(1,len(QMin['template']['excitation'])-1):
           for i in range(0,len(QMin['template']['excitation'][l])):
               outfile.write(' %s'%(QMin['template']['excitation'][l][i]))
               if 'davidson' in QMin['template']['excitation'][l][i]:
                  outfile.write('\n')
       if QMin['unr']=='no':
          if len(QMin['states'])<=2:
             outfile.write('ONLYSING\n') 
          elif QMin['states'][2]==0:
             outfile.write('ONLYSING\n')
#       if QMin['states'][0]==1 and QMin['states'][2]!=0:
#          outfile.write('ONLYTRIP\n')
       outfile.write('end\n\n')
       if len(QMin['gradmap'])>=1:
          outfile.write('GEOMETRY\n iterations 0\nEND\n\n')
          el=list(QMin['gradmap'][-1])
          string=IToMult[el[0]][0]+'%i'% (el[1]-(el[0]<=2))
          if QMin['unr']=='yes':
             if QMin['states'][1]!=0:
                if string != 'D0':
                   outfile.write('EXCITEDGO\n')
                   outfile.write('SINGLET\n')
                   outfile.write('STATE A %i\n'%(el[1]-1))
             elif el[0] != 2:
                   outfile.write('EXCITEDGO\n')
                   outfile.write('SINGLET\n')
                   outfile.write('STATE A %i\n'%(el[1]-1))
             outfile.write('OUTPUT = 4\n')
             outfile.write('CPKS EPS=0.0001\n')
             outfile.write('END\n\n')
          else:
             if string != 'S0':
                outfile.write('EXCITEDGO\n')
                if 'S' in string:
                    outfile.write('SINGLET\n')
                    outfile.write('STATE A %i\n'%(el[1]-1))
                if 'T' in string:
                    outfile.write('TRIPLET\n')
                    outfile.write('STATE A %i\n'%(el[1]))
                outfile.write('OUTPUT = 4\n')
                outfile.write('CPKS EPS=0.0001\n')
                outfile.write('END\n\n')
       if QMin['unr']=='no':
          if len(QMin['states'])>=3:
             if QMin['states'][2]!=0:
                if 'sopert' in QMin['template']:
                    outfile.write('SOPERT\nGSCORR\nPRINT SOMATRIX\n\n')
       if 'init' in QMin:
           filename=QMin['pwd']+'/ADF.t21_init'
           if os.path.isfile(filename):
               outfile.write('RESTART ADF.t21 &\nnogeo\nEND\n\n')
       else:
           outfile.write('RESTART ADF.t21 &\nnogeo\nEND\n\n')
       outfile.write('SCF\niterations %sEND\n\n' %(QMin['template']['scf'][1][1]))
       if not DEBUG:
           outfile.write('NOPRINT LOGFILE')
       new_gradmap_length=len(QMin['gradmap'])-1
       gradmap=QMin['gradmap'][:new_gradmap_length]
       if 'grad' in QMin:
           for i,el in enumerate(gradmap):
               string=IToMult[el[0]][0]+'%i'% (el[1]-(el[0]<=2))
               mkdir(QMin['scratchdir']+'/GRAD/'+string)
               gradpath = QMin['scratchdir']+'/GRAD/'+string
               os.chdir(gradpath)
               outfile = open('ADF_grad'+string+'.run','w')
               outfile.write('atoms \n')
               for l in range(0,int(len(geom))):
                   outfile.write('%s'% (geom[l][0]))
                   for i in range(1,4):
                       outfile.write('  %s  '% (geom[l][i]))
                       if i == 3:
                          outfile.write(' \n')
               outfile.write('end\n\n')
               outfile.write('basis\n')
               for l in range(1,len(QMin['template']['basis'])-1):
                   for i in range(0,len(QMin['template']['basis'][l])):
                       if l!=1 and i != 1:
                          outfile.write(' %s'%(QMin['template']['basis'][l][i]))
                       elif QMin['template']['basis'][l][0] in ELEMENTS:
                          outfile.write(' %s'%(QMin['template']['basis'][l][i]))
                       else:
                          basisset = QMin['template']['basis'][l][i].upper()
                          outfile.write(' '+basisset)
               outfile.write('end\n\n')
               outfile.write('xc\n')
               for l in range(1,len(QMin['template']['xc'])-1):
                   for i in range(0,len(QMin['template']['xc'][l])):
                       outfile.write(' %s'%(QMin['template']['xc'][l][i]))
               outfile.write('end\n\n')
               outfile.write('save TAPE21\n\n')
               outfile.write('SYMMETRY NOSYM\n\n')
               outfile.write('DEPENDENCY\n\n')
               outfile.write('beckegrid\nquality %send\n\n'%(QMin['template']['beckegrid'][1][1]))
               if 'relativistic' in QMin['template']:
                   outfile.write('relativistic %s'%(QMin['template']['relativistic']))
               if 'charge' in QMin['template']:
                   outfile.write('CHARGE %2.1f' %(float(QMin['template']['charge'])))
                   if 'unpelec' in QMin['template']:
                       outfile.write(' %2.1f \n'%(float(QMin['template']['unpelec'])))
                   else:
                       outfile.write('\n')
               if QMin['unr']=='yes':
# or 'unrestricted' in QMin['template']:
                   outfile.write('unrestricted\n\n')
               if 'zlmfit' in QMin['template']:
                   outfile.write('zlmfit\nquality %send\n\n'%(QMin['template']['zlmfit'][1][1]))
               elif 'stofit' in QMin['template']:
                   outfile.write('STOFIT\n\n')
               elif 'exactdensity' in QMin['template']:
                   outfile.write('EXACTDENSITY\n\n')
               outfile.write('excitation\n')
               for l in range(1,len(QMin['template']['excitation'])-1):
                   for i in range(0,len(QMin['template']['excitation'][l])):
                       outfile.write(' %s'%(QMin['template']['excitation'][l][i]))
                       if 'davidson' in QMin['template']['excitation'][l][i]:
                          outfile.write('\n')
               if el[0] == 3 and QMin['unr']=='no':
                  outfile.write('ONLYTRIP\n')
               else:
                  outfile.write('ONLYSING\n')
               outfile.write('end\n\n')
               if 'tda' in QMin['template']:
                   outfile.write('tda\n\n')
               outfile.write('GEOMETRY\n iterations 0\nEND\n\n')
               if QMin['unr']=='yes':
                  if QMin['states'][1]!=0:
                     if string != 'D0':
                        outfile.write('EXCITEDGO\n')
                        outfile.write('SINGLET\n')
                        outfile.write('STATE A %i\n'%(el[1]-1))
                  elif el[0] != 2: 
                        outfile.write('EXCITEDGO\n')
                        outfile.write('SINGLET\n')
                        outfile.write('STATE A %i\n'%(el[1]-1))
                  outfile.write('OUTPUT = 4\n')
                  outfile.write('CPKS EPS=0.0001\n')
                  outfile.write('END\n\n')
               else:
                  if string != 'S0':
                     outfile.write('EXCITEDGO\n')
                     if 'S' in string:
                         outfile.write('SINGLET\n')
                         outfile.write('STATE A %i\n'%(el[1]-1))
                     if 'T' in string:
                         outfile.write('TRIPLET\n')
                         outfile.write('STATE A %i\n'%(el[1]))
                     outfile.write('OUTPUT = 4\n')
                     outfile.write('CPKS EPS=0.0001\n')
                     outfile.write('END\n\n')
               outfile.write('RESTART ADF.t21 &\nnogeo\nEND\n\n')
               outfile.write('SCF\niterations %sEND\n\n' %(QMin['template']['scf'][1][1]))           
               if not DEBUG:
                   outfile.write('NOPRINT LOGFILE')
    outfile.close()

# ======================================================================= #
def runProgram(string,workdir):
  prevdir=os.getcwd()
  if DEBUG:
    print workdir
  os.chdir(workdir)
  if PRINT or DEBUG:
    starttime=datetime.datetime.now()
    sys.stdout.write('START:\t%s\t%s\t"%s"\n' % (workdir,starttime,string))
    sys.stdout.flush()
  try:
    runerror=sp.call(string,shell=True)
  except OSError:
    print 'Call have had some serious problems:',OSError
    sys.exit(55)
  if PRINT or DEBUG:
    endtime=datetime.datetime.now()
    sys.stdout.write('FINISH:\t%s\t%s\tRuntime: %s\tError Code: %i\n' % (workdir,endtime,endtime-starttime,runerror))
  os.chdir(prevdir)
  return runerror

# ======================================================================= #
def run_tddft(QMin):

   workdir = QMin['scratchdir']+'/ADF'
   savedir = QMin['savedir']
   os.environ['NSCM']=str(QMin['ncpu'])
   if 'init' in QMin:
      filename=QMin['pwd']+'/ADF.t21_init'
      if os.path.isfile(filename):
         shutil.copy(QMin['pwd']+'/ADF.t21_init',workdir+'/ADF.t21')
   else:
      shutil.copy(savedir+'/ADF.t21',workdir)
   string = '$ADFBIN/adf -n %i <ADF_tddft.run > ADF_tddft.out'%(QMin['ncpu'])
   runerror=runProgram(string,workdir)
   if runerror == 0:
      os.chdir(workdir)
      shutil.move(workdir+'/TAPE21',workdir+'/ADF.t21')
      shutil.copy(workdir+'/ADF.t21',savedir+'/ADF.t21')      

   #copies the logfile if crashed
   if runerror!=0:
      print 'ADF calculation crashed! Error code = %i'%(runerror)
      s=QMin['savedir']+'/ADF-debug/'
      s1=s
      i=0
      while os.path.exists(s1):
        i+=1
        s1=s+'%i/' % (i)
      if PRINT or DEBUG:
        print '=> Saving all text files from WORK directory to %s\n' % (s1)
      os.mkdir(s1)

      dirs=[
          os.path.join(QMin['scratchdir'],'ADF')]
      maxsize=3*1024**2
      for d in dirs:
        ls=os.listdir(d)
        if DEBUG:
          print d
          print ls
        for i in ls:
          f=os.path.join(d,i)
          if os.stat(f)[6]<=maxsize and not os.path.isdir(f):
            try:
              shutil.copy(f,s1)
            except OSError:
              pass
            if PRINT or DEBUG:
              print i
      sys.exit(56)

## ======================================================================= #
def run_gradients(QMin):
   
   errorcodes = {}
   numjobs = int(len(QMin['gradmap']))-1
   ncpu = int(QMin['ncpu'])
   maxjobs = math.ceil(float(ncpu)/2.0)
   if QMin['gradmap'][0] == 1 and numjobs >1 :
      if QMin['gradmap'][0][0]==1:
         numjobs=numjobs-1
   if numjobs>=maxjobs:
      nproc = maxjobs
      if ncpu > 1:
         ncores=2
      else:
         ncores=1
   if numjobs<maxjobs:
      nproc=numjobs
      ncores=int(ncpu/numjobs)
   pool=Pool(processes=int(nproc))
   for i in range(0,int(numjobs)):
      string=''
      if QMin['gradmap'][i][0] == 1:
         string+='S'
         exci = QMin['gradmap'][i][1]-1
         string+=str(exci)
      elif QMin['gradmap'][i][0] == 2:
         string+='D'
         exci = QMin['gradmap'][i][1]-1
         string+=str(exci)
      elif QMin['gradmap'][i][0] >= 3:
         Multip=QMin['gradmap'][i][0]
         string+=IToMult[Multip][0]
         exci = QMin['gradmap'][i][1]
         string+=str(exci)
      if 'S0' in string and len(QMin['gradmap'])>1:
          continue
      if QMin['unr']=='yes' and 'D0' in string and len(QMin['gradmap'])>1:
          continue
      if QMin['unr']=='yes' and QMin['gradmap'][i][1]==1 and len(QMin['gradmap'])>1:
          continue
      workdir = QMin['scratchdir']+'/GRAD/'+string+'/'
      shutil.copy(QMin['savedir']+'/ADF.t21',workdir+'ADF.t21')
      os.environ['NSCM']=str(ncores)
      string2 = '$ADFBIN/adf -n '+str(ncores)+' <ADF_grad'+string+'.run > ADF_grad'+string+'.out'
      errorcodes[string] = pool.apply_async(runProgram, [string2,workdir])
   pool.close()
   pool.join()

   for i in errorcodes:
       errorcodes[i]=errorcodes[i].get()

   if PRINT:
       string='  '+'='*40+'\n'
       string+='||'+' '*40+'||\n'
       string+='||'+' '*10+'All Tasks completed!'+' '*10+'||\n'
       string+='||'+' '*40+'||\n'
       string+='  '+'='*40+'\n'
       print string
       j=0
       string='Error Codes:\n\n'
       for i in errorcodes:
           string+='\t%s\t%i' % (i+' '*(10-len(i)),errorcodes[i])
           j+=1
           if j==4:
               j=0
               string+='\n'
       print string

   if any((i!=0 for i in errorcodes.values())):
       print 'Some subprocesses did not finish successfully!'


## ======================================================================= #
def get_adf_out(QMin,QMout):

   if QMin['unr']=='yes':
      shutil.copy(QMin['scratchdir']+'/OVERLAP/cicoef.b', QMin['savedir']+'/cicoef')
   else:
      shutil.copy(QMin['scratchdir']+'/OVERLAP/cicoef_S.b', QMin['savedir']+'/cicoef_S')
      if len(QMin["states"])>=3 and QMin['states'][2]!=0:
         shutil.copy(QMin['scratchdir']+'/OVERLAP/cicoef_T.b', QMin['savedir']+'/cicoef_T')
   shutil.copy(QMin['scratchdir']+'/OVERLAP/mocoef.b', QMin['savedir']+'/mocoef')

   QMout['dipolemoments']={} 
   nstates = QMin['nstates']
   nmstates = QMin['nmstates']
   os.chdir(QMin['scratchdir'])
   import kf
   file = kf.kffile('ADF/ADF.t21')
   GS_energy = float(file.read('Energy','Bond Energy'))
   Sing_energies = file.read('Excitations SS A', 'excenergies')
   if QMin['unr']=='no' and len(QMin["states"])>=3 and QMin['states'][2]!=0:
      Trip_energies = file.read('Excitations ST A', 'excenergies')
   State_energies = []
   State_energies.append(GS_energy)
   if QMin['unr']=='yes':
      for a in range(1,nstates):
          E=GS_energy+float(Sing_energies[a-1])
          State_energies.append(E)
      multip=int(nmstates)/int(nstates)
      for a in range(1,multip):
          State_energies.append(GS_energy)
          for b in range(1,nstates):
              E=GS_energy+float(Sing_energies[b-1])
              State_energies.append(E)
   if QMin['states'][0] !=0 and QMin['unr']=='no':
      for a in range(1,QMin['states'][0]):
          E=GS_energy+float(Sing_energies[a-1])
          State_energies.append(E)
   if len(QMin["states"])>=3 and QMin['states'][2]!=0 and QMin['unr']=='no': 
      for a in range(0,QMin['states'][2]):
          E=GS_energy+float(Trip_energies[a])
          State_energies.append(E)
      for a in range(0,2):
          End = QMin['states'][0]+QMin['states'][2]
          Triplets = State_energies[QMin['states'][0]:End]
          for b in range(QMin['states'][2]):
              State_energies.append(Triplets[b])
   if 'dm' in QMin:
      Excited_dipole= file.read('Excitations SS A', 'transition dipole moments')
      Ground_dipole=file.read('Properties', 'Dipole')
#      file1=open('ADF/ADF_tddft.out')
#      f=file1.readlines()
#      l = -1
#      GS_dipole = []
#      for line in f :
#          l = l+1
#          sec = re.search('Dipole Moment\s*\**\s*\(Debye\)',line)
#          if sec !=None:
#             GS_dip =f[l+3].split()
#             GS_dipole = GS_dip[2:]
      GS_dipole = []
      for xyz in range(3):
          GS_dipole.append(Ground_dipole[xyz])
      QMout['dipolemoments']['GS']=GS_dipole
      c=-3
      if QMin['unr']=='yes':
        for a in range(0,nstates-1):
            b=a+1
            c=c+3
            Multip=int(QMin['template']['unpelec'])+1
            if Multip >=3:
               b=b+1
            statename =IToMult[Multip][0]+str(b)+'_GS'
            Excdipole = []
            for xyz in range(3):
                Excdipole.append(Excited_dipole[c+xyz])
            QMout['dipolemoments'][statename] = Excdipole
      else:
        for a in range(0,QMin['states'][0]-1):
            b=a+1
            c=c+3
            statename = 'S'+str(b)+'_GS'
            Excdipole = []
            for xyz in range(3):
                Excdipole.append(Excited_dipole[c+xyz])
            QMout['dipolemoments'][statename] = Excdipole
   if 'h' in QMin:
      h=makecmatrix(nmstates,nmstates)
      for i in range(0,nmstates):
          for j in range(0,nmstates):
              h[0][0]=GS_energy
              if i==j!=0:
                 if i < QMin['states'][0]:
                    h[i][i]=State_energies[i]
                 else:
                    h[i][i]=State_energies[i]
      QMout['h']=h
   if 'soc' in QMin:
      getSOCM(QMin,QMout)
      SOCM = QMout['h']
      for i in range(nmstates):
          SOCM[i][i]+=GS_energy
      QMout['h']=SOCM
   if 'grad' in QMin:
      QMout['gradients']={}
      for i,el in enumerate(QMin['gradmap']):
          string=IToMult[el[0]][0]+'%i'% (el[1]-(el[0]<=2))
          Excited_state_dipole = []
          if 'S0' in string and len(QMin['gradmap'])>1:
             continue
          if QMin['unr']=='yes' and 'D0' in string and len(QMin['gradmap'])>1:
              continue
          if QMin['unr']=='yes' and QMin['gradmap'][i][1]==1 and len(QMin['gradmap'])>1:
              continue
          if QMin['gradmap'][i]==QMin['gradmap'][-1]:
             file2=open('ADF/ADF_tddft.out')
          else:
             file2=open('GRAD/'+string+'/ADF_grad'+string+'.out')
          f=file2.readlines()
          natom =QMin['natom']
          if QMin['unr']=='yes':
            if QMin['gradmap'][i][0]>=3:
               GS_grad_name=IToMult[el[0]][0]+'1'
               if QMin['gradmap'][i][1] != 1:
                  #GS_grad_name=IToMult[el[0]][0]+'1' 
                  if not GS_grad_name in QMout['gradients']:
                     GS_Grad = []
                     l2 = -1
                     for line in f:
                         l2 = l2+1
                         GS_Gradient = re.search('Ground state gradients:',line)
                         if GS_Gradient != None:
                            for lines in f[l2+5:l2+5+natom]:
                               line_split = lines.split()
                               for xyz in range(3):
                                   line_split[2+xyz]=float(line_split[2+xyz])*au2a
                               GS_Grad.append(line_split[2:])
                            QMout['gradients'][GS_grad_name]=GS_Grad
                  l1 = -1
                  Grad = []
                  for line in f:
                      l1 = l1+1
                      sec = re.search('\s*Excited\s*state\s*dipole\s*moment\s*=\s*(-?[\d\.]+)\s*(-?[\d\.]+)\s*(-?[\d\.]+)',line)
                      if sec != None:
                         excix=float(sec.group(1))/2.54
                         Excited_state_dipole.append(excix)
                         exciy=float(sec.group(2))/2.54
                         Excited_state_dipole.append(exciy)
                         exciz=float(sec.group(3))/2.54
                         Excited_state_dipole.append(exciz)
                         QMout['dipolemoments'][string]=Excited_state_dipole
                      Gradient = re.search('Energy gradients wrt nuclear displacements',line)
                      if Gradient != None:
                         for lines in f[l1+6:l1+6+natom]:
                            line_split = lines.split()
                            for xyz in range(3):
                                line_split[2+xyz]=float(line_split[2+xyz])*au2a
                            Grad.append(line_split[2:])
                  QMout['gradients'][string]=Grad
               elif GS_grad_name in string and len(QMin['gradmap'])==1:
                  l1 = -1
                  Grad = []
                  for line in f:
                      l1 = l1+1
                      Gradient = re.search('Energy gradients wrt nuclear displacements',line)
                      if Gradient != None:
                         for lines in f[l1+6:l1+6+natom]:
                            line_split = lines.split()
                            for xyz in range(3):
                                line_split[2+xyz]=float(line_split[2+xyz])*au2a
                            Grad.append(line_split[2:])
                  QMout['gradients'][string]=Grad
            elif not 'D0' in string:
               if QMin['gradmap'][i][0]==2:
                  if not 'D0' in QMout['gradients']:
                    GS_Grad = []
                    l2 = -1
                    for line in f:
                        l2 = l2+1
                        GS_Gradient = re.search('Ground state gradients:',line)
                        if GS_Gradient != None:
                           for lines in f[l2+5:l2+5+natom]:
                              line_split = lines.split()
                              for xyz in range(3):
                                  line_split[2+xyz]=float(line_split[2+xyz])*au2a
                              GS_Grad.append(line_split[2:])
                           QMout['gradients']['D0']=GS_Grad
               l1 = -1
               Grad = []
               for line in f:
                   l1 = l1+1
                   sec = re.search('\s*Excited\s*state\s*dipole\s*moment\s*=\s*(-?[\d\.]+)\s*(-?[\d\.]+)\s*(-?[\d\.]+)',line)
                   if sec != None:
                      excix=float(sec.group(1))/2.54
                      Excited_state_dipole.append(excix)
                      exciy=float(sec.group(2))/2.54
                      Excited_state_dipole.append(exciy)
                      exciz=float(sec.group(3))/2.54
                      Excited_state_dipole.append(exciz)
                      QMout['dipolemoments'][string]=Excited_state_dipole
                   Gradient = re.search('Energy gradients wrt nuclear displacements',line)
                   if Gradient != None:
                      for lines in f[l1+6:l1+6+natom]:
                         line_split = lines.split()
                         for xyz in range(3):
                             line_split[2+xyz]=float(line_split[2+xyz])*au2a
                         Grad.append(line_split[2:])
               QMout['gradients'][string]=Grad
            elif 'D0' in string and len(QMin['gradmap'])==1:
               l1 = -1
               Grad = []
               for line in f:
                   l1 = l1+1
                   Gradient = re.search('Energy gradients wrt nuclear displacements',line)
                   if Gradient != None:
                      for lines in f[l1+6:l1+6+natom]:
                         line_split = lines.split()
                         for xyz in range(3):
                             line_split[2+xyz]=float(line_split[2+xyz])*au2a
                         Grad.append(line_split[2:])
               QMout['gradients'][string]=Grad
          else:
            if not 'S0' in string:
#               if QMin['gradmap'][0][0]==1:
               if not 'S0' in QMout['gradients']:
                  GS_Grad = []
                  l2 = -1
                  for line in f:
                      l2 = l2+1
                      GS_Gradient = re.search('Ground state gradients:',line)
                      if GS_Gradient != None:
                         for lines in f[l2+5:l2+5+natom]:
                            line_split = lines.split()
                            for xyz in range(3):
                                line_split[2+xyz]=float(line_split[2+xyz])*au2a
                            GS_Grad.append(line_split[2:])
                         QMout['gradients']['S0']=GS_Grad
               l1 = -1
               Grad = []
               for line in f:
                   l1 = l1+1
                   sec = re.search('\s*Excited\s*state\s*dipole\s*moment\s*=\s*(-?[\d\.]+)\s*(-?[\d\.]+)\s*(-?[\d\.]+)',line)
                   if sec != None:
                      excix=float(sec.group(1))/2.54
                      Excited_state_dipole.append(excix)
                      exciy=float(sec.group(2))/2.54
                      Excited_state_dipole.append(exciy)
                      exciz=float(sec.group(3))/2.54
                      Excited_state_dipole.append(exciz)
                      QMout['dipolemoments'][string]=Excited_state_dipole
                   Gradient = re.search('Energy gradients wrt nuclear displacements',line)
                   if Gradient != None:
                      for lines in f[l1+6:l1+6+natom]:
                         line_split = lines.split()
                         for xyz in range(3):
                             line_split[2+xyz]=float(line_split[2+xyz])*au2a
                         Grad.append(line_split[2:])
               QMout['gradients'][string]=Grad
            elif 'S0' in string and len(QMin['gradmap'])==1:
               l1 = -1
               Grad = []
               for line in f:
                   l1 = l1+1
                   Gradient = re.search('Energy gradients wrt nuclear displacements',line)
                   if Gradient != None:
                      for lines in f[l1+6:l1+6+natom]:
                         line_split = lines.split()
                         for xyz in range(3):
                             line_split[2+xyz]=float(line_split[2+xyz])*au2a
                         Grad.append(line_split[2:])
               QMout['gradients'][string]=Grad
   return QMout

## ======================================================================= #
def getSOCM(QMin,QMout):
    
  file = open('ADF/ADF_tddft.out')
  import kf
  file1 = kf.kffile('ADF/ADF.t21')
  nrexci = int(file1.read('All excitations','nr excitations'))
  nmstates = 4*nrexci+1
  
  f=file.readlines() 

  line_num1 = 0
  line_num2 = 0
  
  line_num=-1
  for line in f:
      line_num = line_num+1
      SO_real = re.search('SO matrix real part',line)
      SO_imag = re.search('SO matrix imaginary part',line)
      if SO_real != None:
         line_num1=line_num+4
      if SO_imag != None:
         line_num2 = line_num+4

  SO_real_part_diag = []
  SO_imag_part_diag = []
  nrcycles = math.ceil(float(nmstates)/4.0)
  for i in range(0,int(nrcycles)):
     nrlines = nmstates-(i*4)
     endlines = line_num1+nrlines
     Col1 = []
     Col2 = []
     Col3 = []
     Col4 = []
     a=-1
     for lines in f[line_num1:endlines]:
         a=a+1
         SO_mat = lines.split(None)
         if a >= 0:
            Col1.append(SO_mat[1])
         if a >= 1:
            Col2.append(SO_mat[2])
         if a >= 2:
            Col3.append(SO_mat[3])
         if a >= 3:
            Col4.append(SO_mat[4])
     SO_real_part_diag.append(Col1)
     if i <= int(nrexci)-1:
        SO_real_part_diag.append(Col2)
        SO_real_part_diag.append(Col3)
        SO_real_part_diag.append(Col4)
     line_num1=endlines+3
  for i in range(0,int(nrcycles)):
     nrlines = nmstates-(i*4)
     endlines = line_num2+nrlines
     Col1 = []
     Col2 = []
     Col3 = []
     Col4 = []
     a=-1
     for lines in f[line_num2:endlines]:
         a=a+1
         SO_mat = lines.split(None)
         if a >= 0:
            Col1.append(SO_mat[1])
         if a >= 1:
            Col2.append(SO_mat[2])
         if a >= 2:
            Col3.append(SO_mat[3])
         if a >= 3:
            Col4.append(SO_mat[4])
     SO_imag_part_diag.append(Col1)
     if i <= int(nrexci)-1:
        SO_imag_part_diag.append(Col2)
        SO_imag_part_diag.append(Col3)
        SO_imag_part_diag.append(Col4)
     line_num2=endlines+3

  SO_real_part_square = []
  for i in range(0,nmstates):
      Column= []
      for j in range(0,nmstates):
          if j == i:
             val = float(SO_real_part_diag[i][0])
             if abs(val) <=1.0e-15:
                val = 0.00000000
             Column.append(val)
          elif j < i :
             val = float(SO_real_part_diag[j][i-j])
             if abs(val) <=1.0e-15:
                val = 0.00000000
             Column.append(val)
          elif j > i :
             val = float(SO_real_part_diag[i][j-i])
             if abs(val) <=1.0e-15:
                val = 0.00000000
             Column.append(val)
      SO_real_part_square.append(Column)
  
  SO_imag_part_square = []
  for i in range(0,nmstates):
      Column= []
      for j in range(0,nmstates):
          if j == i:
             val = float(SO_imag_part_diag[i][0])
             if abs(val) <=1.0e-15:
                val = 0.0000000
             Column.append(val)
          elif j < i :
             SO_imag_part = -1*float(SO_imag_part_diag[j][i-j])
             val = SO_imag_part
             if abs(val) <=1.0e-15:
                val = 0.0000000
             Column.append(val)
          elif j>i:
             val = float(SO_imag_part_diag[i][j-i])
             if abs(val) <=1.0e-15:
                val = 0.000000
             Column.append(val)
      SO_imag_part_square.append(Column)

  sharcnm=int(QMin['nmstates'])
  sharcsings=int(QMin['states'][0])
  sharctrip=int(QMin['states'][2]) 
  socm = makecmatrix(sharcnm,sharcnm)
  socm_real = makecmatrix(sharcnm,sharcnm)
  socm_imag = makecmatrix(sharcnm,sharcnm)

  reordermap=[(nrexci*4)]
  for i in range(1,sharcsings):
      reordermap.append(i-1)
  for i in range(3):
      a=-3
      for j in range(sharctrip):
          a=a+3
          reordermap.append(nrexci+i+a)
  
  for i in range(sharcnm):
      for j in range(sharcnm):
         socm_real[i][j]=SO_real_part_square[reordermap[i]][reordermap[j]]
         socm_imag[i][j]=SO_imag_part_square[reordermap[i]][reordermap[j]]

  for i in range(sharcnm):
      for j in range(sharcnm):
         socm[i][j]=complex(socm_real[i][j],socm_imag[i][j])
   
  
  QMout['h']=socm
  file.close()
  file1.close()

  return QMout

## ======================================================================= #
def run_smat(QMin):

   workdir = QMin['scratchdir']+'/OVERLAP'
   savedir = QMin['savedir']
#   shutil.copy(savedir+'/ADF.t21',workdir+'/ADF.t21')
#   shutil.copy(savedir+'/ADF.t21.old',workdir+'/ADF_old.t21')
   string = '$ADFBIN/adf -n %i <ADF_AO_overlap.run > ADF_AO_overlap.out'%(QMin['ncpu'])
   runerror=runProgram(string,workdir)
   if runerror == 0:
      os.chdir(workdir)
      shutil.move(workdir+'/TAPE15',workdir+'/ADF.t15')

   #copies the logfile if crashed
   if runerror!=0:
      print 'ADF calculation crashed! Error code = %i'%(runerror)
      s=QMin['savedir']+'/ADF-debug/'
      s1=s
      while os.path.exists(s1):
        i+=1
        s1=s+'%i/' % (i)
      if PRINT or DEBUG:
        print '=> Saving all text files from WORK directory to %s\n' % (s1)
      os.mkdir(s1)

      dirs=[
          os.path.join(QMin['scratchdir'],'OVERLAP')]
      maxsize=3*1024**2
      for d in dirs:
        ls=os.listdir(d)
        if DEBUG:
          print d
          print ls
        for i in ls:
          f=os.path.join(d,i)
          if os.stat(f)[6]<=maxsize and not os.path.isdir(f):
            try:
              shutil.copy(f,s1)
            except OSError:
              pass
            if PRINT or DEBUG:
              print i
      sys.exit(57)


## ======================================================================= #
def get_mocoef(QMin):
   
   path = QMin['scratchdir']+'/OVERLAP'
   os.chdir(path)
   shutil.copy(QMin['scratchdir']+'/ADF/ADF.t21',path)
   filename = 'ADF.t21'
   import kf
   file1 = kf.kffile(filename)
   
   Nrexci = int(file1.read('All excitations','nr excitations'))
   
   line_num=-1
   NAO = file1.read('Basis','naos')
   NMO = 0
   MOcoef=[]
   occuporb=[]
   if QMin['unr']=='no':
      NMO = file1.read('A','nmo_A')
      MOcoef_a = file1.read('A','Eigen-Bas_A')
      MOcoef=MOcoef_a.tolist()
      occuporb_a = file1.read('A','froc_A')
      occuporb=occuporb_a.tolist()
   npart_a = file1.read("A","npart")
   npart = npart_a.tolist()
   nspin=1 
   if QMin['unr']=='yes':
      nspin=2
      ncore=int(QMin['frozcore'])
      NMO_A = file1.read('A','nmo_A')
      MOcoef_a = file1.read('A','Eigen-Bas_A')
      MOcoef_A = MOcoef_a.tolist()
      occuporb_a = file1.read('A','froc_A')
      occuporb_A=occuporb_a.tolist()
      NMO_B = file1.read('A','nmo_B')
      MOcoef_b = file1.read('A','Eigen-Bas_B')
      MOcoef_B = MOcoef_b.tolist()
      occuporb_b = file1.read('A','froc_B')
      occuporb_B=occuporb_b.tolist()
      #NMO=NMO+NMO_B
      NMO=NMO_A+NMO_B
#      NMO=NMO_B[0:ncore]
#      NMO=NMO+NMO_A+NMO_B[ncore:]
#      MOcoef=MOcoef_B[0:ncore]
#      MOcoef.extend(MOcoef_A)
#      MOcoef.extend(MOcoef_B[ncore:])
      MOcoef=MOcoef_A
      MOcoef.extend(MOcoef_B)   
      occuporb=occuporb_B[0:ncore]
      occuporb.extend(occuporb_A)
      occuporb.extend(occuporb_B[ncore:])

#      MOcoef.extend(MOcoef_B)
#      occuporb.extend(occuporb_B)
#      npart.extend(npart)

   outfile = open('mocoef.b','w')
   init_MOcoef_mat = []
   if QMin['unr']=='yes':
      outfile.write('2mocoef\nheader\n 1\nMO-coefficients from ADF\n 1\n %i   %i \n a \nmocoef\n(*) \n' %(int(NAO),int(NMO)))
   else:
      outfile.write('2mocoef\nheader\n 1\nMO-coefficients from ADF\n 1\n %i   %i \n a \nmocoef\n(*) \n' %(int(NAO),int(NMO)))

   for n in range(0,int(NMO)):
       MOcoef_MO = []
       for m in range(0,int(NAO)):
           l = (n*NAO)+m
           MOcoef_MO.append(float(MOcoef[l]))
       init_MOcoef_mat.append(MOcoef_MO)
   
   MOcoef_mat = []
   for n in range(0,int(NMO)):
       MOcoef_MO = []
       for m in range(0,int(NAO)):
           z = npart.index(m+1)
           MOcoef_MO.append(init_MOcoef_mat[n][z])
       MOcoef_mat.append(MOcoef_MO)

   if QMin['unr']=='yes': 
      NMOA=int(NMO)/2
      MOcoef_mat_new=MOcoef_mat[NMOA:(NMOA+ncore)]
      MOcoef_mat_new.extend(MOcoef_mat[:NMOA])
      MOcoef_mat_new.extend(MOcoef_mat[(NMOA+ncore):])
      MOcoeg_mat=MOcoef_mat_new
   
   for n in range(0,int(NMO)):
       MOcoef_new_MO = []
       if NAO >= 3:
          if NAO%3==0:
             num_loops = int(NAO)/3
             a=-3
             for m in range(0,int(num_loops)):
                 a=a+3
                 outfile.write('  %6.12e  %6.12e  %6.12e \n'%(float(MOcoef_mat[n][0+a]),float(MOcoef_mat[n][1+a]),float(MOcoef_mat[n][2+a])))
          else:
             num_loops = int(NAO)/3
             a=-3
             for m in range(0,int(num_loops)):
                 a=a+3
                 outfile.write('  %6.12e  %6.12e  %6.12e \n'%(float(MOcoef_mat[n][0+a]),float(MOcoef_mat[n][1+a]),float(MOcoef_mat[n][2+a])))
             MOcoef_new_MO = MOcoef_mat[n][a+3:]
             for l in range(0,len(MOcoef_new_MO)):
                 outfile.write('  %6.12e  ' % (float(MOcoef_new_MO[l])))
             outfile.write('\n')
       elif NAO < 3:
          for l in range(0,len(MOcoef_mat[n])):
              outfile.write('  %6.12e  ' % (float(MOcoef_new_MO[l])))
          outfile.write('\n')
   
   outfile.write('orbocc \n(*) \n')
   occuporb_new = []
   if NMO >= 3:
      if NMO%3==0:
         num_loop = int(NMO)/3
         a=-3
         for m in range(0,int(num_loop)):
             a=a+3
             outfile.write('  %6.12e  %6.12e  %6.12e \n'%(float(occuporb[0+a]),float(occuporb[1+a]),float(occuporb[2+a])))
      else:
         num_loop = int(NMO)/3
         a=-3
         for m in range(0,int(num_loop)):
             a=a+3
             outfile.write('  %6.12e  %6.12e  %6.12e \n'%(float(occuporb[0+a]),float(occuporb[1+a]),float(occuporb[2+a])))
         occuporb_new=occuporb[a+3:]
         for l in range(0,len(occuporb_new)):
             outfile.write('  %6.12e  ' % (float(occuporb_new[l])))
         outfile.write('\n')
   elif NMO < 3:
      for l in range(0,len(occuporb)):
          outfile.write('  %6.12e  ' % (float(occuporb[l])))
      outfile.write('\n')

   file1.close()
   outfile.close()
#   shutil.copyfile(QMin['scratchdir']+'/OVERLAP/mocoef', QMin['savedir']+'/mocoef')  

## ======================================================================= #
def get_cicoef(QMin):

   path = QMin['scratchdir']+'/OVERLAP'
   os.chdir(path)
   filename ='ADF.t21'
   import kf
   file1 = kf.kffile(filename)

   Lhybrid = file1.read('General','lhybrid')
   Nrexci = int(file1.read('All excitations','nr excitations'))

   line_num=-1
   NAO = file1.read('Basis','naos')
   NMO = file1.read('A','nmo_A')
   ncore=QMin['frozcore']

   excita = int(Nrexci)+1
   Nrelec = file1.read('General','electrons')
   Dimension = 0
   Nocc = 0
   Nvirt=0
   Nocc_A = 0
   Nocc_B = 0
   Nvirt_A=0
   Nvirt_B=0
   if QMin['unr']=='no':
      Nocc = int(Nrelec)/2
      Nvirt = int(NMO)-Nocc
      Dimension = Nocc*Nvirt
   if QMin['unr']=='yes':
      Nocc_A_a = file1.read('A','froc_A')
      Nocc_A_b = Nocc_A_a.tolist()
      Nocc_B_a = file1.read('A','froc_B')
      Nocc_B_b = Nocc_B_a.tolist()
      Nocc_A=0
      Nocc_B=0
      for i in range(0,int(NMO)):
          Nocc_A=Nocc_A+Nocc_A_b[i]
          Nocc_B=Nocc_B+Nocc_B_b[i]
      Nvirt_A=int(NMO)-Nocc_A
      Nvirt_B=int(NMO)-Nocc_B
      Dimension= (Nocc_A*Nvirt_B)*2
      Dimension2=(Nocc_A*Nvirt_A)+(Nocc_B*Nvirt_B)
      NMO_B = file1.read('A','nmo_B')
      NMO=NMO+NMO_B

   threshold = float(QMin['wfthres'])
   if QMin['unr']=='yes':
      Mults= [1]
   else:
      Mults=[1]
      if len(QMin['states'])>=3 and QMin['states'][2]!=0:
         Mults = [1,3]
   for Mult in Mults:
      CIcoef = []
      CIthresh = []
#      CI_thresh_sorted = []
#      CI_thresh_sorted_A = []
#      CI_thresh_sorted_B = []
      exci_info=[]
      for exci in range(1,int(excita)):
         eigen = []
#         eigen_X = []
         eigen_other = []
         eigen_left = []
         eigen_right = []
         if Lhybrid == True and QMin['tda']==False:
            if Mult == 1:
               eigen_right = file1.read('Excitations SS A','eigenvector '+str(exci))
               eigen_left = file1.read('Excitations SS A','left eigenvector '+str(exci))
            else:
               eigen_right = file1.read('Excitations ST A','eigenvector '+str(exci))
               eigen_left = file1.read('Excitations ST A','left eigenvector '+str(exci))
            for a in range(0,int(Dimension)):
               eig_X_1 = (float(eigen_right[a])+float(eigen_left[a]))/(2.0)
               eig_X=eig_X_1**2
               eigen_other.append(eig_X)
               if QMin['unr']=='yes':
                   eig = (float(eigen_right[a])+float(eigen_left[a]))/2.0
               else:
                   eig = (float(eigen_right[a])+float(eigen_left[a]))/(2.0*float(math.sqrt(2)))
               eigen.append(eig)
         else:
            if Mult == 1:
               eigen_right = file1.read('Excitations SS A','eigenvector '+str(exci))
            else:
               eigen_right = file1.read('Excitations ST A','eigenvector '+str(exci))
            for a in range(0,int(Dimension)):
               eig_x = (float(eigen_right[a])**2)
               eig=0
               if QMin['unr']=='yes':
                   eig = float(eigen_right[a])
               else:
                   eig = float(eigen_right[a])/float(math.sqrt(2))
#               eigen_X.append(eig_x)
               eigen_other.append(eig_x)
               eigen.append(eig)
         CIcoef.append(eigen)
         CIthresh.append(eigen_other)
         exci_info_a=[]
         if QMin['unr']=='yes':
            Dimension_A=Dimension/2
            eigen_X_A=eigen_X[0:int(Dimension_A)]
            eigen_X_B=eigen_X[int(Dimension_A):int(Dimension)]
            eigen_X_A.sort(reverse=True)
            eigen_X_B.sort(reverse=True)
            CI_thresh_sorted_A.append(eigen_X_A)
            CI_thresh_sorted_B.append(eigen_X_B)
            CI_thresh_sorted.append(eigen_X)
            CI_thresh_sorted.append(eigen_X)
            for nspin in range(2):
                Nocc_curr=0
                if nspin==0:
                   Nocc_curr=int(Nocc_A)
                   Nvirt_curr=int(Nvirt_A)
                else:
                   Nocc_curr=int(Nocc_B)
                   Nvirt_curr=int(Nvirt_B)
                for a in range(Nocc_curr):
                    for b in range(Nvirt_curr):
                        c=b+(a*int(Nvirt_curr))+(int(nspin)*int(Dimension_A))
                        d=b+Nocc_curr
                        exci_tuple=(exci, a, b, CIcoef[exci-1][c], CIthresh[exci-1][c], nspin)
                        exci_info_a.append(exci_tuple)
            exci_info_a.sort(key=lambda x: x[4],reverse=True)    
            exci_info.append(exci_info_a)
         else:
            for a in range(Nocc):
                for b in range(Nvirt):
                    c=b+(a*Nvirt)
                    d=b+Nocc
                    exci_tuple=(exci, a, b, CIcoef[exci-1][c], CIthresh[exci-1][c])
                    exci_info_a.append(exci_tuple)
            exci_info_a.sort(key=lambda x: x[4],reverse=True)
            exci_info.append(exci_info_a)
#            eigen_X.sort(reverse=True)
#            CI_thresh_sorted.append(eigen_X)

#      new_CI_thresh = []
#      new_CI_thresh_A = []
#      new_CI_thresh_B = []
      new_exci_info=[]
      length=len(exci_info[0])
      for n in range(0,int(Nrexci)):
         thresh= 0
#         thresh_alt=0
         a=0
         while thresh < float(threshold):
                if a == length:
                   break
#                thresh = thresh+float(CI_thresh_sorted[n][a])
                thresh=thresh+float(exci_info[n][a][4])
                a=a+1
         new_exci_info_a=exci_info[n][:a]
         new_exci_info.append(new_exci_info_a)
#         New_CIvect = CI_thresh_sorted[n][:a+1]
#         new_CI_thresh.append(New_CIvect)


      Nrlines=0
      output = []
      config=[]
      if QMin['unr']=='yes':
         GS_config = ncore*'b'+int(Nocc_A)*'a'+int(Nvirt_A)*'e'+(int(Nocc_B)-ncore)*'b'+int(Nvirt_B)*'e'
         for n in range(0,int(Nrexci)):
             Config_string=''
             nconfig=len(new_exci_info[n])
             for a in range(nconfig):
                 numconfig=len(config)
                 nspin=new_exci_info[n][a][5]
                 exci_number=new_exci_info[n][a][0]
                 inorb=new_exci_info[n][a][1]
                 if inorb <ncore:
                    continue
                 fiorb=new_exci_info[n][a][2]
                 posnumber=exci_number+4
                 if nspin==0:
                    Config_string = ncore*'b'+inorb*'a'+'e'+(int(Nocc_A)-inorb-1)*'a'+fiorb*'e'+'a'+(int(Nvirt_A)-fiorb-1)*'e'+(int(Nocc_B)-ncore)*'b'+int(Nvirt_B)*'e'
                 if nspin==1:
                    Config_string = ncore*'b'+int(Nocc_A)*'a'+int(Nvirt_A)*'e'+(inorb-ncore)*'b'+'e'+(int(Nocc_B)-inorb-1)*'b'+fiorb*'e'+'b'+(int(Nvirt_B)-fiorb-1)*'e'
                 curr_config=[inorb,fiorb,nspin,Config_string]
                 for z in range(Nrexci+1):
                     curr_config.append(0.0)
                 if numconfig==0:
                     curr_config[posnumber]=float(new_exci_info[n][0][3])
                     config.append(curr_config)
                 else:
                     config_exists=False
                     config_number=0
                     for b in range(numconfig):
                         if nspin==config[b][2]:
                            if inorb==config[b][0]:
                               if fiorb==config[b][1]:
                                  config_exists=True
                                  config_number=b
                     if config_exists==True:
                        config[config_number][posnumber]=new_exci_info[n][a][3]
                     else:
                        curr_config[posnumber]=new_exci_info[n][a][3]
                        config.append(curr_config)

         Nrlines=len(config)
         outfilename = 'cicoef.b'
         outfile2=open(outfilename,'w')
         outfile2.write('%i %i %i \n' % (int(Nrexci+1), int(NMO), int(Nrlines+1)))
         outfile2.write(GS_config+'  1.000000000000  '+int(Nrexci)*'  0.000000000000  '+'\n')

         for n in range(Nrlines):
             outputstring=str(config[n][3])+'  %6.12f  ' %(float(config[n][4]))
             for a in range(Nrexci):
                 outputstring+='  %6.12f  ' % (float(config[n][a+5]))
             outputstring+='\n'
             outfile2.write('%s' %(outputstring))

      else:
         GS_config = Nocc*'d'+Nvirt*'e'
         for n in range(0,int(Nrexci)):
             Config_string=''
             Config_string2=''
             nconfig=len(new_exci_info[n])
             for a in range(nconfig):
                 numconfig=len(config)
                 exci_number=new_exci_info[n][a][0]
                 inorb=new_exci_info[n][a][1]
                 if inorb <ncore:
                    continue
                 fiorb=new_exci_info[n][a][2]
                 posnumber=exci_number+3
                 if Mult == 1:
                    Config_string = inorb*'d'+'a'+(int(Nocc)-inorb-1)*'d'+fiorb*'e'+'b'+(int(Nvirt)-fiorb-1)*'e'
                    Config_string2 = inorb*'d'+'b'+(int(Nocc)-inorb-1)*'d'+fiorb*'e'+'a'+(int(Nvirt)-fiorb-1)*'e'
                 elif Mult == 3:
                    Config_string = inorb*'d'+'a'+(int(Nocc)-inorb-1)*'d'+fiorb*'e'+'a'+(int(Nvirt)-fiorb-1)*'e'
                 curr_config=[inorb,fiorb,Config_string]
                 if Mult ==1:
                    curr_config2=[inorb,fiorb,Config_string2]
                    for z in range(Nrexci+1):
                        curr_config.append(0.0)
                        if Mult == 1:
                           curr_config2.append(0.0)
                 elif Mult ==3:
                    for z in range(Nrexci):
                        curr_config.append(0.0)
                 if numconfig==0:
                     if Mult==1:
                        curr_config[posnumber]=float(new_exci_info[n][0][3])
                        curr_config2[posnumber]=-1.0*float(new_exci_info[n][0][3])
                        config.append(curr_config)
                        config.append(curr_config2)
                     elif Mult==3:
                        curr_config[posnumber-1]=math.sqrt(2)*float(new_exci_info[n][0][3])
                        config.append(curr_config)
                 else:
                     config_exists=False
                     config_number=0
                     for b in range(numconfig):
                         if inorb==config[b][0]:
                            if fiorb==config[b][1]:
                               if Config_string==config[b][2]:
                                  config_exists=True
                                  config_number=b
                     if config_exists==True:
                        if Mult == 1:
                           config[config_number][posnumber]=new_exci_info[n][a][3]
                           config[config_number+1][posnumber]=-1.0*new_exci_info[n][a][3]
                        elif Mult ==3:
                           config[config_number][posnumber-1]=math.sqrt(2)*new_exci_info[n][a][3]
                     else:
                        if Mult==1:
                           curr_config[posnumber]=new_exci_info[n][a][3]
                           curr_config2[posnumber]=-1.0*new_exci_info[n][a][3]
                           config.append(curr_config)
                           config.append(curr_config2)
                        elif Mult == 3:
                           curr_config[posnumber-1]=math.sqrt(2)*new_exci_info[n][a][3]
                           config.append(curr_config)

         Nrlines=len(config)
         if Mult == 1:
            outfilename = 'cicoef_S.b'
         if Mult == 3: 
            outfilename = 'cicoef_T.b'
         outfile2=open(outfilename,'w')
         if Mult == 1:
            outfile2.write('%i %i %i \n' % (int(Nrexci+1), int(NMO), int(Nrlines+1)))
            outfile2.write(GS_config+'  1.000000000000  '+int(Nrexci)*'  0.000000000000  '+'\n')
         elif Mult == 3: 
            outfile2.write('%i %i %i \n' % (int(Nrexci), int(NMO), int(Nrlines)))

         for n in range(Nrlines):
             if Mult == 1: 
                outputstring=str(config[n][2])+'  %6.12f  ' %(float(config[n][3]))
             elif Mult ==3:
                outputstring=str(config[n][2])
             for a in range(Nrexci):
                 if Mult == 1:
                     outputstring+='  %6.12f  ' % (float(config[n][a+4]))
                 elif Mult ==3:
                     outputstring+='  %6.12f  ' % (float(config[n][a+3]))
             outputstring+='\n'
             outfile2.write('%s' %(outputstring))

#      else:
#         GS_config = Nocc*'d'+Nvirt*'e'
#         for i in range(0,int(Nocc)):
#             for a in range(0,int(Nvirt)):
#                 Config_string = ''
#                 Config_string2 = ''
#                 Config_string3 = ''
#                 Config_string4 = ''
#                 string = ''
#                 m=i*Nvirt
#                 if Mult == 1:
#                    Config_string = i*'d'+'a'+(int(Nocc)-i-1)*'d'+a*'e'+'b'+(int(Nvirt)-a-1)*'e'
#                    Config_string2 = i*'d'+'b'+(int(Nocc)-i-1)*'d'+a*'e'+'a'+(int(Nvirt)-a-1)*'e'
#                 if Mult == 3:
#                    Config_string = i*'d'+'a'+(int(Nocc)-i-1)*'d'+a*'e'+'a'+(int(Nvirt)-a-1)*'e'
#                 list = []
#                 for n in range(0,int(Nrexci)):
#                    p = CIthresh[n][m+a]
#                    for z in range(len(new_CI_thresh[n])):
#                       if p == float(new_CI_thresh[n][z]):
#                          list.append(n)
#                 if len(list) !=0 :
#                    Nrlines = Nrlines + 1
#                    if Mult == 1:
#                       string+=Config_string+'  0.000000000000  '
#                    else:
#                       string+=Config_string
#                    for n in range(0,int(Nrexci)):
#                       if n in list:
#                          if Mult == 3:
#                             CIcoefficient = 1.0 * math.sqrt(2)* float(CIcoef[n][m+a])
#                             string+='  %6.12f  ' % (float(CIcoefficient))
#                          else:
#                             string+='  %6.12f  ' % (float(CIcoef[n][m+a]))
#                       else:
#                          string+='  0.000000000000  '
#                    string+='\n'
#                    if Mult == 1:
#                       string+=Config_string2+'  0.000000000000  '
#                       for n in range(0,int(Nrexci)):
#                          if n in list:
#                             CIcoefficient = 0
#                             if Mult ==1 :
#                                CIcoefficient = -1.0 * float(CIcoef[n][m+a])
#                             if Mult == 3:
#                                CIcoefficient = 1.0 * float(CIcoef[n][m+a])
#                             string+='  %6.12f  ' %(float(CIcoefficient))
#                          else:
#                             string+='  0.000000000000  '
#                       string+='\n'
#                    output.append(string)
#         if Mult == 1:
#            Nrlines=Nrlines*2
#            outfilename = 'cicoef_S.b'
#         if Mult == 3:
#            Nrlines=Nrlines
#            outfilename = 'cicoef_T.b'
#         outfile2=open(outfilename,'w')
#         if Mult != 1:
#            outfile2.write('%i %i %i \n' % (int(Nrexci), int(NMO), int(Nrlines)))
#         else:
#            outfile2.write('%i %i %i \n' % (int(Nrexci+1), int(NMO), int(Nrlines+1)))
#            outfile2.write(GS_config+'  1.000000000000  '+int(Nrexci)*'  0.000000000000  '+'\n')
#   
#         for n in range(0,len(output)):
#             outfile2.write('%s' %(output[n]))
    
   outfile2.close()
   file1.close() 

#   shutil.copyfile(QMin['scratchdir']+'/OVERLAP/cicoef_S', QMin['savedir']+'/cicoef_S')
#   shutil.copyfile(QMin['scratchdir']+'/OVERLAP/cicoef_T', QMin['savedir']+'/cicoef_T')     

##### ======================================================================= #
###def get_smat(QMin):
    
   ###path = QMin['scratchdir']+'/OVERLAP'
   ###os.chdir(path)
   ###filename = 'ADF.t15'
   ###import kf
   ###file1 = kf.kffile(filename)
   ###NAO = file1.read('Basis','naos')
   ###Smat = file1.read('Matrices','Smat')
   ###Diag_SMAT=[]
   ###i=0
   ###j=[]
   ###k=0
   ###Smat_final = []
   ###for a in range(0,int(NAO)):
       ###i=i+1
       ###j.append(i)
       ###Smat_column=[]
       ###l=math.fsum(j)
       ###for b in range(int(k),int(l)):
          ###Smat_column.append(Smat[b])
       ###k=l
       ###Diag_SMAT.append(Smat_column)
   ###geom_NAO = int(NAO)/2
   ###for a in range(int(geom_NAO),int(NAO)):
       ###Smat_column=[]
       ###for b in range(0,int(geom_NAO)):
           ###Smat_column.append(Diag_SMAT[a][b])
       ###Smat_final.append(Smat_column)
   
   ###outfile = open(QMin['scratchdir']+'/OVERLAP/AO_overl.mixed','w')
   ###outfile.write(' %i %i \n' %(int(geom_NAO),int(geom_NAO)))

   ###for b in range(0,int(geom_NAO)):
       ###for a in range(0,int(geom_NAO)):
           ###outfile.write('  %6.12e  '%(float(Smat_final[a][b])))
           ###if a == int(geom_NAO)-1:
              ###outfile.write(' \n')#

   ###outfile.close()
   ###file1.close()
####   shutil.copy(path+'/AO_overl.mixed',QMin['scratchdir']+'/OVERLAP/AO_overl.mixed')


## ======================================================================= #
def get_smat(QMin):

    path = QMin['scratchdir']+'/OVERLAP'
    os.chdir(path)
    filename = 'ADF.t15'
    import kf
    file1 = kf.kffile(filename)
    NAO = file1.read('Basis','naos')
    Smat = file1.read('Matrices','Smat')
    file1.close()

    # Smat is lower triangular matrix, len is NAO*(NAO+1)/2
    ao_ovl=makermatrix(NAO,NAO)
    x=0
    y=0
    for el in Smat:
        ao_ovl[x][y]=el
        ao_ovl[y][x]=el
        x+=1
        if x>y:
            x=0
            y+=1
    # ao_ovl is now full matrix NAO*NAO
    # we want the lower left quarter, but transposed

    # write AO overlap matrix to savedir
    string='%i %i\n' % (NAO/2,NAO/2)
    for irow in range(NAO/2,NAO):
        for icol in range(0,NAO/2):
            string+='% .15e ' % (ao_ovl[icol][irow])          # note the exchanged indices => transposition
            string+='\n'
    filename=os.path.join(QMin['scratchdir'],'OVERLAP','AO_overl.mixed')
    writefile(filename,string)











## ======================================================================= #
def run_wfoverlap(QMin):

    workdir = QMin['scratchdir']+'/OVERLAP'
    os.chdir(workdir)
    os.environ['OMP_NUM_THREADS']=str(QMin['ncpu'])
    ncore = int(QMin['frozcore'])
    if QMin['unr']=='yes':
      shutil.copy(QMin['savedir']+'/cicoef.old', workdir+'/cicoef.a')
    else:   
      shutil.copy(QMin['savedir']+'/cicoef_S.old', workdir+'/cicoef_S.a')
      if len(QMin['states'])>=3:
         if QMin['states'][2]!=0: 
            shutil.copy(QMin['savedir']+'/cicoef_T.old', workdir+'/cicoef_T.a')
    shutil.copy(QMin['savedir']+'/mocoef.old', workdir+'/mocoef.a')
    if QMin['unr']=='yes':
      ncore=ncore*2
      outfile = open('wfovl.in','w')
      outfile.write('a_mo=mocoef.a \nb_mo=mocoef.b \na_det=cicoef.a \nb_det=cicoef.b\nmix_aoovl=AO_overl.mixed\nncore=%i'%(ncore))
      outfile.close()
      string='%s -f wfovl.in > wfovl.out  || exit $?'%(QMin['wfoverlap'])
      runerror=runProgram(string,workdir)
    else:
      outfile = open('wfovl_S.in','w')
      outfile.write('a_mo=mocoef.a \nb_mo=mocoef.b \na_det=cicoef_S.a \nb_det=cicoef_S.b\nmix_aoovl=AO_overl.mixed\nncore=%i'%(ncore))
      outfile.close()
      string='%s -f wfovl_S.in > wfovl_S.out  || exit $?'%(QMin['wfoverlap'])
      runerror=runProgram(string,workdir)
      if len(QMin['states'])>=3:
         if QMin['states'][2]!=0:
            outfile1 = open('wfovl_T.in','w')
            outfile1.write('a_mo=mocoef.a \nb_mo=mocoef.b \na_det=cicoef_T.a \nb_det=cicoef_T.b\nmix_aoovl=AO_overl.mixed\nncore=%i'%(ncore))
            outfile1.close()
            string2='%s -f wfovl_T.in > wfovl_T.out  || exit $?'%(QMin['wfoverlap']) 
            runerror=runProgram(string2,workdir)
    if runerror!=0:
      print 'wfoverlap call not successful!'
      sys.exit(58)

## ======================================================================= #
def get_wfoverlap(QMin,QMout):

   nmstates = int(QMin['nmstates'])
   QMout['overlap'] = makecmatrix(nmstates,nmstates)
   if QMin['unr']=='yes':
      out = readfile(QMin['scratchdir']+'/OVERLAP/wfovl.out')
   else:
      out = readfile(QMin['scratchdir']+'/OVERLAP/wfovl_S.out')
      if len(QMin['states'])>=3:
         if QMin['states'][2]!=0:
            out1 = readfile(QMin['scratchdir']+'/OVERLAP/wfovl_T.out')
   nstates = int(QMin['nstates'])
   if QMin['unr']=='yes':
     Multip=int(QMin['template']['unpelec'])+1
     for m in range(Multip):
         for i in range(nstates):
             for j in range(nstates):
                a=i+((m)*nstates)
                b=j+((m)*nstates)
                ilines = -1
                outfile=out
                while True:
                   ilines+=1
                   if ilines==len(outfile):
                       print 'Overlap of states %i - %i not found!' % (i+1,b+1)
                       sys.exit(59)
                   if containsstring('Overlap matrix <PsiA_i|PsiB_j>', outfile[ilines]):
                       break
                ilines+=i+2
                f=outfile[ilines].split()
                QMout['overlap'][a][b]=float(f[j+2])
   else:
      nSings = int(QMin['states'][0])
      if len(QMin['states'])>=3:
         if QMin['states'][2]!=0:
            nTrips = int(QMin['states'][2])
      for i in range(nstates):
         for j in range(nstates):
            ilines = -1
            if i <nSings and j <nSings:
               outfile=out
               while True:
                  ilines+=1
                  if ilines==len(outfile):
                      print 'Overlap of states %i - %i not found!' % (i+1,b+1)
                      sys.exit(59)
                  if containsstring('Overlap matrix <PsiA_i|PsiB_j>', outfile[ilines]):
                      break
               ilines+=i+2
               f=outfile[ilines].split()
               QMout['overlap'][i][j]=float(f[j+2])
            elif i >= nSings and j >= nSings:
               a=i-nSings
               b=j-nSings
               outfile = out1
               while True:
                  ilines+=1
                  if ilines==len(outfile):
                      print 'Overlap of states %i - %i not found!' % (a+1,b+1)
                      sys.exit(60)      
                  if containsstring('Overlap matrix <PsiA_i|PsiB_j>', outfile[ilines]):
                      break
               ilines+=a+2
               f=outfile[ilines].split()
               QMout['overlap'][i][j]=float(f[b+2])
               QMout['overlap'][i+nTrips][j+nTrips]=float(f[b+2])
               QMout['overlap'][i+(2*nTrips)][j+(2*nTrips)]=float(f[b+2])
            else:
               QMout['overlap'][i][j]=float(0.0)  
#   for i in range(nstates):
#     for j in range(nstates):
#       s1 = i+1
#       s2 = j+1
#       ilines=-1
#       outuse=out
#       if s1 and s2 > nSings:
#         outuse=out1
#         while True:
#           ilines+=1
#           if ilines==len(outuse):
#             print 'Overlap of states %i - %i not found!' % (s1,s2)
#             sys.exit(61)
#           if containsstring('Overlap matrix <PsiA_i|PsiB_j>', outuse[ilines]):
#             break
#         ilines+=1+s1
#         f=outuse[ilines].split()
#         print f
#         QMout['overlap'][i][j]=float(f[s2+1])
#       elif s1 or s2 > nSings:
#         QMout['overlap'][i][j]= 0.0
#       else:
#         outuse=out
#         while True:
#           ilines+=1
#           if ilines==len(outuse):
#             print 'Overlap of states %i - %i not found!' % (s1,s2)
#             sys.exit(62)
#           if containsstring('Overlap matrix <PsiA_i|PsiB_j>', outuse[ilines]):
#             break
#         ilines+=1+s1 
#         f=outuse[ilines].split()
#         print f
#         QMout['overlap'][i][j]=float(f[s2+1])

   return QMout

## ======================================================================= #
def CreateQMout(QMin,QMout):

  nstates = QMin['nstates']
  nmstates = QMin['nmstates']
  natom = QMin['natom']
  nrsings = QMin['states'][0]
  if len(QMin['states'])>=3:
     nrtrips = QMin['states'][2]
  if 'grad' in QMin:
     Grad = []
     if QMin['unr'] == 'no':
#         nrsings = QMin['states'][0]
#         nrtrips = QMin['states'][2]
         for i in range(nmstates):
             Grad.append(makecmatrix(3,natom))
         for i in range(nmstates):
             string=''
             if i <nrsings:
                string+='S'+str(i)
                if string in QMout['gradients']:
                   for j in range(natom):
                       for xyz in range(3):
                           Grad[i][j][xyz]=float(QMout['gradients'][string][j][xyz])
                else:
                    for j in range(natom):
                        for xyz in range(3):
                          Grad[i][j][xyz]=float(0.0)
             elif i >=nrsings and nrtrips !=0 :
                if (i-nrsings)%nrtrips == 0:
                   if 'T1' in QMout['gradients']:
                      for j in range(natom):
                          for xyz in range(3):
                              Grad[i][j][xyz]=float(QMout['gradients']['T1'][j][xyz])
                   else:
                      for j in range(natom):
                          for xyz in range(3):
                            Grad[i][j][xyz]=float(0.0)
                elif (i-nrsings)%nrtrips == 1:
                   for a in range(1,nrtrips):
                       if 'T'+str(a+1) in QMout['gradients']:
                          for j in range(natom):
                              for xyz in range(3):
                                  Grad[i+a-1][j][xyz]=float(QMout['gradients']['T'+str(a+1)][j][xyz])
                       else:
                          for j in range(natom):
                              for xyz in range(3):
                                Grad[i+a-1][j][xyz]=float(0.0)
                else:
                   continue 
     else: 
         Multip=int(QMin['template']['unpelec'])+1
         for i in range(nmstates):
             Grad.append(makecmatrix(3,natom))
         a=0
         for m in range(Multip):
             for i in range(nstates):
                 string=''
                 if QMin['states'][1]!=0:
                    string+=IToMult[Multip][0]+str(i)
                 else:
                    string+=IToMult[Multip][0]+str(i+1)
                 if string in QMout['gradients']:
                    a=(m*nstates)+i
                    for j in range(natom):
                        for xyz in range(3):
                            Grad[a][j][xyz]=float(QMout['gradients'][string][j][xyz])
                 else:
                    a=(m*nstates)+i
                    for j in range(natom):
                        for xyz in range(3):
                          Grad[a][j][xyz]=float(0.0)
     QMout['grad']=Grad

  if 'dm' in QMin:
      dm = []
      for xyz in range(0,3):
          dm.append(makecmatrix(nmstates,nmstates))
          if QMin['unr']=='no':
             for i in range(nmstates):
                 for j in range(nmstates):
                     string = ''
                     if i == 0 and j<nrsings:
                        if j > 0:
                           string='S'+str(j)+'_GS'
                        else:
                           string = 'GS'
                        dm[xyz][i][j] = complex(QMout['dipolemoments'][string][xyz])
                     elif j == 0  and i<nrsings:
                        if i > 0:
                           string='S'+str(i)+'_GS'
                        else:
                           string = 'GS'
                        dm[xyz][i][j] = complex(QMout['dipolemoments'][string][xyz])
                     elif i == j and len(QMout['dipolemoments'])>nrsings:
                        if i < nrsings:
                           if 'S'+str(i) in QMout['dipolemoments']:
                              string = 'S'+str(i)
                              dm[xyz][i][j] = complex(QMout['dipolemoments'][string][xyz])
                        else:
                           if (i-nrsings)%nrtrips == 0:
                              if 'T1' in QMout['dipolemoments']:
                                 dm[xyz][i][i]=complex(QMout['dipolemoments']['T1'][xyz])
                           elif (i-nrsings)%nrtrips == 1:
                              for a in range(1,nrtrips):
                                 if 'T'+str(a+1) in QMout['dipolemoments']:
                                    dm[xyz][i+a-1][i+a-1]=complex(QMout['dipolemoments']['T'+str(a+1)][xyz])
                                 else:
                                    dm[xyz][i+a-1][i+a-1]=complex(0.0)
                           else:
                              continue
                     else:
                        dm[xyz][i][j] = complex(0.0)
          else:
             Multip=int(QMin['template']['unpelec'])+1
             for m in range(Multip):
                 for i in range(nstates):
                     for j in range(nstates):
                         string=''
                         if i == 0:
                            if j>0:
                               if QMin['states'][1]!=0:
                                  string+=IToMult[Multip][0]+str(j)+'_GS'
                               else:
                                  a=j+1
                                  string+=IToMult[Multip][0]+str(a)+'_GS'
                            else:
                               string='GS'
                            dm[xyz][i+(m*nstates)][j+(m*nstates)] = complex(QMout['dipolemoments'][string][xyz]) 
                         elif j == 0 :
                            if i > 0:
                               if QMin['states'][1]!=0:
                                  string+=IToMult[Multip][0]+str(i)+'_GS'
                               else:
                                  a=i+1
                                  string+=IToMult[Multip][0]+str(a)+'_GS'
                            else:
                               string='GS'
                            dm[xyz][i+(m*nstates)][j+(m*nstates)] = complex(QMout['dipolemoments'][string][xyz])     
                         elif i == j:
                            if QMin['states'][1]!=0:
                               string+=IToMult[Multip][0]+str(i)
                            else:
                               a=j+1
                               string+=IToMult[Multip][0]+str(a)
                            if string in QMout['dipolemoments']:
                               dm[xyz][i+(m*nstates)][i+(m*nstates)] = complex(QMout['dipolemoments'][string][xyz])
                         else:
                            dm[xyz][i][j] = complex(0.0)
      QMout['dm']=dm
              
  return QMout

# ======================================================================= #

def get_zeroQMout(QMin):
    nmstates=QMin['nmstates']
    natom=QMin['natom']
    QMout={}
    if 'h' in QMin or 'soc' in QMin:
        QMout['h']=[ [ complex(0.0) for i in range(nmstates) ] for j in range(nmstates) ]
    if 'dm' in QMin:
        QMout['dm']=[ [ [ complex(0.0) for i in range(nmstates) ] for j in range(nmstates) ] for xyz in range(3) ]
    if 'overlap' in QMin:
        QMout['overlap']=[ [ complex(0.0) for i in range(nmstates) ] for j in range(nmstates) ]
    if 'grad' in QMin:
        QMout['grad']=[ [ [0.,0.,0.] for i in range(natom) ] for j in range(nmstates) ]
    return QMout

# ======================================================================= #
def cleanupSCRATCH(SCRATCHDIR):
    ''''''
    if PRINT:
        print '===> Removing directory %s\n' % (SCRATCHDIR)
    try:
        if True:
            shutil.rmtree(SCRATCHDIR)
        else:
            print 'not removing anything. SCRATCHDIR is %s' % SCRATCHDIR
    except OSError:
        print 'Could not remove directory %s' % (SCRATCHDIR)


# ========================== Main Code =============================== #
def main():

    # Retrieve PRINT and DEBUG
    try:
        envPRINT=os.getenv('SH2ADF_PRINT')
        if envPRINT and envPRINT.lower()=='false':
            global PRINT
            PRINT=False
        envDEBUG=os.getenv('SH2ADF_DEBUG')
        if envDEBUG and envDEBUG.lower()=='true':
            global DEBUG
            DEBUG=True
    except ValueError:
        print 'PRINT or DEBUG environment variables do not evaluate to numerical values!'
        sys.exit(63)

    # Process Command line arguments
    if len(sys.argv)!=2:
        print 'Usage:\n./SHARC_ADF.py <QMin>\n'
        print 'version:',version
        print 'date:',versiondate
        print 'changelog:\n',changelogstring
        sys.exit(64)
    QMinfilename=sys.argv[1]

    # Print header
    printheader()

    # Read QMinfile
    QMin=readQMin(QMinfilename)

    sys.path.append(QMin['ADFHOME']+'/scripting')
#    import kf

    #Process Tasks
    Tasks=gettasks(QMin)
    if DEBUG or PRINT:
       printtasks(Tasks)

    # get output
    QMout=runeverything(Tasks,QMin)
    
    printQMout(QMin,QMout)

    # Measure time
    runtime=measuretime()
    QMout['runtime']=runtime

    # Write QMout
    writeQMout(QMin,QMout,QMinfilename)

#    # Remove Scratchfiles from SCRATCHDIR
#    if not DEBUG:
#        cleanupSCRATCH(QMin['scratchdir'])
    if PRINT or DEBUG:
        print '#================ END ================#'

if __name__ == '__main__':
    main()






# kate: indent-width 4
