#!/usr/bin/env python2

#    ====================================================================
#||                                                                                                                                        ||
#||                                                         General Remarks                                                ||
#||                                                                                                                                        ||
#    ====================================================================
#
# This script uses several different specification for the electronic states under consideration.
# Generally, the input specs are like "3 Singlets, 0 Doublets and 3 Triplets"
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
# A new task keyword in QMin has to be added to:
#             - readQMin (for consistency check)
#             - gettasks (planning the MOLCAS calculation)
#             - print QMin (optional)
#
# A new task in the Tasks array needs changes in:
#             - gettasks 
#             - writeMOLCASinput 
#             - redotasks
#             - printtasks

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

version='1.0'
versiondate=datetime.date(2014,10,8)


changelogstring='''
06.02.2013:
- changed the records for cpmcscf to 5xxx.9

07.02.2013:
- added angular keyword (angular momentum evaulation and extraction)
- extraction of best obtained accuracy in cpmcscf
- nogprint,orbitals,civectors added in MOLCAS input

08.02.2013:
- added removal of SCRATCHDIR after job finished successfully
- added expansion of environment variables and ~ in paths

13.03.2013:
- added facility for selective analytical NACs
- added input for nac ana select
- added environment variables for PRINT and DEBUG

08.05.2013:
- added CI:pspace task

22.05.2013:
- Smat is written transposed now (to be compatible with Fortran)

06.06.2013:
- MCSCF convergence thresholds increased by default, to help cpmcscf convergence

11.10.2013:
- changed keyword "nac" to "nacdr", "nacdt" and "overlap"
=>NOT COMPATIBLE WITH OLDER VERSIONS OF SHARC!

09.01.2014:
- Changed script to work with MOLCAS instead of MOLPRO (still work in progress)
Functions with a #-symbol have been changed.
readQMin -> inputfile to dictionary with keywords about what should be calculated, number of states, ...
gettasks -> converts dictionary of QMin to list of tasks with parameters
cycleMOLPRO -> cycle through task list until it is empty
    # writeMOLPROinput -> writes MOLPRO input for the given tasks -> has to be rewritten for MOLCAS usage
    # runMOLPRO -> starts the MOLPRO job
    # redotasks -> scans MOLPRO output for error messages. checks if tasks performed succesfully. returns a new tasklist including all crashed tasks
catMOLPROoutput -> concatenates all MOLPRO output files
getQMout
    # getcienergy -> casscf energy of a state specified by mult and statenumber
    # getsocme -> get single SOC matrix element
      getcidm -> TODO later
    # getgrad -> get gradient for certain atom and certain mult,state
    # getsmate -> returns the overlap matrix with possible sign changes of columns if diagonal elements are negative
    # getmrcioverlap -> right now only returns the unity matrix
printQMout
cleanupSCRATCH
writeQMout

07.07.2014:
- Gradients can be setup with MOLCAS in parallel fashion, using 1 core per gradient.
- QM/MM support added (using MOLCAS and TINKER).

03.10.2014:
- Changed title lines
- readQMin was harmonized with the other interfaces:
  * reads MOLCAS.template as before
  * reads SH2CAS.inp (keywords "molcas", "scratchdir", "memory")
  * Project is the comment on the second line of QM.in, stripped of whitespace and prepended with "Comment_"
  * SHQM directory is the pwd (the directory where the interface is started
  * gradaccudefault and gradaccumax as in MOLPRO interface
- for SS-CASSCF (in any given multiplicity) no MCLR is executed
- changed getgrad to also find SS-CASSCF gradients
- in writeMOLCASinput, section task[0]=="ddrdiab" fixed a bug (mixup of variables mult and i)

08.10.2014:     1.0
- official release version, no changes to 0.3
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

# conversion factors
B2Ang = 1.889725989 # bohr (a.u.) to angstrom

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
def itmult(states):
    '''Takes an array of the number of states in each multiplicity and generates an iterator over all multiplicities with non-zero states.

    Example:
    [3,0,3] yields two iterations with
    1
    3

    Arguments:
    1 list of integers: States specification

    Returns:
    1 integer: multiplicity'''

    for i in range(len(states)):
        if states[i]<1:
            continue
        yield i+1
    return

# ======================================================================= #
def itnstates(states):
    '''Takes an array of the number of states in each multiplicity and generates an iterator over all states specified. Different values of MS for each state are not taken into account.

    Example:
    [3,0,3] yields six iterations with
    1,1
    1,2
    1,3
    3,1
    3,2
    3,3

    Arguments:
    1 list of integers: States specification

    Returns:
    1 integer: multiplicity
    2 integer: state'''

    for i in range(len(states)):
        if states[i]<1:
            continue
        for j in range(states[i]):
            yield i+1,j+1
    return

# ======================================================================= #
def itnmstates(states):
    '''Takes an array of the number of states in each multiplicity and generates an iterator over all states specified. Iterates also over all MS values of all states.

    Example:
    [3,0,3] yields 12 iterations with
    1,1,0
    1,2,0
    1,3,0
    3,1,-1
    3,2,-1
    3,3,-1
    3,1,0
    3,2,0
    3,3,0
    3,1,1
    3,2,1
    3,3,1

    Arguments:
    1 list of integers: States specification

    Returns:
    1 integer: multiplicity
    2 integer: state
    3 integer: MS value'''

    for i in range(len(states)):
        if states[i]<1:
            continue
        for k in range(i+1):
            for j in range(states[i]):
                yield i+1,j+1,k-i/2.
    return

# ======================================================================= #
def ittwostates(states):
    '''Takes an array of the number of states in each multiplicity and generates an iterator over all pairs of states (s1/=s2 and s1<s2), which have the same multiplicity. Different values of MS for each state are not taken into account.

    Example:
    [3,0,3] yields six iterations with
    1 1 2
    1 1 3
    1 2 3
    3 1 2
    3 1 3
    3 2 3

    Arguments:
    1 list of integers: States specification

    Returns:
    1 integer: multiplicity
    2 integer: state 1
    3 integer: state 2'''

    for i in range(len(states)):
        if states[i]<2:
            continue
        for j1 in range(states[i]):
            for j2 in range(j1+1,states[i]):
                yield i+1,j1+1,j2+1
    return

# ======================================================================= #
def ittwostatesfull(states):
    '''Takes an array of the number of states in each multiplicity and generates an iterator over all pairs of states (all combinations), which have the same multiplicity. Different values of MS for each state are not taken into account.

    Example:
    [3,0,3] yields 18 iterations with
    1 1 1
    1 1 2
    1 1 3
    1 2 1
    1 2 2
    1 2 3
    1 3 1
    1 3 2
    1 3 3
    3 1 1
    3 1 2
    3 1 3
    3 2 1
    3 2 2
    3 2 3
    3 3 1
    3 3 2
    3 3 3

    Arguments:
    1 list of integers: States specification

    Returns:
    1 integer: multiplicity
    2 integer: state 1
    3 integer: state 2'''

    for i in itmult(states):
        for j in itnstates([states[i-1]]):
            for k in itnstates([states[i-1]]):
                yield i,j[1],k[1]
    return

# ======================================================================= #
def IstateToMultState(i,states):
    '''Takes a state index in nmstates counting scheme and converts it into (mult,state).

    Arguments:
    1 integer: state index
    2 list of integers: states specification

    Returns:
    1 integer: mult
    2 integer: state'''

    for mult,state,ms in itnmstates(states):
        i-=1
        if i==0:
            return mult,state
    print 'state %i is not in states:',states
    sys.exit(12)

def IToMultStateMS(i,states):
    '''Takes a state index in nmstates counting scheme and converts it into (mult,state,ms).

    Arguments:
    1 integer: state index
    2 list of integers: states specification

    Returns:
    1 integer: mult
    2 integer: state
    3 integer: ms'''

    for mult,state,ms in itnmstates(states):
        i -= 1
        if i==0:
            return mult,state,ms
    print 'state %i is not in states:',states
    sys.exit(13)

# ======================================================================= #
def MultStateToIstate(mult,state,states):
    '''Takes a tuple (mult,state) and returns all indices i in nmstates scheme, which correspond to this mult and state.

    Arguments:
    1 integer: mult
    2 integer: state
    3 list of integers: states specification

    Returns:
    1 integer: state index in nmstates scheme'''

    if mult-1>len(states) or state>states[mult-1]:
        print 'No state %i, mult %i in states:' % (state,mult),states
        sys.exit(14)
    i=1
    for imult,istate,ms in itnmstates(states):
        if imult==mult and istate==state:
            yield i
        i+=1
    return

# ======================================================================= #
def MultStateToIstateJstate(mult,state1,state2,states):
    '''Takes (mult,state1,state2) and returns all index tuples (i,j) in nmstates scheme, which correspond to this mult and pair of states. Only returns combinations, where both states have the same MS value.

    Arguments:
    1 integer: mult
    2 integer: state1
    3 integer: state2
    4 list of integers: states specification

    Returns:
    1 integer: state1 index in nmstates scheme
    2 integer: state2 index in nmstates scheme'''

    if mult-1>len(states) or state1>states[mult-1] or state2>states[mult-1]:
        print 'No states %i, %i mult %i in states:' % (state1,state2,mult),states
        sys.exit(15)
    i=1
    k=-1
    for imult,istate,ms in itnmstates(states):
        if imult==mult and istate==state1:
            j=1
            for jmult,jstate,ms2 in itnmstates(states):
                if jmult==mult and jstate==state2:
                    k+=1
                    if k%(mult+1)==0:
                        yield i,j
                j+=1
        i+=1
    return

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

# ======================================================================= #        OK
def linekeyword(line):
    '''Takes a string, makes it lowercase and returns the first field.

    Arguments:
    1 string

    Returns:
    1 string'''

    return line.lower().split()[0]

# ======================================================================= #         OK
def lineargs(line):
    '''Takes a string, makes it lowercase and returns a list with all fields after the first one.

    Arguments:
    1 string

    Returns:
    1 list of strings'''

    return line.lower().split()[1:]

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

# ======================================================================= #
def printheader():
    '''Prints the formatted header of the log file. Prints version number and version date

    Takes nothing, returns nothing.'''

    print starttime,os.environ['HOSTNAME'],os.getcwd()
    if not PRINT:
        return
    string='\n'
    string+='    '+'='*80+'\n'
    string+='||'+' '*80+'||\n'
    string+='||'+' '*25+'SHARC - MOLCAS2012 - Interface'+' '*25+'||\n'
    string+='||'+' '*80+'||\n'
    string+='||'+' '*19+'Authors: Sebastian Mai and Martin Richter'+' '*20+'||\n'
    string+='||'+' '*80+'||\n'
    string+='||'+' '*(36-(len(version)+1)/2)+'Version: %s' % (version)+' '*(35-(len(version))/2)+'||\n'
    lens=len(versiondate.strftime("%d.%m.%y"))
    string+='||'+' '*(37-lens/2)+'Date: %s' % (versiondate.strftime("%d.%m.%y"))+' '*(37-(lens+1)/2)+'||\n'
    string+='||'+' '*80+'||\n'
    string+='    '+'='*80+'\n\n'
    print string
    if DEBUG:
        print changelogstring

# ======================================================================= #
def printQMin(QMin):
    '''If PRINT, prints a formatted Summary of the control variables read from the input file.

    Arguments:
    1 dictionary: QMin'''

    if DEBUG:
        pprint.pprint(QMin)
    if not PRINT:
        return
    print '==> QMin Job description for:\n%s' % (QMin['comment'])
    string='Tasks:'
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
    print string
    string='States: '
    for i in itmult(QMin['states']):
        string+='\t%i %s' % (QMin['states'][i-1],IToMult[i])
    print string
    string='Found Geo'
    if 'veloc' in QMin:
        string+=' and Veloc! '
    else:
        string+='! '
    string+='NAtom is %i.\n' % (QMin['natom'])
    print string
    string=''
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
    for i in QMin:
        if not i=='h' and not i=='dm' and not i=='soc' and not i=='geo' and not i=='veloc' and not i=='states' and not i=='comment':
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
        if task[0]=='restart':
            print 'Restart'
        elif task[0]=='expec':
            print 'Exp. Val.:\t%s' % (task[1])
        elif task[0]=='mcscf':
            print 'MCSCF:\tOrbitals'
        elif task[0]=='mcscf:pspace':
            print 'MCSCF:\tOrbitals\t\t\t\t\tP-Space Threshold: %.2f' % (task[1])
        elif task[0]=='ci':
            print 'CI:\tWavefunction\tMultiplicity: %i\tStates: %i' % (task[1],task[2])
        elif task[0]=='ci:pspace':
            print 'CI:\tWavefunction\tMultiplicity: %i\tStates: %i\tP-Space Threshold: %.2f' % (task[1],task[2],task[3])
        elif task[0]=='cihlsmat':
            print 'CI:\tSpin-Orbit Matrix'
            for j in range(len(task)-1):
                print '\t\t\tMultiplicity: %i' % (task[j+1])
        elif task[0]=='citrans':
            print 'CI:\tTrans. Mom\tMultiplicity: %i' % (task[1])
        elif task[0]=='ddr':
            print 'DDR:\tNon-Adiab. C.\tMultiplicity: %i\tStates: %i - %i' % (task[1],task[2],task[3])
        elif task[0]=='cpgrad':
            print 'MCSCF:\tz-Vector for Gradient'
            for i in range(len(task)-1):
                print '\t\t\tMultiplicity: %i\tState: %i\tAccuracy: %18.15f' % (task[i+1][0],task[i+1][1],task[i+1][2])
        elif task[0]=='forcegrad':
            print 'FORCE:\tGradient\tMultiplicity: %i\tState: %i' % (task[1][0],task[1][1])
        elif task[0]=='cpnac':
            print 'MCSCF:\tz-Vector for Non-Adiabatic Coupling'
            for i in range(len(task)-1):
                print '\t\t\tMultiplicity: %i\tStates: %i - %i\tAccuracy: %18.15f' % (task[i+1][0],task[i+1][1],task[i+1][2],task[i+1][3])
        elif task[0]=='forcenac':
            print 'FORCE:\tNAC\t\tMultiplicity: %i\tStates: %i - %i' % (task[1][0],task[1][1],task[1][2])
        elif task[0]=='casdiab':
            print 'MCSCF:\tDiab. Orbitals'
        elif task[0]=='cidiab':
            print 'CI:\t2x Trans. Mom\tMultiplicity: %i' % (task[1])
        elif task[0]=='ddrdiab':
            print 'OVERLAP' # 'DDR:\tOverlap Matrix\tMultiplicity: %i\tState: %i' % (task[1],task[2])
        elif task[0]=='dm':
            print 'DM' # 'DDR:\tOverlap Matrix\tMultiplicity: %i\tState: %i' % (task[1],task[2])
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
            string+='% .5f\t' % (grad[atom][xyz])
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

    if DEBUG:
        pprint.pprint(QMout)
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
    # Non-adiabatic couplings
    if 'nacdt' in QMin:
        print '=> Numerical Non-adiabatic couplings:\n'
        matrix=QMout['nacdt']
        printcomplexmatrix(matrix,states)
        matrix=deepcopy(QMout['mrcioverlap'])
        for i in range(nmstates):
            for j in range(nmstates):
                matrix[i][j]=complex(matrix[i][j])
        print '=> MRCI overlaps:\n'
        printcomplexmatrix(matrix,states)
        if 'phases' in QMout:
            print '=> Wavefunction Phases:\n'
            for i in range(nmstates):
                print '% 3.1f % 3.1f' % (QMout['phases'][i].real,QMout['phases'][i].imag)
            print '\n'
    if 'nacdr' in QMin:
            print '=> Analytical Non-adiabatic coupling vectors:\n'
            istate=0
            for imult,i,msi in itnmstates(states):
                jstate=0
                for jmult,j,msj in itnmstates(states):
                    if imult==jmult and msi==msj:
                        print '%s\tStates %i - %i\tMs= % .1f:' % (IToMult[imult],i,j,msi)
                        print '\t\t...is identical zero...\n' #printgrad(QMout['nacdr'][istate][jstate],natom,QMin['geo']) # TODO NACDR
                    jstate+=1
                istate+=1
    if 'overlap' in QMin:
        print '=> Overlap matrix:\n'
        matrix=QMout['overlap']
        printcomplexmatrix(matrix,states)
        if 'phases' in QMout:
            print '=> Wavefunction Phases:\n'
            for i in range(nmstates):
                print '% 3.1f % 3.1f' % (QMout['phases'][i].real,QMout['phases'][i].imag)
            print '\n'
    # Angular momentum matrices
    if 'angular' in QMin:
        print '=> Angular Momentum Matrices:\n'
        for xyz in range(3):
            print 'Polarisation %s:' % (IToPol[xyz])
            matrix=QMout['angular'][xyz]
            printcomplexmatrix(matrix,states)
    sys.stdout.flush()

# ======================================================================= #         OK
def nextblock(data,program='*',occ=1):
    '''Scans the list of strings data for the occ next occurence of MOLCAS program block for program. Returns the line number where the block ends and the block itself.

    Arguments:
    1 list of strings: data
    2 string: MOLCAS program name (like "MULTI" in "1PROGRAM * MULTI" )
    3 integer: return the i-th block of the specified program

    Returns:
    1 integer: line number in data where the specified block ends
    2 list of strings: The specified block'''

    progdata=[]
    i=-1
    while True:
        try:
            i+=1
            line=data[i].split()
            if line==[]:
                continue
            if containsstring('1PROGRAM',line[0]) and containsstring(program,line[2]):
                occ-=1
                if occ==0:
                    break
        except IndexError:
            print 'Block %s not found in routine nextblock! Probably MOLCAS encountered an error not anticipated in this script. Check the MOLCAS output!' % (program)
            sys.exit(16)
    progdata.append(data[i])
    i+=1
    while i<len(data) and not containsstring('1PROGRAM',data[i]):
        progdata.append(data[i])
        i+=1
    return i,progdata

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

# ======================================================================= #
def getversion(out):
  for i in range(50):
    line=out[i]
    s=line.split()
    for j,el in enumerate(s):
      if 'version' in el:
        v=float(s[j+1])
        print 'Found MOLCAS version %3.1f\n' % (v)
        return v

# ======================================================================= #
def getcienergy(out,mult,state,version):
    '''Searches a complete MOLCAS output file for the MRCI energy of (mult,state).

    Arguments:
    1 list of strings: MOLCAS output
    2 integer: mult
    3 integer: state

    Returns:
    1 float: total CI energy of specified state in hartree'''
    
    rasscf = False
    correct_mult = False
    for i, line in enumerate(out):
        if line.find('MOLCAS executing module RASSCF') != -1:
            rasscf = True # found RASSCF calculation
        elif line.find('Spin quantum number') != -1 and rasscf:
            spin = float(line.split()[3])
            if int(round(spin*2)) + 1 == mult:
                correct_mult = True
        elif line.find('::    RASSCF root number') != -1 and rasscf and correct_mult:
            if int(line.split()[4]) == state:
                if 7<=version<8:
                    return float(line.split()[8].strip())
                elif 8<=version<9:
                    return float(line.split()[7].strip())
    print 'CI energy of state %i in mult %i not found!' % (state,mult)
    sys.exit(17)

# ======================================================================= #
def getcidm(out,mult,state1,state2,pol,version):
    '''Searches a complete MOLCAS output file for a cartesian component of a dipole moment between the two specified states.

    Only takes one multiplicity, since in this script, only non-relativistic dipole moments are calculated. 
    If state1==state2, then this returns a state dipole moment, otherwise a transition dipole moment.

    Arguments:
    1 list of strings: MOLCAS output
    2 integer: mult
    3 integer: state1
    4 integer: state2
    5 integer (0,1,2) or character (X,Y,Z): Polarisation

    Returns:
    1 float: cartesian dipole moment component in atomic units'''
    if pol=='X' or pol=='Y' or pol=='Z':
        pol=IToPol[pol]
    
    rassi = False
    foundjobiph1 = False
    foundjobiph2 = False
    if 7<=version<8:
        version_dependent_number=11
    elif 8<=version<9:
        version_dependent_number=12
    for i, line in enumerate(out):
        if line.find('MOLCAS executing module RASSI') != -1:
            rassi = True
        if line.find('Specific data for JOBIPH file') != -1 and rassi:
            # check if multiplicity of this jobiph is the requested one
            jobiph = int(line.split()[-1].strip()[-3:])
            if jobiph == 1 and int(out[i+version_dependent_number].split()[-1]) == mult:
                foundjobiph1 = True
                jobiph1 = jobiph
            elif jobiph == 2 and int(out[i+version_dependent_number].split()[-1]) == mult:
                foundjobiph2 = True
                jobiph2 = jobiph
            else:
                rassi = False # not the correct jobiph files used. go on to next rassi calculation
        if line.find('JobIph:') != -1 and rassi and foundjobiph1 and foundjobiph2:
            # get indices of requested states
            jobiphlist = line.split()[1:]
            indexlist = out[i-1].split()[1:]
            rootslist = out[i+1].split()[2:]
            idx_max = int(indexlist[-1])
            for idx in range(len(indexlist)):
                # its necessary to use jobiph2 in both cases cause this file holds the current data
                # wheres jobiph1 holds the old data
                if int(jobiphlist[idx]) == jobiph2 and int(rootslist[idx]) == state1:
                    idx1 = int(indexlist[idx])
                if int(jobiphlist[idx]) == jobiph2 and int(rootslist[idx]) == state2:
                    idx2 = int(indexlist[idx])
        if line.find('PROPERTY: MLTPL  1   COMPONENT:') != -1 and rassi and foundjobiph1 and foundjobiph2:
            if int(line.split()[-1]) == pol + 1:
                # found correct polarisation
                # calculate efficient index including breaks of matrices every 4 columns
                idx1_eff = i+3+(idx2-1)/4*(idx_max+6)+idx1
                idx2_eff = idx2 - ((idx2-1)/4)*4
                return float(out[idx1_eff].split()[idx2_eff])
    print 'DM element not found!', mult,state1,state2,pol
    sys.exit(18)

# ======================================================================= #
def getciang(out,mult,state1,state2,pol):
    '''Searches a complete MOLCAS output file for a cartesian component of a angular momentum between the two specified states.

    Only takes one multiplicity, since in this script, only non-relativistic angular momenta are calculated. 
    If state1==state2, then this returns zero, otherwise a transition angular momentum.

    Arguments:
    1 list of strings: MOLCAS output
    2 integer: mult
    3 integer: state1
    4 integer: state2
    5 integer (0,1,2) or character (X,Y,Z): Polarisation

    Returns:
    1 complex: cartesian angular momentum component in atomic units'''

    if state1==state2:
        return complex(0.,0.)
    if pol=='X' or pol=='Y' or pol=='Z':
        pol=IToPol[pol]
    ilines=0
    while ilines<len(out):
        if containsstring('1PROGRAM \* CI',out[ilines]):
            while ilines<len(out):
                if containsstring('Reference symmetry',out[ilines]):
                    if containsstring(IToMult[mult],out[ilines]):

                        while not containsstring('\*\*\*', out[ilines]):
                            if containsstring('MRCI trans.*<.*\|L.\|.*>',out[ilines]):
                                braket=out[ilines].replace('<',' ').replace('>',' ').replace('|',' ').replace('.',' ').split()
                                s1=int(braket[2])
                                s2=int(braket[5])
                                p=IToPol[braket[4][1]]
                                if p==pol and s1==state1 and s2==state2:
                                    return complex(out[ilines].split()[3].replace('i','j'))
                                if p==pol and s1==state2 and s2==state1:
                                    return -complex(out[ilines].split()[3].replace('i','j')).conjugate()
                            ilines+=1
                        return complex(0.,0.)
                    else:
                        break
                ilines+=1
        ilines+=1
    print 'CI angular momentum of states %i and %i in mult %i not found!' % (state1,state2,mult)
    sys.exit(19)

def getMOLCASstatenumber(mult, state, ms, states):
    '''Calculates the state number used in MOLCAS from a given multiplicity, state number and MS value

    Arguments:
    1 integer: multiplicity
    2 integer: state number in this multiplicity
    3 integer: MS value
    4 list of integers: states specs

    Returns:
    1 integer: state number in MOLCAS'''
    
    statenumber = 0
    for m, mstates in enumerate(states): # iterate over multiplicities
        if m+1 < mult:
            statenumber += mstates*(m+1)
        else: # correct multiplicity found
            for nstate in range(1, mstates+1): # iterate over states
                if nstate < state:
                    statenumber += m+1
                else: # correct state found
                    statenumber += int(ms + 1 + 0.5*(mult))
                    return statenumber
    print 'getMOLCASstatenumber Error: mult=%i, state=%i, ms=%i not in' % (mult, state, ms), states
    sys.exit(20)

# ======================================================================= #
def getsocme(out, mult1, state1, ms1, mult2, state2, ms2, states,version):
    '''Searches a MOLCAS output for an element of the Spin-Orbit hamiltonian matrix. Also converts from cm^-1 to hartree and adds the diagonal shift.

    Arguments:
    1 list of strings: MOLCAS output
    2-4 integer: multiplicity, state and ms for state1
    5-7 integer: multiplicity, state and ms for state2
    8 list of integer: states specs

    Returns:
    1 complex: SO hamiltonian matrix element in hartree'''
     
    rcm_to_Eh=4.556335e-6

    if mult1 == mult2 and state1 == state2 and ms1 == ms2:
        # diagonal element can be taken from spin-free hamiltonian
        return complex(getcienergy(out,mult1,state1,version), 0.0)
    else:
        # get state numbers in molcas counting scheme
        s1 = getMOLCASstatenumber(mult1, state1, ms1, states)
        s2 = getMOLCASstatenumber(mult2, state2, ms2, states)
        # get matrix element from Spin-orbit section
        socme = complex(0.0, 0.0)
        soc_section = False
        for i, line in enumerate(out):
            if line.find('Spin-orbit section') != -1:
                soc_section = True
            elif line.find('  I1  S1  MS1    I2  S2  MS2    Real part    Imag part') != -1 and soc_section:
                linecounter = i
                while True:
                    linecounter += 1
                    tmpline = out[linecounter]
                    if tmpline.find('----------------------------------------------------') != -1:
                        break
                    tmpline = tmpline.split()
                    if int(tmpline[0]) in (s1, s2) and int(tmpline[3]) in (s1, s2) and tmpline[0] != tmpline[3]:
                        # found correct element. now check if indices are switched or not
                        if int(tmpline[0]) == s1:
                            socme = complex(float(tmpline[6]), float(tmpline[7]))
                        else:
                            socme = complex(float(tmpline[6]), -float(tmpline[7]))
                        break
        return socme * rcm_to_Eh

# ======================================================================= #
def getgrad(out,mult,state,natom, QMin):
    '''Searches a MOLCAS output for a SA-MCSCF gradient of a specified state.

    Arguments:
    1 list of strings: MOLCAS output
    2 integer: mult
    3 integer: state
    4 integer: natom

    Returns:
    1 list of list of floats: gradient vector (natom x 3) in atomic units'''
    #print 'getgrad called for', mult, state, natom 
    mclr = False
    alaska = False
    rasscf = False
    multfound = False
    statefound = False
    espf = False

    if 7<=QMin['version']<8:
        version_dependent_string='After ESPF, gradients are'
        version_dependent_number=2
    elif 8<=QMin['version']<9:
        version_dependent_string='Molecular gradients, after ESPF'
        version_dependent_number=8

    if 'total_qmmm_natom' in QMin:
        if QMin['total_qmmm_natom'] < natom:
            tmpnatom = natom #QMin['total_qmmm_natom']
        else:
            tmpnatom = natom
    else:
        tmpnatom = natom

    for i, line in enumerate(out):
        if 'ESPF' in line:
            espf = True # qmmm calculation give differently formatted gradients
        if 'MOLCAS executing module RASSCF' in line:
            rasscf=True
            mclr=False
            alaska=False
        elif ' MOLCAS executing module MCLR' in line:
            rasscf=False
            mclr = True
            alaska = False
        elif 'MOLCAS executing module ALASKA' in line and multfound and statefound:
            rasscf=False
            mclr = False
            alaska = True

        elif 'Spin quantum number' in line and rasscf:
            if int(round(float(line.split()[3])*2))+1 == mult:
                multfound = True
            else:
                multfound = False
            if QMin['states'][mult-1]==1:
                statefound=True
            else:
                statefound=False

        elif ' Lagrangian multiplier is calculated for root no.' in line and mclr and multfound:
            if int(line.split()[8].strip()) == state:
                statefound = True

        elif 'Molecular gradients' in line and alaska and multfound and statefound and not espf: # never go here for qmmm (because of espf check)
            # found correct gradient
            grad = []
            for atom in range(tmpnatom):
                grad.append([ float(out[i+xyz+6+atom*3].split()[2].strip()) for xyz in range(3) ])
            return grad
        elif version_dependent_string in line and alaska and multfound and statefound and espf: # always go here for qmmm (because of espf check)
            # found correct gradient
            grad = []
            for atom in range(tmpnatom): # go through all atoms
                if atom+1 in QMin['active_qmmm_atoms']: # if current atom is in the list of active qmmm atoms
                    atomindex = QMin['active_qmmm_atoms'].index(atom+1) # get index in the list of qmmm atoms -> is also the index in the list of gradients calculated by MOLCAS
                    try:
                        grad.append([ float(out[i+version_dependent_number+atomindex].split()[xyz+1].strip()) for xyz in range(3) ])
                    except ValueError:
                        print 'ValueError:'
                        print atom, atomindex
                        print QMin['active_qmmm_atoms']
                        sys.exit(21)
                    except IndexError:
                        print 'IndexError:'
                        print atom
                        print i, atomindex
                        print out[i+6+atomindex].strip()
                        print out[i+7+atomindex].strip()
                        print out[i+8+atomindex].strip()
                        print out[i+9+atomindex].strip()
                        sys.exit(22)
                else:
                    grad.append([ 0.0 for xyz in range(3)]) # set gradients to zero for inactive atoms
            #if tmpnatom < natom:
            #    # for qmmm with fixed atoms: fill gradients of fixed atoms with zeros
            #    for diff in range(natom-tmpnatom):
            #        grad.append([0.0 for xyz in range(3)])
            return grad
    print 'Gradient of state %i in mult %i not found!' % (state,mult)
    sys.exit(23)

# ======================================================================= #
def getnacana(out,mult,state1,state2,natom):
    '''Searches a MOLCAS output file for an analytical non-adiabatic coupling vector from SA-MCSCF. 

    Arguments:
    1 list of strings: MOLCAS output
    2 integer: mult
    3 integer: state1
    4 integer: state2
    5 integer: natom

    Returns:
    1 list of list of floats: non-adiabatic coupling vector (natom x 3) in atomic units'''

    ilines=0
    grad=[]
    multfound=False
    statefound=False
    # diagonal couplings are zero
    if state1==state2:
        for i in range(natom):
            grad.append([0.,0.,0.])
        return grad
    # look for FORCE program block
    while ilines<len(out):
        if containsstring('1PROGRAM \* FORCE',out[ilines]):
            # look for multiplicity and state
            jlines=ilines
            while not containsstring('\*\*\*',out[jlines]):
                if containsstring(IToMult[mult],out[jlines]):
                    multfound=True
                    break
                jlines+=1
            jlines=ilines
            while not containsstring('\*\*\*',out[jlines]):
                if containsstring('SA-MC NACME FOR STATES',out[jlines]):
                    line=out[jlines].replace('.',' ').replace('-',' ').split()
                    # make sure the NACs are antisymmetric
                    if state1==int(line[5]) and state2==int(line[7]):
                        statefound=True
                        factor=1.
                    if state1==int(line[7]) and state2==int(line[5]):
                        statefound=True
                        factor=-1.
                    break
                jlines+=1
            if multfound and statefound:
                jlines+=4
                for i in range(natom):
                    line=out[jlines+i].split()
                    for j in range(3):
                        try:
                            line[j+1]=factor*float(line[j+1])
                        except ValueError:
                            print 'Bad float in gradient in line %i! Maybe natom is wrong.' % (ilines+i)
                    grad.append(line[1:])
                return grad
            else:
                multfound=False
                statefound=False
                ilines+=1
                continue
        ilines+=1
    print 'Non-adiatic coupling of states %i - %i in mult %i not found!' % (state1,state2,mult)
    sys.exit(24)

# ======================================================================= #
def getmrcioverlap(out,mult,state1,state2):
    '''Searches a MOLCAS output for a single MRCI overlap (from a CI trans calculation).

    Arguments:
    1 list of strings: MOLCAS output
    2 integer: mult
    3 integer: state1
    4 integer: state2

    Returns:
    1 float: MRCI overlap (THIS MATRIX IS NOT SYMMETRIC!)'''
    
    if state1 == state2:
        return 1.0
    else:
        return 0.0
#    ilines=0
#    while ilines<len(out):
#        if containsstring('Ket wavefunction restored from record .*\.3',out[ilines]):
#            line=out[ilines].replace('.',' ').split()
#            if mult==int(line[5])-6000:
#                break
#        ilines+=1
#    while not containsstring('\*\*\*',out[ilines]):
#        if containsstring('!MRCI overlap',out[ilines]):
#            braket=out[ilines].replace('<',' ').replace('>',' ').replace('|',' ').replace('.',' ').split()
#            s1=int(braket[2])
#            s2=int(braket[4])
#            # overlap matrix is NOT symmetric! 
#            if s1==state1 and s2==state2:
#                return float(out[ilines].split()[3])
#        ilines+=1
#    return 0.

# ======================================================================= #
def getnacnum(out,mult,state1,state2):
    '''Searches a MOLCAS output for a single non-adiabatic coupling matrix element from a DDR calculation.

    Arguments:
    1 list of strings: MOLCAS output
    2 integer: mult
    3 integer: state1
    4 integer: state2

    Returns:
    1 float: NAC matrix element'''

    # diagonal couplings are zero
    if state1==state2:
        return 0.
    ilines=0
    multfound=False
    statefound=False
    while ilines<len(out):
        if containsstring('Construct non-adiabatic coupling elements by finite difference method', out[ilines]):
            jlines=ilines
            while not containsstring('\*\*\*',out[jlines]):
                if containsstring('Transition density \(R\|R\+DR\)',out[jlines]):
                    line=out[jlines].replace('.',' ').replace('-',' ').split()
                    if mult==int(line[4])-8000:
                        multfound=True
                    if state1==int(line[8]) and state2==int(line[10]):
                        statefound=True
                        factor=1.
                    if state1==int(line[10]) and state2==int(line[8]):
                        statefound=True
                        factor=-1.
                    if multfound and statefound:
                        jlines+=5
                        return factor*float(out[jlines].split()[2])
                    else:
                        multfound=False
                        statefound=False
                        ilines+=1
                jlines+=1
        ilines+=1
    print 'Non-adiatic coupling of states %i - %i in mult %i not found!' % (state1,state2,mult)
    sys.exit(25)

# ======================================================================= #
def getsmate(out,mult,state1,state2,states):
    '''Searches a MOLCAS output for an element of the total adiabatic-diabatic transformation matrix.

    Arguments:
    1 list of strings: MOLCAS output
    2 integer: mult
    3 integer: state1
    4 integer: state2
    5 list of integer: states specs

    Returns:
    1 float: Adiabatic-Diabatic transformation matrix element (MATRIX IS NOT SYMMETRIC!)'''

    rassi = False
    for i, line in enumerate(out):
        if line.find('MOLCAS executing module RASSI') != -1:
            rassi = True
            jobiphmult = []
        elif line.find('MOLCAS executing module') != -1:
            rassi = False
        elif line.find('SPIN MULTIPLICITY:') != -1 and rassi:
            jobiphmult.append(int(line.split()[2].strip()))
        if line.find('OVERLAP MATRIX FOR THE ORIGINAL STATES:') != -1 and  rassi:
            # check if all loaded jobiphs have correct multiplicity
            multcorrect = True
            for m in jobiphmult:
                if m != mult:
                    multcorrect = False
            if multcorrect:
                # convert state1 and state2 numbers to indices of overlap matrix indices
                # matrix given from molcas is of size 2Nx2N with N being the number of
                # states in the given multiplicity.
                # We are interested in the lower left quarter of this matrix.
                # TODO: Corrections of sign changes still has to be checked!
                #   - if diagonal element is about -1 full column or row has to be multiplied by -1
                #   EDIT 10. 1. 2014: Not necessary since SHARC does this internally.
                lineoffset = 2
                N = states[mult-1]
                for tmp in range(N+state1-1):
                    lineoffset += tmp/5 + 1 # MOLCAS prints matrix only with 5 elements per line
                lineoffset += (state2 - 1)/5
                lineoffset_diag = lineoffset + (state1 - 1)/5 # state1 = state2
                col = state2 - ((state2 - 1)/5) * 5 - 1
                col_diag = state1 - ((state1 - 1)/5) * 5 - 1 # state1 = state2
                ovl = float(out[i + lineoffset].split()[col].strip())
                ovl_diag = float(out[i + lineoffset_diag].split()[col_diag].strip())
                #if ovl_diag < 0.0:
                #    ovl *= -1.0
                return ovl
    print 'Overlap of states %i - %i in mult %i not found!' % (state1,state2,mult)
    sys.exit(26)

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
def writeQMoutang(QMin,QMout):
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
    string+='! %i Angular Momentum Matrices (3x%ix%i, complex)\n' % (9,nmstates,nmstates)
    for xyz in range(3):
        string+='%i %i\n' % (nmstates,nmstates)
        for i in range(nmstates):
            for j in range(nmstates):
                string+='%s %s ' % (eformat(QMout['angular'][xyz][i][j].real,9,3),eformat(QMout['angular'][xyz][i][j].imag,9,3))
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
def writeQMoutnacnum(QMin,QMout):
    '''Generates a string with the NAC matrix in SHARC format.

    The string starts with a ! followed by a flag specifying the type of data. In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements. Blocks are separated by a blank line.

    Arguments:
    1 dictionary: QMin
    2 dictionary: QMout

    Returns:
    1 string: multiline string with the NAC matrix'''

    states=QMin['states']
    nstates=QMin['nstates']
    nmstates=QMin['nmstates']
    natom=QMin['natom']
    string=''
    string+='! %i Non-adiabatic couplings (ddt) (%ix%i, complex)\n' % (4,nmstates,nmstates)
    string+='%i %i\n' % (nmstates,nmstates)
    for i in range(nmstates):
        for j in range(nmstates):
            string+='%s %s ' % (eformat(QMout['nacdt'][i][j].real,9,3),eformat(QMout['nacdt'][i][j].imag,9,3))
        string+='\n'
    string+='\n'
    # also write wavefunction phases
    string+='! %i Wavefunction phases (%i, complex)\n' % (7,nmstates)
    for i in range(nmstates):
        string+='%s %s ' % (eformat(QMout['phases'][i],9,3),eformat(0.,9,3))
    string+='\n\n'
    return string

# ======================================================================= #
def writeQMoutnacana(QMin,QMout):
    '''Generates a string with the NAC vectors in SHARC format.

    The string starts with a ! followed by a flag specifying the type of data. On the next line, natom and 3 are written, followed by the gradient, with one line per atom and a blank line at the end. Each MS component shows up (nmstates x nmstates vectors are written).

    Arguments:
    1 dictionary: QMin
    2 dictionary: QMout

    Returns:
    1 string: multiline string with the NAC vectors'''

    states=QMin['states']
    nstates=QMin['nstates']
    nmstates=QMin['nmstates']
    natom=QMin['natom']
    string=''
    string+='! %i Non-adiabatic couplings (ddr) (%ix%ix%ix3, real)\n' % (5,nmstates,nmstates,natom)
    i=0
    for imult,istate,ims in itnmstates(states):
        j=0
        for jmult,jstate,jms in itnmstates(states):
            string+='%i %i ! %i %i %i %i %i %i\n' % (natom,3,imult,istate,ims,jmult,jstate,jms)
            for atom in range(natom):
                for xyz in range(3):
                    string+='%s ' % (0.0 ) #eformat(QMout['nacdr'][i][j][atom][xyz],9,3)) # TODO NACDR
                string+='\n'
            #string+='\n'
            j+=1
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
            string+='%s %s ' % (eformat(QMout['overlap'][i][j].real,9,3),eformat(QMout['overlap'][i][j].imag,9,3))
        string+='\n'
    string+='\n'
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
            sys.exit(27)
    else:
        try:
            os.makedirs(SCRATCHDIR)
        except OSError:
            print 'Can not create SCRATCHDIR=%s\n' % (SCRATCHDIR)
            sys.exit(28)

# ======================================================================= #
def removequotes(string):
  if string.startswith("'") and string.endswith("'"):
    return string[1:-1]
  elif string.startswith('"') and string.endswith('"'):
    return string[1:-1]
  else:
    return string

# ======================================================================= #
def getsh2colkey(sh2col,key):
  i=-1
  while True:
    i+=1
    try:
      line=re.sub('#.*$','',sh2col[i])
    except IndexError:
      break
    line=line.split(None,1)
    if line==[]:
      continue
    if key.lower() in line[0].lower():
      return line
  return ['','']

# ======================================================================= #
def get_sh2col_environ(sh2col,key,environ=True,crucial=True):
  line=getsh2colkey(sh2col,key)
  if line[0]:
    LINE=line[1]
  else:
    if environ:
      LINE=os.getenv(key.upper())
      if not LINE:
        if crucial:
          print 'Either set $%s or give path to %s in SH2CAS.inp!' % (key.upper(),key.upper())
          sys.exit(29)
        else:
          return ''
    else:
      if crucial:
        print 'Give path to %s in SH2CAS.inp!' % (key.upper())
        sys.exit(30)
      else:
        return ''
  LINE=os.path.expandvars(LINE)
  LINE=os.path.expanduser(LINE)
  LINE=removequotes(LINE).strip()
  if containsstring(';',LINE):
    print "$%s contains a semicolon. Do you probably want to execute another command after %s? I can't do that for you..." % (key.upper(),key.upper())
    sys.exit(31)
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
      sys.exit(32)
    if 'end' in line:
      break
    fields=line.split()
    try:
      nacpairs.append([int(fields[0]),int(fields[1])])
    except ValueError:
      print '"nacdr select" is followed by pairs of state indices, each pair on a new line!'
      sys.exit(33)
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
    try:
        QMinfile=open(QMinfilename,'r')
    except IOError:
        print 'QM input file "%s" not found!' % (QMinfilename)
        sys.exit(34)
    QMinlines=QMinfile.readlines()
    QMinfile.close()
    QMin={}

    # Get natom
    try:
        natom=int(QMinlines[0])
    except ValueError:
        print 'first line must contain the number of atoms!'
        sys.exit(35)
    QMin['natom']=natom
    if len(QMinlines)<natom+4:
        print 'Input file must contain at least:\nnatom\ncomment\ngeometry\nkeyword "states"\nat least one task'
        sys.exit(36)

    # Save Comment line
    QMin['comment']=QMinlines[1]

    # Get geometry and possibly velocity (for backup-analytical non-adiabatic couplings)
    QMin['geo']=[]
    QMin['veloc']=[]
    hasveloc=True
    for i in range(2,natom+2):
        if not containsstring('[a-zA-Z][a-zA-Z]?[0-9]*.*[-]?[0-9]+[.][0-9]*.*[-]?[0-9]+[.][0-9]*.*[-]?[0-9]+[.][0-9]*', QMinlines[i]):
            print 'Input file does not comply to xyz file format! Maybe natom is just wrong.'
            sys.exit(37)
        fields=QMinlines[i].split()
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
        args=line.lower().split()[1:]
        if key in QMin:
            print 'Repeated keyword %s in line %i in input file! Check your input!' % (linekeyword(line),i+1)
            continue  # only first instance of key in QM.in takes effect
        if len(args)>=1 and 'select' in args[0]:
            pairs,i=get_pairs(QMinlines,i)
            QMin[key]=pairs
        else:
            QMin[key]=args


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


    # Various logical checks
    if not 'states' in QMin:
        print 'Number of states not given in QM input file %s!' % (QMinfilename)
        sys.exit(38)

    possibletasks=['h','soc','dm','grad','nacdr','nacdt','overlap']
    if not any([i in QMin for i in possibletasks]):
        print 'No tasks found! Tasks are "h", "soc", "dm", "grad", "nacdt", "nacdr" and "overlap".'
        sys.exit(39)

    if 'samestep' in QMin and 'init' in QMin:
        print '"Init" and "Samestep" cannot be both present in QM.in!'
        sys.exit(40)

    if 'overlap' in QMin and 'init' in QMin:
        print '"overlap" cannot be calculated in the first timestep! Delete either "overlap" or "init"'
        sys.exit(41)

    if not 'init' in QMin and not 'samestep' in QMin:
        QMin['newstep']=[]

    if not any([i in QMin for i in ['h','soc','dm','grad','nacdt','nacdr']]) and ('overlap' in QMin or 'ion' in QMin):
        QMin['h']=[]

    if len(QMin['states'])>8:
        print 'Higher multiplicities than octets are not supported!'
        sys.exit(42)

    if 'h' in QMin and 'soc' in QMin:
        QMin=removekey(QMin,'h')

    if 'nacdt' in QMin or 'nacdr' in QMin:
        print 'Within the SHARC-MOLCAS interface couplings can only be calculated via the overlap method. "nacdr" and "nacdt" are not supported.'
        sys.exit(43)


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
                    sys.exit(44)
                if QMin['grad'][i]>nmstates:
                    print 'State for requested gradient does not correspond to any state in QM input file state list!'
                    sys.exit(45)

    # Check for correct nac list
    if 'nacdr' in QMin:
        if len(QMin['nacdr'])>=1:
            nacpairs=QMin['nacdr']
        for i in range(len(nacpairs)):
            if nacpairs[i][0]>nmstates or nacpairs[i][1]>nmstates:
                print 'State for requested non-adiabatic couplings does not correspond to any state in QM input file state list!'
                sys.exit(46)
        else:
            QMin['nacdr']=[ [j+1,i+1] for i in range(nmstates) for j in range(i)]




    # open SH2COL.inp
    sh2colf=open('SH2CAS.inp','r')
    sh2col=sh2colf.readlines()
    sh2colf.close()

    QMin['pwd']=os.getcwd()
    QMin['SHQM']=QMin['pwd']

    QMin['molcas']=get_sh2col_environ(sh2col,'molcas')
    os.environ['MOLCAS']=QMin['molcas']
    QMin['qmexe']=QMin['molcas']

    QMin['tinker']=get_sh2col_environ(sh2col,'tinker',crucial=False)
    os.environ['TINKER']=QMin['tinker']

    QMin['scratchdir']=get_sh2col_environ(sh2col,'scratchdir',False)
    QMin['WorkDir']=QMin['scratchdir']
    os.environ['WorkDir']=QMin['scratchdir']

    QMin['memory']=10
    line=getsh2colkey(sh2col,'memory')
    if line[0]:
        try:
            QMin['memory']=int(line[1])
        except ValueError:
            print 'MOLCAS memory does not evaluate to numerical value!'
            sys.exit(47)
    else:
        print 'WARNING: Please set memory for MOLCAS in SH2CAS.inp (in MB)! Using 10 MB default value!'
    os.environ['MOLCASMEM']=str(QMin['memory'])

    QMin['Project']=get_sh2col_environ(sh2col,'Project',crucial=False)
    if not 'Project' in QMin or QMin['Project']=='':
        QMin['Project']='Comment_'+re.sub('\s+', '', QMin['comment'])
    os.environ['Project']=QMin['Project']


    ## Set default gradient accuracies and get accuracies from environment
    QMin['gradaccudefault']=1e-7
    QMin['gradaccumax']=1e-2
    #try:
        #line=getsh2colkey(sh2col,'gradaccudefault')
        #if line[0]:
            #QMin['gradaccudefault']=float(line[1])
        #line=getsh2colkey(sh2col,'gradaccumax')
        #if line[0]:
            #QMin['gradaccumax']=float(line[1])
    #except ValueError:
        #print 'Gradient accuracy-related environment variables do not evaluate to numerical values!'
        #sys.exit(48)

    return QMin

# ======================================================================= #
def gettasks(QMin):
    '''Sets up a list of list specifying the kind and order of MOLCAS calculations.

    Each of the lists elements is a list, with a keyword as the first element and a number of additional information depending on the task.

    The list is set up according to a number of task keywords in QMin and the states specifications. These are:
    - h                         Calculate the non-relativistic hamiltonian
    - soc                     Calculate the spin-orbit hamiltonian
    - dm                        Calculate the non-relativistic dipole moment matrices
    - grad                    Calculate non-relativistic SA-MCSCF gradients for the specified states
                    * all                     Calculate gradients for all states in "states"
                    * list of int     Calculate only the gradients of these states (nmstates scheme indices)
    - nac                     Calculate the non-adiabatic couplings
                    * num                     Use the MOLCAS DDR program to obtain the matrix < i |d/dt| j >
                    * ana                     Use MOLCAS CPMCSCF to obtain the matrix of vectors < i |d/dR| j > 
                    * smat                    Use MOLCAS DDR to obtain the transformation matrix < i(t) | j(t+dt) >
                    * numfromana        Use MOLCAS CPMCSCF to obtain v * < i |d/dR| j >

    From this general requests, the specific MOLCAS tasks are created.
    Tasks are:
    - restart                             Use old wavefunction files, do not obtain new orbitals
    - mcscf                                 Dont use old wavefunction files, write a new geometry, do a MCSCF calculation to obtain orbitals
    - mcscf:pspace                    Like mcscf, but do not move the old wavefunctions files and include pspace threshold in input file
                    * 1 float: pspace threshold
    - ci                                        Recalculate the MCSCF wavefunction in the MRCI module for all states of mult
                    * 1 integer: mult
    - cihlsmat                            Calculate the SOC matrix with the AMFI approximation for the given multiplicities
                    * list of integer: multiplicities
    - cpgrad                                Solve the z-vector equations for the gradient of the specified state
                    * 1 integer: mult
                    * 2 integer: state
                    * 3 float: accuracy
    - forcegrad                         Calculate the gradient for this state
                    * 1 integer: mult
                    * 2 integer: state
    - citrans                             Calculate the transition moments between the last step and the current step for the given mult
                    * 1 integer: mult
    - ddr                                     Calculate the NAC matrix element for the specified states
                    * 1 integer: mult
                    * 2 integer: state1
                    * 3 integer: state2
    - cpnac                                 Solve the z-vector equations for the NAC vector between the given states
                    * 1 integer: mult
                    * 2 integer: state1
                    * 3 integer: state2
                    * 4 float: accuracy
    - forcenac                            Calculate the NAC vector for the given states
                    * 1 integer: mult
                    * 2 integer: state1
                    * 3 integer: state2
    - casdiab                             Calculate diabatic orbitals (which maximise the overlap to the last step orbitals)
    - cidiab                                Calculate the transition moments for the current step and between current and last step
                    * 1 integer: mult
    - ddrdiab                             Calculate the adiabatic-diabatic transformation matrix for all states in mult
                    * 1 integer: mult
                    * 2 integer: states

    Arguments:
    1 dictionary: QMin

    Returns:
    1 list of lists: tasks'''

    states=QMin['states']
    nstates=QMin['nstates']
    nmstates=QMin['nmstates']
    # Currently implemented keywords: soc, dm, grad, nac, restart
    tasks=[]
    # calculate new orbitals if no restart
    # appends "mcscf"
    if 'restart' in QMin:
        tasks.append(['restart'])
    else:
        tasks.append(['mcscf'])
    if 'angular' in QMin:
        tasks.append(['expec','lop'])
    # recalculate wavefunctions with ci module
    # appends for each multiplicity "ci <mult> <states[mult]>"
    possibletasks=['dm','h','soc','nacdt','overlap','angular']
    if any([i in QMin for i in possibletasks]):
    #if 'dm' in QMin or 'soc' in QMin or 'h' in QMin or ( 'nac' in QMin and QMin['nac'][0]=='num') or ( 'nac' in QMin and QMin['nac'][0]=='smat') or 'angular' in QMin:
        for i in itmult(states):
            tasks.append(['ci',i,states[i-1]])
    # calculate the spin orbit matrix
    # appends "hlsmat <list of mults>"
    if 'soc' in QMin:
        tupel=['cihlsmat']
        for i in itmult(states):
            tupel.append(i)
        tasks.append(tupel)
    # calculate gradients
    # appends a series of "cpmscsf <list of states>" "force <state>"...
    if 'grad' in QMin:
        grads=[]
        if QMin['grad'][0]=='all':
            for mult,state in itnstates(states):
                grads.append([mult,state,QMin['gradaccudefault']])
        else:
            args=QMin['grad']
            for i in range(len(args)):
                mult1,state1=IstateToMultState(args[i],states)
                alreadydone=False
                for j in range(i):
                    mult2,state2=IstateToMultState(args[j],states)
                    if mult1==mult2 and state1==state2:
                        alreadydone=True
                if not alreadydone:
                    grads.append([mult1,state1,QMin['gradaccudefault']])
        for i in range(len(grads)):
            tasks.append(['cpgrad',grads[i]])
            tasks.append(['forcegrad',grads[i][0:2]])
    # calculate non-adiabatic couplings (ddr)
    if 'nacdt' in QMin:
        # Case of ddr non-adiabatic couplings
        if not 'dt' in QMin:
            print 'Task "NAC Num" needs keyword dt!'
            sys.exit(49)
        for mult in itmult(states):
            tasks.append(['citrans',mult])
        for mult,i1,i2 in ittwostates(states):
            tasks.append(['ddr',mult,i1,i2])
        # Case of analytical non-adiabatic couplings
    if 'nacdr' in QMin:
        nacs=[]
        if len(QMin['nacdr'])==2 and QMin['nacdr'][0]=='select':
            nacpairs=QMin['nacdr'][1]
            for i in range(len(nacpairs)):
                m1,i1=IstateToMultState(nacpairs[i][0],states)
                m2,i2=IstateToMultState(nacpairs[i][1],states)
                if m1==m2 and i1!=i2:
                    alreadydone=False
                    for j in range(i):
                        m1a,i1a=IstateToMultState(nacpairs[j][0],states)
                        m2a,i2a=IstateToMultState(nacpairs[j][1],states)
                        if m1==m1a==m2a and ( (i1==i1a and i2==i2a) or (i1==i2a and i2==i1a) ):
                            alreadydone=True
                    if not alreadydone:
                        nacs.append([m1,i1,i2,QMin['gradaccudefault']])
        else:
            for mult,i1,i2 in ittwostates(states):
                nacs.append([mult,i1,i2,QMin['gradaccudefault']])
        for i in range(len(nacs)):
            tasks.append(['cpnac',nacs[i]])
            tasks.append(['forcenac',nacs[i][0:3]])
        # Case of overlap matrix
        # TODO: needs to recalculate the CI vectors!
    if 'overlap' in QMin:
        tasks.append(['ddrdiab'])
        if DEBUG:
          print 'TEST: added ddrdiab because of overlap keyword'
    if 'dm' in QMin:
        tasks.append(['dm'])
    if 'smat' in QMin: # previously called overlap
        tasks.append(['casdiab'])
        for mult in itmult(states):
            tasks.append(['cidiab',mult])
            tasks.append(['ddrdiab',mult,states[mult-1]])
    # Case numfromana
    if 'nacdtfromdr' in QMin:
        for mult,i1,i2 in ittwostates(states):
            tasks.append(['cpnac',[mult,i1,i2,QMin['gradaccudefault']]])
            tasks.append(['forcenac',[mult,i1,i2]])
    return tasks

# ======================================================================= #
def writeMOLCASinput(tasks, QMin):
    '''Prepares all files for the next MOLCAS run as specified by the tasks list. Creates the geometry file, moves/copies wavefunction files and writes the MOLCAS input file based on the template file.

    The routine accomplishes:
    - writes geometry file "geom.xyz"
    - opens file "MOLCAS.template"
    - copies title and memory specs from template
    - sets up MOLCAS wavefunction files:
                    * if new orbitals are needed, renames the old wavefunction files
                    * if restart/mcscf:pspace is requested, does not rename files
                    * checks whether old wavefunctions exist, if NACs are needed
                    * writes the corresponding file units into MOLCAS input
    - copies global options (basis set, DK, etc.) from template to input
    - sets up MOLCAS geometry input (no reorientation, correct units, no symmetry)
    - reads and parses the casscf block of the template to obtain the active space and SA information, checks for consistency
    - finally, creates input for all tasks in the list

    Arguments:
    1 list of lists: tasks list
    2 dictionary: QMin'''
    
    # use the MOLCAS template file: currently only supports templates for casscf and global options ==================== #
    filename=os.path.join(QMin['SHQM'],'MOLCAS.template')
    try:
        templatefile=open(filename,'r')
    except IOError:
        print 'Need MOLCAS setup file "MOLCAS.template"!\nCould not find it at %s' % (filename)
        sys.exit(50)
    template=templatefile.readlines()
    templatefile.close()
    # get settings from molcas setup file =============================================================================== #
    QMsettings = {}
    QMsettings['roots'] = [0 for i in range(8)] # holds number of states for each multiplicity until octets
    QMsettings['qmmm'] = False
    for line in template:
        # ignore comments and blank lines
        if line.strip()=='' or line.strip()[0]=='!':
            continue
        line = line.split()
        if len(line) == 2:
            if line[0] in ('nactel', 'inactive', 'ras2'):
                QMsettings[line[0]] = int(line[1])
            # PARALLEL_GRADIENTS option is currently DEACTIVATED
            #elif line[0] == 'parallel_gradients':
                #QMsettings[line[0]] = int(line[1])
                #QMin['parallel_gradients'] = int(line[1])
            elif line[0] == 'basis':
                QMsettings[line[0]] = line[1].strip()
        elif line[0] == 'spin':
            try:
                QMsettings['roots'][int(line[1])-1] = int(line[3].strip())
            except IndexError:
                print 'IndexError in line:', line
                sys.exit(51)
        elif line[0].strip() == 'qmmm':
            QMsettings['qmmm'] = True
            print 'QM/MM request found'
            if QMin['tinker']=='':
              print 'Please set $TINKER or give path to tinker in SH2CAS.inp!'
              sys.exit(52)

    # CAS(2,2) does not allow for SOC calculations
    if QMsettings['nactel']==2 and QMsettings['ras2']==2:
        if 'soc' in QMin:
            print 'WARNING: CAS(2,2) yields zero cm^-1 for all SOC matrix elements in MOLCAS!'
    
    # if WorkDir does not exist, create it
    if not os.path.exists(QMin['WorkDir']):
        print 'WorkDir does not exist'
        #try:
        os.makedirs(QMin['WorkDir'])
        #except OSError:
        #    print 'OSError'
        #    os.makedirs(QMin['WorkDir'], 0000)
        #    os.chmod(QMin['WorkDir'], 0755)
    # copy *JobIph.old files there if you find them in $SHQM
    for element in os.listdir(QMin['SHQM']):
        if element.endswith('.JobIph.old'): # found an old JobIph file
            # see if states of this multiplicity are requested
            a=element.split('.')
            if len(a)==4:
              mult = int(a[-3])
            if QMsettings['roots'][mult-1] > 0:
                # copy the file to $WorkDir
                shutil.copy(os.path.join(QMin['SHQM'],element), os.path.join(QMin['WorkDir'],'%s.%i.JobIph.old' % (QMin['Project'],mult)) )
        if element.endswith('MOLCAS.qmmm.key'): # copy also .key file
            shutil.copy(os.path.join(QMin['SHQM'],element), os.path.join(QMin['WorkDir'],'%s.key' % (QMin['Project'])))
        if element.endswith('MOLCAS.qmmm.template'): # copy also .key file
            shutil.copy(os.path.join(QMin['SHQM'],element), os.path.join(QMin['WorkDir'],'%s.xyz.template' % (QMin['Project'])))

    # switch to WorkDir
    os.chdir(QMin['WorkDir'])

    # set up the geometry file ======================================================================================== #
    if QMsettings['qmmm']:
        QMin['active_qmmm_atoms'] = []
        # check the key file for the number of atoms that MOLCAS will calculate gradients for
        keyfile = open('%s.key' % QMin['Project'], 'r')
        keycontent = keyfile.readlines()
        keyfile.close()
        for line in keycontent:
            if line.find('QMMM') != -1 and line.find('QMMM-ELECTRO') == -1:
                QMin['total_qmmm_natom'] = int(line.split()[1])
            elif line[0:3] == 'MM ' or line[0:3] == 'QM ': # find definition of active atoms in QM and MM part
                line = line.split()
                if len(line) == 3 and int(line[1]) < 0: # found range definition (e.g.: "QM -1 15")
                    start_index = -int(line[1])
                    end_index = int(line[2])
                    for tmpindex in range(start_index, end_index+1):
                        QMin['active_qmmm_atoms'].append(tmpindex)
                else: # normal set of active atom indices
                    for element in line[1:]:
                        QMin['active_qmmm_atoms'].append(int(element))
        print 'total amount of qmmm atoms given in key file:', QMin['total_qmmm_natom']
        print 'number of indices given in key file:', len(QMin['active_qmmm_atoms'])
        # write xyz file using .xyz.template file
        geomtmpfile = open('%s.xyz.template' % QMin['Project'], 'r')
        geomtemplate = geomtmpfile.readlines()
        geomtmpfile.close()
        geostr = '%i\n' % QMin['natom']
        for i in range(QMin['natom']):
            tmpline = geomtemplate[i+1].split()
            for xyz in range(3):
                tmpline[xyz+2] = ' %f ' % QMin['geo'][i][xyz+1]
            geostr += ' '.join(tmpline) + '\n'
        geofile=open('%s.xyz' % QMin['Project'], 'w')
        geofile.write(geostr)
        geofile.close()
    else:
        geofile=open('geom.xyz','w')
        geofile.write('%i\n' % (QMin['natom']))
        if 'unit' in QMin:
            geofile.write('%s\n' % (QMin['unit'][0].strip()))
        else:
            geofile.write('Geometry for: '+QMin['comment'])
        for i in range(QMin['natom']):
            line=QMin['geo'][i][0]
            for j in range(3):
                line+=' %15.9f' % QMin['geo'][i][j+1]
            line+='\n'
            geofile.write(line)
        geofile.close()


    # open the MOLCAS input file ======================================================================================= #
    m_in = '' # initialise molcas input string
    
    # molcas reads title and memory from environment variables in runQM.sh
    # set up &gateway
    if QMsettings['qmmm']:
        #m_in += '>  EXPORT  TINKER=$MOLCAS/tinker/bin_qmmm\n&gateway\nTinker\ngroup=Nosym\nbasis=%s\n' % QMsettings['basis']
        m_in += '&gateway\nTinker\ngroup=Nosym\nbasis=%s\n' % QMsettings['basis']
    else:
        m_in += '&gateway\ncoord=geom.xyz\ngroup=Nosym\nbasis=%s\n' % QMsettings['basis']
    # set up &seward
    m_in += '&SEWARD\nR02O02\namfi\n'
    if QMsettings['qmmm']:
        m_in += '&Espf\nExternal=Tinker\n'

    m_in_parallel = m_in # template for parallel gradient job input
    # set up MOLCAS file units, depending on restart or orbital calculation ============================================= #
    print tasks[0]
    if tasks[0]==['mcscf'] or tasks[0]==['restart']:
        # check how many different spins are requested
        multlist = []
        for i, mult in enumerate(QMsettings['roots']):
            if mult != 0:
                multlist.append(i)
        # create one RASSCF calculation for every multiplicity
        pname = QMin['Project'] #os.getenv('Project')
        if not pname:
            pname = 'universalT1000'
            print 'Did not find environment variable Project. Setting pname to default %s.' % pname
        for mult in multlist:
            if os.path.exists('%s.%i.JobIph.old' % (pname, mult+1)):
                m_in += '>> LINK %s.%i.JobIph.old JOBOLD\n' % (pname, mult+1)
            m_in += '&RASSCF\n'
            if os.path.exists('%s.%i.JobIph.old' % (pname, mult+1)):
                m_in += 'jobiph\n'
            # move wf.last to wf.prelast and wf.current to wf.last
            #exist=os.path.exists('wf.current')
            #if exist:
            #    try:
            #        os.rename('wf.last','wf.prelast')
            #    except OSError:
            #        pass
            #    try:
            #        os.rename('wf.current','wf.last')
            #    except OSError:
            #        pass
            #exist = os.path.exists('wf.last')
            #if exist:
            #    m_in += 'fileorb = wf.last\n'
            m_in += 'spin=%i\nnactel=%i 0 0\ninactive=%i\nras2=%i\nciroot=%i %i 1\n' % (mult+1, QMsettings['nactel'], QMsettings['inactive'],
                                                                                        QMsettings['ras2'], QMsettings['roots'][mult], QMsettings['roots'][mult])
            m_in += '>> COPY $WorkDir/$Project.JobIph $WorkDir/$Project.%i.JobIph\n' % (mult+1)
    #elif tasks[0]==['restart']:
    #    # check if wf file is actually there
    #    exist=os.path.isfile('wf.current')
    #    if not exist:
    #        exist=os.path.isfile('%s/wf.current' % (QMin['scratchdir']))
    #    if not exist:
    #        print 'Restart requested, but no wf.current found!'
    #        sys.exit(53)
    #    inp.write('file,1,./integrals\n')
    #    inp.write('file,2,./wf.current\n')
    else:
        print 'Tasks should start with either mcscf or restart!'
        sys.exit(54)
    
    # ======================= Here starts parsing of the tasks step by step ================= #
    ddrdiab_or_dm = False
    for itask in range(len(tasks)):
        task=tasks[itask]
        string=''
        # restart: do nothing ============================================================================================== #
        if task[0]=='restart':
            pass
        # expec: everything already taken care of ========================================================================== #
        elif task[0]=='expec':
            pass
        # mcscf: create a casscf block including maxiter, ASblock, orbital records, WFblock ================================ #
        elif task[0]=='mcscf':
            pass
        elif task[0]=='mcscf:pspace':
            pass
        elif task[0]=='ci':
            pass
        # same as above, but with nstati statement (convergence helper) ==================================================== #
        elif task[0]=='ci:nstati':
            print 'ci:nstati is not yet implemented!'
            sys.exit(55)
        # same as above, but with pspace statement (convergence helper) ==================================================== #
        elif task[0]=='ci:pspace':
            pass
        # make spin orbit calculation including all given multiplicities =================================================== #
        elif task[0]=='cihlsmat':
            multcounter = 0
            stateslist = []
            for mult, states in enumerate(QMin['states']): # take states from QM.in file
                if states > QMsettings['roots'][mult]:
                    print 'ERROR: Amount of requested states for multiplicity %i is larger than number of roots in MOLCAS_setup' % (mult)
                    print 'position: cihlsmat task'
                    print 'mult, states:', mult, states
                    print 'roots of mult:', QMsettings['roots'][mult]
                    print QMin
                    print QMsettings
                    sys.exit(56)
                if states != 0:
                    multcounter += 1
                    stateslist.append(states)
                    m_in += '>> LINK $Project.%i.JobIph JOB%03i\n' % (mult+1, multcounter)
            m_in += '&RASSI\nNr  of  Job=%i' % multcounter
            for s in stateslist:
                m_in += ' %i' % s
            m_in += '\n'
            for s in stateslist:
                for i in range(1, s+1):
                    m_in += ' %i' % i
                m_in += '\n'
            m_in += 'Spin Orbit\nSOCOupling=0.0\nEJob\nmein\n'
        # make a casscf cp equation calculation, restart the orbitals, cpmcscf cards ======================================= #
        elif task[0]=='cpgrad': # task contains lists with multiplicity and requested state gradient ('cpgrad', [mult, state], [mult, state], ...)
            for grad in range(1, len(task)):
                if 'parallel_gradients' in QMsettings:
                    # setup directory for gradient calculation
                    dirname = 'parallel-gradient-%i-%i' % (task[grad][0], task[grad][1]) # name should be unique
                    dirpath = os.path.join(QMin['WorkDir'], dirname)
                    if os.path.exists(dirpath):
                        shutil.rmtree(dirpath)
                    print 'parallel_gradients requested, setting up directory: %s' % dirpath
                    os.mkdir(dirpath)
                    # write input for single gradient calculation
                    parallel_gradient_input = '' #m_in_parallel
                    parallel_gradient_input += '>> LINK $Project.%i.JobIph JOBOLD\n' % (task[grad][0])
                    parallel_gradient_input += '&RASSCF\n'
                    parallel_gradient_input += 'jobiph\n' #fileorb = $Project.RasOrb\n'
                    parallel_gradient_input += 'spin=%i\nnactel=%i 0 0\ninactive=%i\nras2=%i\nciroot=%i %i 1\n' % (task[grad][0], QMsettings['nactel'], QMsettings['inactive'],
                                                                                                QMsettings['ras2'], QMsettings['roots'][task[grad][0]-1], QMsettings['roots'][task[grad][0]-1])
                    parallel_gradient_input += 'rlxroot=%i\nCIONly\n&MCLR\nTHREshold=1.0e-4\n&ALASKA\n' % (task[grad][1]) # 1e-4 is the default mclr threshold of MOLCAS
                    gradfile = open(os.path.join(dirpath, 'molcasgrad.inp'), 'w')
                    gradfile.write(parallel_gradient_input)
                    gradfile.close()
                else:
                    # create one RASSCF calculation for every gradient in the current file
                    m_in += '>> LINK $Project.%i.JobIph JOBOLD\n' % (task[grad][0])
                    m_in += '&RASSCF\n'
                    m_in += 'jobiph\n' #fileorb = $Project.RasOrb\n'
                    m_in += 'spin=%i\nnactel=%i 0 0\ninactive=%i\nras2=%i\nciroot=%i %i 1\nCIONly\n' % (task[grad][0], QMsettings['nactel'], QMsettings['inactive'],
                                                                                                QMsettings['ras2'], QMsettings['roots'][task[grad][0]-1], QMsettings['roots'][task[grad][0]-1])
                if QMsettings['roots'][task[grad][0]-1]>1:
                    m_in += 'rlxroot=%i\n&MCLR\n' % task[grad][1]
                m_in+='&ALASKA\n'
        # forcegrad: samc record is as above =============================================================================== #
        elif task[0]=='forcegrad':
            pass
        # citrans: transition density matrix =============================================================================== #
        elif task[0]=='citrans':
            pass
        # ddr: dm record from above + states =============================================================================== #
        elif task[0]=='ddr':
            pass
        # cpnac: make a cpmcscf nacme calculation ========================================================================== #
        elif task[0]=='cpnac':
            pass
        # forcenac: evaluate the cp nacme ================================================================================== #
        elif task[0]=='forcenac':
            pass
        # casdiab: diabatize orbitals ====================================================================================== #
        elif task[0]=='casdiab':
            pass
        # cidiab: transition density matrices ============================================================================== #
        elif task[0]=='cidiab':
            pass
        elif task[0]=='dm':
            ddrdiab = False
            for t in tasks:
                if t[0]=='ddrdiab':
                    ddrdiab = True
            if ddrdiab:
                if DEBUG:
                    print 'TEST: DM task will not be setup since there already is a overlap calculation'
                continue
            # create input for DM calculation for each multiplicity
            # check how many different spins are requested
            multlist = []
            for i, states in enumerate(QMin['states']): # take states from QM.in file
                if states > QMsettings['roots'][i]:
                    print 'ERROR: Amount of requested states for multiplicity %i is larger than number of roots in MOLCAS_setup' % (i)
                    print 'position: dm task'
                    print 'mult, states:', i, states
                    print 'roots of mult:', QMsettings['roots'][i]
                    print QMin
                    print QMsettings
                    sys.exit(57)
            #for i, states in enumerate(QMsettings['roots']):
                if states == 0:
                    continue
                    #multlist.append(i)
                # create a RASSI calculation for the mu
                #mult = i + 1 #task[1] - 1
                m_in += '>> LINK $Project.%i.JobIph JOB001\n>> LINK $Project.%i.JobIph JOB002\n&RASSI\nNr  of  Job=2' % (i+1, i+1)
                #states = QMsettings['states'][i]
                m_in += ' %i %i\n' % (states, states)
                for j in range(2):
                    for k in range(1, states+1):
                        m_in += ' %i' % k
                    m_in += '\n'
                m_in += 'overlaps\nonel\nmein\n'
 
        # ddrdiab: overlap matrices ======================================================================================== #
        elif task[0]=='ddrdiab':
            # create input for non-adiabatic couplings for each multiplicity
            # check how many different spins are requested
            multlist = []
            for i, states in enumerate(QMin['states']): # take states from QM.in file
                if states > QMsettings['roots'][i]:
                    print 'ERROR: Amount of requested states for multiplicity %i is larger than number of roots in MOLCAS_setup' % (i)
                    print 'position: ddrdiab task'
                    print 'mult, states:', i, states
                    print 'roots of mult:', QMsettings['roots'][i]
                    print QMin
                    print QMsettings
                    sys.exit(58)
            #for i, states in enumerate(QMsettings['roots']):
                if states == 0:
                    continue
                    #multlist.append(i)
                # create a RASSI calculation for the mu
                #mult = i + 1 #task[1] - 1
                m_in += '>> LINK $Project.%i.JobIph.old JOB001\n>> LINK $Project.%i.JobIph JOB002\n&RASSI\nNr  of  Job=2' % (i+1, i+1)
                #states = QMsettings['roots'][i]
                m_in += ' %i %i\n' % (states, states)
                for j in range(2):
                    for k in range(1, states+1):
                        m_in += ' %i' % k
                    m_in += '\n'
                m_in += 'overlaps\nonel\nmein\n'
        else: # ============================================================================================================ #
            print 'Unknown task keyword %s found in writeMOLCASinput!' % task[0]
            print task, task[0]
            if task[0] == 'mcscf':
                print 'bla'
            print m_in
            sys.exit(59)
    inp=open('MOLCAS.inp','w')
    inp.write(m_in)
    inp.close()
    return

# ======================================================================= #
def runMOLCAS(QMin):
    '''Calls MOLCAS in a shell with the SCRATCHDIR directory as integral directory. 

    Arguments:
    1 dictionary: QMin

    Returns:
    1 integer: MOLCAS exit code'''

    #string='%s MOLCAS.inp -W%s -I%s -d%s' % (QMin['qmexe'],QMin['pwd'],QMin['scratchdir'],QMin['scratchdir'])
    os.chdir(QMin['WorkDir'])
    string = 'molcas MOLCAS.inp > MOLCAS.out 2> MOLCAS.err'
    #string = 'ls' # TEST-PHASE
    if PRINT:
        print datetime.datetime.now()
        print '===> Running MOLCAS:\n\n%s\n\nError Code:' % (string)
        sys.stdout.flush()
    try:
        runerror=sp.call(string,shell=True) # TODO: Why is the shell necessary here?
        if PRINT:
            print '%s\n\n' % (runerror)
    except OSError:
        print 'MOLCAS call have had some serious problems:',OSError
        sys.exit(60)
    # that was the first run of MOLCAS. if parallel gradients have been requested, now the gradients should be calculated
    if 'parallel_gradients' in QMin:
        runMOLCAS_parallel_gradients(QMin)
    return runerror

# ======================================================================= #

def runMOLCAS_PoolWorker(QMin):
    os.environ['WorkDir'] = QMin['WorkDir'] # set WorkDir to local gradient calculation directory
    os.chdir(QMin['WorkDir'])
    string = '%s molcasgrad.inp > molcasgrad.out' % (QMin['qmexe'])
    #string = 'ls' # TEST-PHASE
    try:
        while True:
            if PRINT:
                print datetime.datetime.now()
                print '===> Running MOLCAS job: %s\n%s' % (QMin['WorkDir'],string)
                sys.stdout.flush()
            runerror=sp.call(string,shell=True) # TODO: Why is the shell necessary here?
            if PRINT:
                print 'Error Code for %s: %s' % (QMin['WorkDir'], runerror)
            outfile = open(os.path.join(QMin['WorkDir'], 'molcasgrad.out'), 'r')
            content = outfile.readlines()
            outfile.close()
            rerun = False
            #print rerun
            for line in content:
                if line.find('Convergence problem!') != -1:
                    rerun = True
                    #print rerun
            if rerun:
                print 'Noticed convergence problem.'
                infile = open(os.path.join(QMin['WorkDir'], 'molcasgrad.inp'), 'r')
                content = infile.readlines()
                infile.close()
                for i, line in enumerate(content):
                    if line.find('THREshold') != -1:
                        thresh = float(line.split('=')[1].strip())
                        if thresh >= 1.:
                            print 'Threshold for MCLR is already %f. Will abort and not increase it further!' % thresh
                        else:
                            print 'Increasing MCLR threshold from %f to %f' % (thresh, thresh*10.)
                            thresh *= 10.
                            content[i] = 'THREshold=%f\n' % thresh
                            #break
                outfile = open(os.path.join(QMin['WorkDir'], 'molcasgrad.inp'), 'w')
                outfile.write(''.join(content))
                outfile.close()
                continue # goto ;) beginning of while True cycle and call MOLCAS again
            else:
                break # break the while True cycle if you reach this point, i.e. you dont restart because of errorcode 96
    except OSError:
        print QMin['WorkDir'], ' MOLCAS call has had some serious problems:',OSError
        sys.exit(61)
 
# ======================================================================= #

def runMOLCAS_parallel_gradients(QMin):
    # get list of parallel gradient calculations
    print 'Setting up parallel_gradients calculations'
    joblist = []
    for element in os.listdir(QMin['WorkDir']):
        if element.startswith('parallel-gradient-') and os.path.isdir(os.path.join(QMin['WorkDir'], element)):
            joblist.append(element)
    # copy files from preceeding MOLCAS run (jobiph, geom.xyz)
    for job in joblist:
        os.link(os.path.join(QMin['WorkDir'], 'geom.xyz'), os.path.join(QMin['WorkDir'], job, 'geom.xyz'))
        for element in os.listdir(QMin['WorkDir']):
            if os.path.isfile(os.path.join(QMin['WorkDir'], element)) and ( element.endswith('OrdInt') or element.endswith('JobIph') or element.endswith('OneInt') or element.endswith('OneRel')): #element.endswith('.JobIph'):
                print 'linking file %s to %s' % (element, os.path.join(QMin['WorkDir'], job))
                os.link(os.path.join(QMin['WorkDir'], element), os.path.join(QMin['WorkDir'], job, element))
            if os.path.isfile(os.path.join(QMin['WorkDir'], element)) and ( element.endswith('RunFile') ): #element.endswith('.JobIph'):
                print 'copying file %s to %s' % (element, os.path.join(QMin['WorkDir'], job))
                shutil.copy(os.path.join(QMin['WorkDir'], element), os.path.join(QMin['WorkDir'], job))
    # setup worker pool
    pool = Pool(processes=QMin['parallel_gradients'])
    # start jobs
    for job in joblist:
        tmpQMin = deepcopy(QMin)
        tmpQMin['WorkDir'] = os.path.join(QMin['WorkDir'], job)
        pool.apply_async(runMOLCAS_PoolWorker , [tmpQMin])
    # wait for jobs to finish
    pool.close()
    pool.join()
    # attach output files to main output file of MOLCAS
    # get content of main output file
    os.chdir(QMin['WorkDir'])
    molfile = open(os.path.join(QMin['WorkDir'], 'MOLCAS.out'), 'r')
    content = molfile.readlines()
    molfile.close()
    # attach gradient outputs
    for job in joblist:
        gradfile = open(os.path.join(QMin['WorkDir'], job, 'molcasgrad.out'))
        content.extend(gradfile.readlines())
        gradfile.close()
    # write it all to the old MOLCAS.out file
    molfile = open(os.path.join(QMin['WorkDir'], 'MOLCAS.out'), 'w')
    molfile.write(''.join(content))
    molfile.close()
    # delete gradient directories
    return # for debug purposes, remove or comment this line to delete gradient directories after finishing calculations
    for element in os.listdir(QMin['WorkDir']):
        if element.startswith('parallel-gradient-') and os.path.isdir(os.path.join(QMin['WorkDir'], element)):
            shutil.rmtree(os.path.join(QMin['WorkDir'], element))

# ======================================================================= #
def redotasks(tasks,QMin):
    '''Screens the MOLCAS output file for error messages and reconstructs the tasks list. The new list contains all remaining tasks which have not been accomplished. The task which caused the crash is redone with altered parameters to ensure convergence.

    Currently, the script can deal with the following MOLCAS errors:
    - EXCESSIVE GRADIENT IN CI:
                    This error occurs sometimes if the initial guess for the CI vectors in the MCSCF calculation is bad. Usually, this can be dealt with by including more CSFs in the primary configuration space. 
                    If this error occurs, the script will restart MOLCAS with a pspace threshold of 1. If this does not lead to success, the threshold is increased further. If the calculation still crashes with a threshold of 9, the script returns with exit code 1.
    - NO CONVERGENCE IN CPMCSCF:
                    This error in the calculation of gradients and non-adiabatic coupling vectors occurs if the active space contains strongly doubly occupied/empty orbitals and the associated orbital rotation gradients are very small.
                    If this error occurs, the corresponding calculation is started with a looser convergence criterium. How the criterium is altered can be changed using environment variables GRADACCUDEFAULT, GRADACCUMAX, GRADACCUSTEP

    Arguments:
    1 list of lists: task list
    2 dictionary: QMin

    Returns:
    1 list of lists: new task list'''

    newtasks=[]
    outfile=open('MOLCAS.out','r')
    out=outfile.readlines()
    outfile.close()
    if not 'Happy landing' in out[-3] and not 'Happy landing' in out[-4]:
    #if out[-3].find('Happy landing!') != -1 or out[-4].find('Happy landing!') != -1:
        # something went wrong
        newtasks = tasks
        print 'Did not find "Happy Landing"'
        sys.exit(62)
    return newtasks

# ======================================================================= #
def catMOLCASoutput(outcounter):
    '''Reads all MOLCAS output files from the current time step and concatenates them for the extraction of the requested quantities.

    Arguments:
    1 integer: number of output files

    Returns:
    1 list of strings: Concatenation of all MOLCAS output files'''

    if PRINT:
        print '===> Processing output from:\n'
    out=[]
    for i in range(outcounter):
        if PRINT:
            print 'MOLCAS%04i.out' % (i+1)
        outfile=open('MOLCAS%04i.out' % (i+1),'r')
        out.extend(outfile.readlines())
        outfile.close()
    print '\n'
    return out

# ======================================================================= #
def getQMout(out,QMin):
    '''Constructs the requested matrices and vectors using the get<quantity> routines.

    The dictionary QMout contains all the requested properties. Its content is dependent on the keywords in QMin:
    - 'h' in QMin:
                    QMout['h']: list(nmstates) of list(nmstates) of complex, the non-relaticistic hamiltonian
    - 'soc' in QMin:
                    QMout['h']: list(nmstates) of list(nmstates) of complex, the spin-orbit hamiltonian
    - 'dm' in QMin:
                    QMout['dm']: list(3) of list(nmstates) of list(nmstates) of complex, the three dipole moment matrices
    - 'grad' in QMin:
                    QMout['grad']: list(nmstates) of list(natom) of list(3) of float, the gradient vectors of every state (even if "grad all" was not requested, all nmstates gradients are contained here)
    - 'nac' in QMin and QMin['nac']==['num']:
                    QMout['nac']: list(nmstates) of list(nmstates) of complex, the non-adiabatic coupling matrix
                    QMout['mrcioverlap']: list(nmstates) of list(nmstates) of complex, the MRCI overlap matrix
                    QMout['h']: like with QMin['h']
    - 'nac' in QMin and QMin['nac']==['ana']:
                    QMout['nac']: list(nmstates) of list(nmstates) of list(natom) of list(3) of float, the matrix of coupling vectors
    - 'nac' in QMin and QMin['nac']==['smat']:
                    QMout['nac']: list(nmstates) of list(nmstates) of complex, the adiabatic-diabatic transformation matrix
                    QMout['mrcioverlap']: list(nmstates) of list(nmstates) of complex, the MRCI overlap matrix
                    QMout['h']: like with QMin['h']

    Arguments:
    1 list of strings: Concatenated MOLCAS output
    2 dictionary: QMin

    Returns:
    1 dictionary: QMout'''


    # get version of MOLCAS
    QMin['version']=getversion(out)

    # Currently implemented keywords: h, soc, dm, grad, nac (num,ana,smat)
    states=QMin['states']
    nstates=QMin['nstates']
    nmstates=QMin['nmstates']
    natom=QMin['natom']
    QMout={}
    # h: get CI energies of all ci calculations and construct hamiltonian, returns a matrix(nmstates,nmstates)
    if 'h' in QMin or 'nacdt' in QMin or 'overlap' in QMin:
        # no spin-orbit couplings, hamilton operator diagonal, only one loop
        h=makecmatrix(nmstates,nmstates)
        for istate in range(nmstates):
            mult,state=IstateToMultState(istate+1,states)
            h[istate][istate]=complex(getcienergy(out,mult,state,QMin['version']))
        QMout['h']=h
    # SOC: get SOC matrix and construct hamiltonian, returns a matrix(nmstates,nmstates)
    if 'soc' in QMin:
        # soc: matrix is not diagonal, two nested loop
        soc=makecmatrix(nmstates,nmstates)
        for istate in range(nmstates):
            for jstate in range(nmstates):
                mult1, state1, ms1 = IToMultStateMS(istate+1, states)
                mult2, state2, ms2 = IToMultStateMS(jstate+1, states)
                soc[istate][jstate]=getsocme(out, mult1, state1, ms1, mult2, state2, ms2 ,states,QMin['version'])
        QMout['h']=soc
    # DM: get vector of three dipole matrices, three nested loops, returns a list of three matrices(nmstates,nmstates)
    if 'dm' in QMin:
        dm=[]
        for xyz in range(3):
            dm.append(makecmatrix(nmstates,nmstates))
            for mult,state1,state2 in ittwostatesfull(states):
                for istate,jstate in MultStateToIstateJstate(mult,state1,state2,states):
                    dm[xyz][istate-1][jstate-1]=complex(getcidm(out,mult,state1,state2,xyz,QMin['version']))
        QMout['dm']=dm
    # Grad: for argument all single loop, otherwise a bit more complex, returns a list of nmstates vectors
    if 'grad' in QMin:
        grad=[]
        if QMin['grad']==['all']:
            for istate in range(nmstates):
                mult,state=IstateToMultState(istate+1,states)
                grad.append(getgrad(out,mult,state,natom,QMin))
        else:
            for istate in range(nmstates):
                gradatom=[]
                for iatom in range(natom):
                    gradatom.append([0.,0.,0.])
                grad.append(gradatom)
            for iarg in range(len(QMin['grad'])):
                mult,state=IstateToMultState(QMin['grad'][iarg],states)
                for istate in MultStateToIstate(mult,state,states):
                    if 'unit' in QMin: # if unit is defined
                        if QMin['unit'] == 'angstrom': # and is angstrom
                            grad[istate-1]=getgrad(out,mult,state,natom,QMin) / B2Ang # convert from angstrom to bohr
                        else:
                            grad[istate-1]=getgrad(out,mult,state,natom,QMin) # dont do it if unit is not angstrom
                    else:
                        grad[istate-1]=getgrad(out,mult,state,natom,QMin) # or is not defined (bohr is assumed)
        QMout['grad']=grad
    # NAC: case of keyword "num": returns a matrix(nmstates,nmstates)
    # and also collects the mrci overlaps for later error evaluation
    if 'nacdt' in QMin:
        pass
#        nac=makecmatrix(nmstates,nmstates)
#        mrcioverlap=makermatrix(nmstates,nmstates)
#        for mult,state1,state2 in ittwostatesfull(states):
#            for istate,jstate in MultStateToIstateJstate(mult,state1,state2,states):
#                nac[istate-1][jstate-1]=complex(getnacnum(out,mult,state1,state2))
#                mrcioverlap[istate-1][jstate-1]=getmrcioverlap(out,mult,state1,state2)
#        QMout['nacdt']=nac
#        QMout['mrcioverlap']=mrcioverlap
    # NAC: case of keyword "ana": returns a matrix(nmstates,nmstates) of vectors
    if 'nacdr' in QMin:
        pass
#        grad=[]
#        for i in range(natom):
#            grad.append([0.,0.,0.])
#        nac=[ [ grad for i in range(nmstates) ] for j in range(nmstates) ]
#        if len(QMin['nacdr'])==2 and QMin['nacdr'][0]=='select':
#            nacpairs=QMin['nacdr'][1]
#            for i in range(len(nacpairs)):
#                m1,i1=IstateToMultState(nacpairs[i][0],states)
#                m2,i2=IstateToMultState(nacpairs[i][1],states)
#                if m1==m2:
#                    for istate,jstate in MultStateToIstateJstate(m1,i1,i2,states):
#                        nac[istate-1][jstate-1]=getnacana(out,m1,i1,i2,natom)
#                m1,i1=IstateToMultState(nacpairs[i][1],states)
#                m2,i2=IstateToMultState(nacpairs[i][0],states)
#                if m1==m2:
#                    for istate,jstate in MultStateToIstateJstate(m1,i1,i2,states):
#                        nac[istate-1][jstate-1]=getnacana(out,m1,i1,i2,natom)
#        else:
#            for mult,state1,state2 in ittwostatesfull(states):
#                for istate,jstate in MultStateToIstateJstate(mult,state1,state2,states):
#                    nac[istate-1][jstate-1]=getnacana(out,mult,state1,state2,natom)
#        QMout['nacdr']=nac
    # NAC: case of keyword "smat": returns a matrix(nmstates,nmstates)
    if 'overlap' in QMin:
        nac=makecmatrix(nmstates,nmstates)
        mrcioverlap=makermatrix(nmstates,nmstates)
        for mult,state1,state2 in ittwostatesfull(states):
            for istate,jstate in MultStateToIstateJstate(mult,state1,state2,states):
                nac[istate-1][jstate-1]=complex(getsmate(out,mult,state1,state2,states))
                mrcioverlap[istate-1][jstate-1]=getmrcioverlap(out,mult,state1,state2)
        QMout['overlap']=nac
        QMout['mrcioverlap']=nac #mrcioverlap TODO: correct? but didnt find anything else in molcas output.
    # NAC: case of numfromana
    if 'nacdtfromdr' in QMin:
        pass
#        grad=[]
#        for i in range(natom):
#            grad.append([0.,0.,0.])
#        nac=[ [ grad for i in range(nmstates) ] for j in range(nmstates) ]
#        for mult,state1,state2 in ittwostatesfull(states):
#            for istate,jstate in MultStateToIstateJstate(mult,state1,state2,states):
#                nac[istate-1][jstate-1]=getnacana(out,mult,state1,state2,natom)
#        QMout['nacdt']=nac
    if 'angular' in QMin:
        pass
#        ang=[]
#        for xyz in range(3):
#            ang.append(makecmatrix(nmstates,nmstates))
#            for mult,state1,state2 in ittwostatesfull(states):
#                for istate,jstate in MultStateToIstateJstate(mult,state1,state2,states):
#                    ang[xyz][istate-1][jstate-1]=complex(getciang(out,mult,state1,state2,xyz))
#        QMout['angular']=ang
    return QMout

# ======================================================================= #
#def mrcioverlapsok(QMin,QMout):
    #'''Checks for all diagonal elements of the MRCI overlaps whether their absolute value is above the relevant threshold.

    #Arguments:
    #1 dictionary: QMin
    #2 dictionary: QMout

    #Returns:
    #1 Boolean'''

    #ok=True
    #mrcioverlap=QMout['mrcioverlap']
    #states=QMin['states']
    #nmstates=QMin['nmstates']
    #h=QMout['h']
    #for istate in range(nmstates):
        #if abs(mrcioverlap[istate][istate])<QMin['CHECKNACS_MRCIO']:
            #ok=False
    #return ok

# ======================================================================= #
#def setnacszero(QMin,QMout):
    #'''Sets non-adiabatic coupling elements to zero, if there corresponding MRCI overlaps are bad and the two coupled states are too far separated.

    #Arguments:
    #1 dictionary: QMin
    #2 dictionary: QMout

    #Returns:
    #1 list of list of complex: nac matrix'''

    #if PRINT:
        #print '===> Checking non-adiabatic couplings:\n'
    #mrcioverlap=QMout['mrcioverlap']
    #states=QMin['states']
    #nmstates=QMin['nmstates']
    #h=QMout['h']
    #nac=QMout['nacdt']
    #for istate in range(nmstates):
        #if abs(mrcioverlap[istate][istate])<QMin['CHECKNACS_MRCIO']:
            #if PRINT:
                #print '=> MRCI overlap of state \t%i is bad:' % (istate)
            #for jstate in range(nmstates):
                #if abs(h[istate][istate]-h[jstate][jstate])>QMin['CHECKNACS_EDIFF']:
                    #nac[istate][jstate]=complex(0.)
                    #nac[jstate][istate]=complex(0.)
                    #if PRINT:
                        #print '- setting nac[\t%i][\t%i]=-nac[\t%i][\t%i]=0.' % (istate+1,jstate+1,jstate+1,istate+1)
    #return nac

# ======================================================================= #
#def redoNacjob(QMin):
    #'''Plans a completely new calculation to obtain CPMCSCF non-adiabatic couplings, in the case that the numerical couplings are corrupted.

    #Arguments:
    #1 dictionary: QMin

    #Returns:
    #1 dictionary: a new QMin, which requests nac ana'''

    #QMin2={}
    ## needed: geo, gradaccu..., natom, nstates, nmstates, states, pwd, qmexe, scratchdir, unit
    #necessary=['comment','geo','gradaccudefault','gradaccumax','gradaccustep',
                         #'natom','nmstates','nstates','pwd','qmexe','scratchdir','unit','states']
    #for i in necessary:
        #QMin2[i]=QMin[i]
    ## only task: analytical couplings
    #QMin2['restart']=[]
    #QMin2['nacdr']=[]
    #return QMin2

## ======================================================================= #
#def contractNACveloc(QMin,QMout,QMout2):
    #'''Contracts the matrix of vectorial non-adiabatic couplings with the velocity vector to obtain < i | d/dt| j >.

    #< i | d/dt| j > = sum_atom sum_cart v_atom_cart * < i | d/dR_atom_cart| j >

    #Arguments:
    #1 dictionary: QMin, containing veloc
    #2 dictionary: the QMout containing soc, dm, grad
    #3 dictionary: the new QMout with the analytical couplings

    #Returns:
    #1 dictionary: QMout, including everything from the old QMout and the new NAC matrix'''

    ## calculates the scalar product of the analytical couplings and the velocity,
    ## and puts the resulting matrix into QMout
    #veloc=QMin['veloc']
    #nmstates=QMin['nmstates']
    #natom=QMin['natom']
    #nacdr=QMout2['nacdr']
    #nacdt=makecmatrix(nmstates,nmstates)
    #for istate in range(nmstates):
        #for jstate in range(nmstates):
            #scal=complex(0.)
            #for iatom in range(natom):
                #for ixyz in range(3):
                    #scal+=veloc[iatom][ixyz]*nacdr[istate][jstate][iatom][ixyz]
            #nacdt[istate][jstate]=scal
    #QMout['nacdt']=nacdt
    #return QMout

# ======================================================================= #
def getphases(QMin,QMout):
    '''Obtains the wavefunction phases from the MRCI vector overlaps. The phases are passed on to SHARC to retain a consistent wavefunction phase in the dynamics.

    Arguments:
    1 dictionary: QMin
    2 dictionary: QMout, containing mrcioverlaps

    Returns:
    1 list of float: wavefunction phases'''

    # The phases are sign(mrcioverlap)
    if 'mrcioverlap' in QMout:
        mrcioverlap=QMout['mrcioverlap']
        nmstates=QMin['nmstates']
        phases=[]
        for i in range(nmstates):
            if mrcioverlap[i][i]>=0:
                phases.append(1.)
            else:
                phases.append(-1.)
        return phases
    else:
        nmstates=QMin['nmstates']
        phases=[]
        for i in range(nmstates):
            phases.append(1.)
        return phases

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
    if 'angular' in QMin:
        string+=writeQMoutang(QMin,QMout)
    if 'grad' in QMin:
        string+=writeQMoutgrad(QMin,QMout)
    if 'nacdt' in QMin:
        string+=writeQMoutnacnum(QMin,QMout)
    if 'nacdr' in QMin:
        string+=writeQMoutnacana(QMin,QMout)
    if 'overlap' in QMin:
        string+=writeQMoutnacsmat(QMin,QMout)
    if 'nacdtfromdr' in QMin:
        string+=writeQMoutnacnum(QMin,QMout)
    string+=writeQMouttime(QMin,QMout)
    try:
        outfile=open(os.path.join(QMin['SHQM'],outfilename),'w')
        outfile.write(string)
        outfile.close()
    except IOError:
        print 'Could not write QM output!'
        sys.exit(63)
    return

# ======================================================================= #
def cycleMOLCAS(QMin,Tasks):
    '''Iteratively writes MOLCAS input, calls MOLCAS (via runMOLCAS) and redoes the tasks list until the tasks list is empty. Renames the MOLCAS output files after each run. 

    Arguments:
    1 dictionary: QMin
    2 list of lists: task list

    Returns:
    1 integer: number of MOLCAS output files'''

    # Loop: write molpro input, run molpro, read molpro output, decide: ready or rewrite the Tasks array
    # Run until no jobs other than a bare restart are necessary
    outcounter=0
    while Tasks!=[]:
        writeMOLCASinput(Tasks, QMin)
        runerror=runMOLCAS(QMin)
        if runerror!=0:
          print 'Error code is non-zero, aborting...'
          sys.exit(64)
        Tasks=redotasks(Tasks,QMin)
        printtasks(Tasks)
        outcounter+=1
        if DEBUG:
            # find MOLCAS-....out files
            filecounter = 0
            while True:
                if os.path.isfile('../MOLCAS-%s-%02i.out' % (QMin['step'][0],filecounter)):
                    filecounter += 1
                else:
                    break
            shutil.copy('MOLCAS.out','../MOLCAS-%s-%02i.out' % (QMin['step'][0],filecounter))
        os.rename('MOLCAS.out','MOLCAS%04i.out' % (outcounter))
    if runerror!=0:
        print 'MOLCAS failed with unknown error!'
        sys.exit(65)
    if PRINT:
        string='    '+'='*40+'\n'
        string+='||'+' '*40+'||\n'
        string+='||'+' '*10+'All Tasks completed!'+' '*10+'||\n'
        string+='||'+' '*40+'||\n'
        string+='    '+'='*40+'\n\n'
        print string
    return outcounter

# ======================================================================= #
#def checknac(QMin,QMout):
    #'''Checks the results from DDR calculations for correctness. Obtains uncorrupted couplings via nac ana if possible. It also obtains wavefunction phases, even if CHECKNACS is disabled.

    #In MOLCAS, the calculation of non-adiabatic couplings by means of the DDR procedure is very efficient. However, in the case of strong orbital mixing caused by intruder states this procedure yields highly incorrect values without any error message. In this routine, an intruder state is detected by means of the MRCI overlaps and the problem probably solved by calculating the couplings analytically. To this end, a new QMin dictionary is created, specifying the calculation of these couplings. After the calculation is finished, the coupling matrix is obtained from the scalar product of the velocity and the vector couplings.

    #This check is engaged via the environment variable CHECKNACS. It only checks the results for "nac num" and "nac smat". In the former case, a correct matrix is constructed analytically, if velocities are availible, in the latter case an error message is printed and the dynamics aborted.

    #Arguments:
    #1 dictionary: QMin
    #2 dictionary: QMout

    #Returns:
    #1 dictionary: QMout (including 'phases' and possibly with a corrected 'nac' matrix)'''

    ##only if NACS are to be checked
    #if QMin['CHECKNACS']:
        ## in the case of numeric couplings, which are bad
        #if 'nacdt' in QMin and not mrcioverlapsok(QMin,QMout):
            #print 'MRCI overlaps seem to be bad. Most probably an intruder state messed up the active space...'
            #if QMin['CORRECTNACS']:
                #if 'veloc' in QMin:
                    #print 'Trying to obtain non-corrupted non-adiabatic couplings from "nac ana"...\n'
                    ## Generate a new QMin dictionary containing the new job, set up the tasks
                    #QMin_redoNac=redoNacjob(QMin)
                    #Tasks_redoNac=gettasks(QMin_redoNac)
                    #printtasks(Tasks_redoNac)
                    ## Run Molpro with this job until success
                    #outcounter=cycleMOLCAS(QMin_redoNac,Tasks_redoNac)
                    ## Extract analytical non-adiabatic couplings
                    #out_redoNac=catMOLCASoutput(outcounter)
                    #QMout_redoNac=getQMout(out_redoNac,QMin_redoNac)
                    ## Build the d/dt matrix from the couplings and the velocities
                    #QMout=contractNACveloc(QMin,QMout,QMout_redoNac)
                #else:
                    #print 'No velocities availible. Aborting the dynamics because of corrupted non-adiabatic couplings!\n'
                    #sys.exit(66)
            #else:
                #print 'Screening couplings for bad values and set these to zero...\n'
                #QMout['nacdt']=setnacszero(QMin,QMout)
        #if 'overlap' in QMin and not mrcioverlapsok(QMin,QMout):
            #print 'MRCI overlaps seem to be bad. Most probably an intruder state messed up the active space...'
            #print 'Aborting the dynamics because of corrupted overlap matrix!\n'
            #sys.exit(67)
    ## finally, obtain the wavefunction phases from the mrcioverlaps
    #if 'nacdt' in QMin:
        #QMout['phases']=getphases(QMin,QMout)
        ## "overcorrect" the NACs, so that SHARC can correct the phase
        #for i in range(QMin['nmstates']):
            #for j in range(QMin['nmstates']):
                #QMout['nacdt'][i][j]/=(QMout['phases'][i]*QMout['phases'][j])
    #return QMout

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

# ======================================================================= #
def saveJobIphs(QMin):
    '''Saves all JobIph files that are named like $Project.%i.JobIph within $WorkDir in the $SHQM directory and renames them to ".old".'''
    for element in os.listdir(QMin['WorkDir']):
        if element.endswith('.JobIph') and element.split('.')[-2].isdigit():
            shutil.copyfile(os.path.join(QMin['WorkDir'], element), os.path.join(QMin['SHQM'],element+'.old'))

# ======================================================================= #
#def get_environment_variable_from_template(path, name):
    #'''Tries to get the value of an environment variable "name" from the file given in path.
#Command in file is: set variable name to VALUE
#Returns a string holding the value of the variable.'''
    #try:
        #templatefile=open(path,'r')
    #except IOError:
        #print 'Need to search for variable %s in MOLCAS setup file!\nCould not find it at %s' % (name,path)
        #raise
        #sys.exit(68)
    #template=templatefile.readlines()
    #templatefile.close()
    #for line in template:
        #if line.lower().find('set variable') != -1 and line.split()[2] == name:
            #return line.split()[4].strip()
    #return None

# ========================== Main Code =============================== #
def main():
    '''This script realises an interface between the semi-classical dynamics code SHARC and the quantum chemistry program MOLCAS 2012. It allows the automatised calculation of non-relativistic and spin-orbit Hamiltonians, Dipole moments, gradients and non-adiabatic couplings at the CASSCF level of theory for an arbitrary number of states of different multiplicities. It also includes a small number of MOLCAS error handling capabilities (restarting non-converged calculations etc.).

    Input is realised through two files and a number of environment variables.

    QM.in:
        This file contains all information which are known to SHARC and which are independent of the used quantum chemistry code. This includes the current geometry and velocity, the number of states/multiplicities, the time step and the kind of quantities to be calculated.

    MOLCAS.template:
        This file is a minimal MOLCAS input file containing all molecule-specific parameters, like memory requirement, basis set, Douglas-Kroll-Hess transformation, active space and state-averaging. 

    Environment variables:
        Additional information, which are necessary to run MOLCAS, but which do not actually belong in a MOLCAS input file. 
        The necessary variables are:
            * QMEXE: is the path to the MOLCAS executable
            * SCRATCHDIR: is the path to a scratch directory for fast I/O Operations.
        Some optional variables are concerned with MOLCAS error handling (defaults in parenthesis):
            * GRADACCUDEFAULT: default accuracy for MOLCAS CPMCSCF (1e-7)
            * GRADACCUMAX: loosest allowed accuracy for MOLCAS CPMCSCF (1e-2)
            * GRADACCUSTEP: factor for decreasing the accuracy for MOLCAS CPMCSCF (1e-1)
            * CHECKNACS: check whether non-adiabatic couplings are corrupted by intruder states (False)
            * CHECKNACS_MRCIO: threshold for intruder state detection, see mrcioverlapsok() (0.85)'''

    # Retrieve PRINT and DEBUG
    try:
        envPRINT=os.getenv('SH2PRO_PRINT')
        if envPRINT and envPRINT.lower()=='false':
            global PRINT
            PRINT=False
        envDEBUG=os.getenv('SH2PRO_DEBUG')
        if envDEBUG and envDEBUG.lower()=='true':
            global DEBUG
            DEBUG=True
    except ValueError:
        print 'PRINT or DEBUG environment variables do not evaluate to numerical values!'
        sys.exit(69)

    # Process Command line arguments
    if len(sys.argv)!=2:
        print 'Usage:\n./SHARC_MOLCAS.py <QMin>\n'
        print 'version:',version
        print 'date:',versiondate
        print 'changelog:\n',changelogstring
        sys.exit(70)
    QMinfilename=sys.argv[1]

    # Print header
    printheader()

    # Read QMinfile
    QMin=readQMin(QMinfilename)
    printQMin(QMin)

    # Process Tasks
    Tasks=gettasks(QMin)
    printtasks(Tasks)

    # Run MOLCAS until all jobs are done
    outcounter=cycleMOLCAS(QMin,Tasks)

    # Parse MOLCAS Output
    out=catMOLCASoutput(outcounter)
    QMout=getQMout(out,QMin)
    printQMout(QMin,QMout)



    # Measure time
    runtime=measuretime()
    QMout['runtime']=runtime

    # Write QMout
    writeQMout(QMin,QMout,QMinfilename)
    # save JobIphs to SHQM for later use
    saveJobIphs(QMin)

    # Remove Scratchfiles from SCRATCHDIR
    if not DEBUG:
        cleanupSCRATCH(QMin['WorkDir'])
    if PRINT or DEBUG:
        print '#================ END ================#'

if __name__ == '__main__':
    main()
