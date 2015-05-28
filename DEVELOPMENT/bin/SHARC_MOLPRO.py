#!/usr/bin/env python2

#  ====================================================================
#||                                                                    ||
#||                             General Remarks                        ||
#||                                                                    ||
#  ====================================================================
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
#   for MS in range(mult):
#     for state in range(states[mult]):
#       i+=1
#       print i, mult+1, state+1, MS-i/2
#
# more about this below in the docstrings of the iterator functions

# ======================================================================= #

# IMPLEMENTATION OF ADDITIONAL TASKS KEYWORDS, JOBS, ETC:
#
# A new task keyword in QMin has to be added to:
#       - readQMin (for consistency check)
#       - gettasks (planning the MOLPRO calculation)
#       - print QMin (optional)
#
# A new task in the Tasks array needs changes in:
#       - gettasks 
#       - writeMOLPROinput 
#       - redotasks
#       - printtasks

# ======================================================================= #
# Modules:
# Operating system, isfile and related routines, move files, create directories
import os
# External Calls to MOLPRO
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
from socket import gethostname

# ======================================================================= #

version='1.0'
versiondate=datetime.date(2014,10,8)


changelogstring='''
06.02.:
- changed the records for cpmcscf to 5xxx.9

07.02.:
- added angular keyword (angular momentum evaulation and extraction)
- extraction of best obtained accuracy in cpmcscf
- nogprint,orbitals,civectors added in MOLPRO input

08.02.:
- added removal of SCRATCHDIR after job finished successfully
- added expansion of environment variables and ~ in paths

13.03.:
- added facility for selective analytical NACs
- added input for nac ana select
- added environment variables for PRINT and DEBUG

08.05.:
- added CI:pspace task

22.05.:
- Smat is written transposed now (to be compatible with Fortran)

06.06.:
- MCSCF convergence thresholds increased by default, to help cpmcscf convergence

11.10.:
- changed keyword "nac" to "nacdr", "nacdt" and "overlap"
=>NOT COMPATIBLE WITH OLDER VERSIONS OF SHARC!

19.02.2014:
- modified qmout write routines to be compatible with the new SHARC

11.03.2014:
- changed keyword "restart" to "samestep" to avoid ambiguity
=>NOT COMPATIBLE WITH OLDER VERSIONS OF SHARC!

11.06.2014:
- "grad" can now have no arguments. "grad" is equivalent to "grad all"

16.07.2014:
- savedir from QM.in or SH2PRO.inp

22.09.2014:
- improved calculation of overlap matrices

08.10.2014:     1.0
- official release version, no changes to 0.2

18.12.2014:
- fixed a bug where CPMCSCF solutions converging in zero iterations (full CI case) are not treated properly
- fixed a bug where gradients are not read out if "grad" is given without specifying "all" or the states

20.01.2015:
- command line options for MOLPRO are now read from SH2PRO.inp and passed to MOLPRO, allowing e.g. parallel execution.'''

# ======================================================================= #
# holds the system time when the script was started
starttime=datetime.datetime.now()

# global variables for printing (PRINT gives formatted output, DEBUG gives raw output)
DEBUG=False
PRINT=True

# hash table for conversion of multiplicity to the keywords used in MOLPRO
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

# hash table for conversion of polarisations to the keywords used in MOLPRO
IToPol={
        0: 'X', 
        1: 'Y', 
        2: 'Z', 
        'X': 0, 
        'Y': 1, 
        'Z': 2
        }

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
  sys.exit(11)

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
    sys.exit(12)
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
    sys.exit(13)
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

# ======================================================================= #     OK
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

  print starttime,gethostname(),os.getcwd()
  if not PRINT:
    return
  string='\n'
  string+='  '+'='*80+'\n'
  string+='||'+' '*80+'||\n'
  string+='||'+' '*25+'SHARC - MOLPRO2012 - Interface'+' '*25+'||\n'
  string+='||'+' '*80+'||\n'
  string+='||'+' '*29+'Author: Sebastian Mai'+' '*30+'||\n'
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
    if task[0]=='samestep':
      print 'Samestep'
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
      print 'CI:\t2x Trans. Mom\tMultiplicity: %i\tState: %i' % (task[1],task[2])
    elif task[0]=='ddrdiab':
      print 'DDR:\tOverlap Matrix\tMultiplicity: %i\tState: %i' % (task[1],task[2])
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
      print '=> Wavefunction Phases:\n%i\n' % (nmstates)
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
            printgrad(QMout['nacdr'][istate][jstate],natom,QMin['geo'])
          jstate+=1
        istate+=1
  if 'overlap' in QMin:
    print '=> Overlap matrix:\n'
    matrix=QMout['overlap']
    printcomplexmatrix(matrix,states)
    if 'phases' in QMout:
      print '=> Wavefunction Phases:\n%i\n' % (nmstates)
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

# ======================================================================= #     OK
def nextblock(data,program='*',occ=1):
  '''Scans the list of strings data for the next occurence of MOLPRO program block for program. Returns the line number where the block ends and the block itself.

  Arguments:
  1 list of strings: data
  2 string: MOLPRO program name (like "MULTI" in "1PROGRAM * MULTI" )
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
      print 'Block %s not found in routine nextblock! Probably MOLPRO encountered an error not anticipated in this script. Check the MOLPRO output!' % (program)
      sys.exit(14)
  progdata.append(data[i])
  i+=1
  while i<len(data) and not containsstring('1PROGRAM',data[i]):
    progdata.append(data[i])
    i+=1
  return i,progdata

# ======================================================================= #     OK
def makecmatrix(a,b):
  '''Initialises a complex axb matrix.

  Arguments:
  1 integer: first dimension
  2 integer: second dimension

  Returns;
  1 list of list of complex'''

  mat=[ [ complex(0.,0.) for i in range(a) ] for j in range(b) ]
  return mat

# ======================================================================= #     OK
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
def getcienergy(out,mult,state):
  '''Searches a complete MOLPRO output file for the MRCI energy of (mult,state).

  Arguments:
  1 list of strings: MOLPRO output
  2 integer: mult
  3 integer: state

  Returns:
  1 float: total CI energy of specified state in hartree'''

  ilines=0
  # look for CI program block
  while ilines<len(out):
    if containsstring('1PROGRAM \* CI',out[ilines]):
      # look for multiplicity
      while ilines<len(out):
        if containsstring('Reference symmetry',out[ilines]):
          if containsstring(IToMult[mult],out[ilines]):
            # look for energy
            while ilines<len(out):
              if containsstring('!(MRCI|CI\(SD\)) STATE[\s0-9]+\.1 Energy',out[ilines]):
                kstate=int(out[ilines].replace('.',' ').replace('E',' ').split()[2])
                if kstate==state:
                  return float(out[ilines].split()[-1])
              ilines+=1
          else:
            break
        ilines+=1
    ilines+=1
  print 'CI energy of state %i in mult %i not found!' % (state,mult)
  sys.exit(15)

# ======================================================================= #
def getcidm(out,mult,state1,state2,pol):
  '''Searches a complete MOLPRO output file for a cartesian component of a dipole moment between the two specified states.

  Only takes one multiplicity, since in this script, only non-relativistic dipole moments are calculated. 
  If state1==state2, then this returns a state dipole moment, otherwise a transition dipole moment.

  Arguments:
  1 list of strings: MOLPRO output
  2 integer: mult
  3 integer: state1
  4 integer: state2
  5 integer (0,1,2) or character (X,Y,Z): Polarisation

  Returns:
  1 float: cartesian dipole moment component in atomic units'''

  if pol=='X' or pol=='Y' or pol=='Z':
    pol=IToPol[pol]
  ilines=0
  while ilines<len(out):
    if containsstring('1PROGRAM \* CI',out[ilines]):
      while ilines<len(out):
        if containsstring('Reference symmetry',out[ilines]):
          if containsstring(IToMult[mult],out[ilines]):
            # expectation values are in the results section, transition moments seperately
            if state1==state2:
              while not containsstring('\*\*\*', out[ilines]):
                if containsstring('!.*STATE[\s0-9]+\.1 Dipole moment',out[ilines]):
                  kstate=int(out[ilines].replace('.',' ').replace('E',' ').split()[2])
                  if kstate==state1:
                    return float(out[ilines].split()[-3+pol])
                ilines+=1
            else:
              while not containsstring('\*\*\*', out[ilines]):
                if containsstring('MRCI trans.*<.*\|DM.\|.*>',out[ilines]):
                  braket=out[ilines].replace('<',' ').replace('>',' ').replace('|',' ').replace('.',' ').split()
                  s1=int(braket[2])
                  s2=int(braket[5])
                  p=IToPol[braket[4][2]]
                  if p==pol and ( (s1==state1 and s2==state2) or (s1==state2 and s2==state1) ):
                    return float(out[ilines].split()[3])
                ilines+=1
              #return 0.
          else:
            break
        ilines+=1
    ilines+=1
  if state1==state2:
    print 'Dipole moment of state %i, mult %i, not found!' % (state1,mult)
    sys.exit(16)
  else:
    #print 'CI dipole moment of states %i and %i in mult %i not found!' % (state1,state2,mult)
    return 0.

# ======================================================================= #
def getciang(out,mult,state1,state2,pol):
  '''Searches a complete MOLPRO output file for a cartesian component of a angular momentum between the two specified states.

  Only takes one multiplicity, since in this script, only non-relativistic angular momenta are calculated. 
  If state1==state2, then this returns zero, otherwise a transition angular momentum.

  Arguments:
  1 list of strings: MOLPRO output
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
  sys.exit(17)

# ======================================================================= #
def getsocme(out,mstate1,mstate2,states):
  '''Searches a MOLPRO output for an element of the Spin-Orbit hamiltonian matrix. Also converts from cm^-1 to hartree and adds the diagonal shift.

  Arguments:
  1 list of strings: MOLPRO output
  2 integer: state1 index (nmstates scheme)
  3 integer: state2 index (nmstates scheme)
  4 list of integer: states specs

  Returns:
  1 complex: SO hamiltonian matrix element in hartree'''

  rcm_to_Eh=4.556335e-6
  nmstates=0
  for i in itmult(states):
    nmstates+=states[i-1]*(i)
  if mstate1>nmstates or mstate2>nmstates:
    print 'mstate in getsocme larger than number of states!'
    sys.exit(18)
  # find reference energy
  ilines=0
  while not containsstring('Lowest unperturbed energy E0=',out[ilines]):
    ilines+=1
    if ilines==len(out):
      print 'SO Matrix not found!'
      sys.exit(19)
  eref=complex(float(out[ilines].split()[4]),0)
  while not containsstring('Spin-Orbit Matrix \(CM-1\)',out[ilines]):
    ilines+=1
    if ilines==len(out):
      print 'SO Matrix not found!'
      sys.exit(20)
  ilines+=5
  # get a single matrix element
  block=(mstate2-1)/10
  yoffset=(mstate1-1)*3 + block*(3*nmstates+3)
  xoffset=(mstate2-1)%10
  real=float(out[ilines+yoffset].split()[4+xoffset])*rcm_to_Eh
  imag=float(out[ilines+yoffset+1].split()[xoffset])*rcm_to_Eh
  if mstate1==mstate2:
    real+=eref
  return complex(real,imag)

# ======================================================================= #
def getgrad(out,mult,state,natom):
  '''Searches a MOLPRO output for a SA-MCSCF gradient of a specified state.

  Arguments:
  1 list of strings: MOLPRO output
  2 integer: mult
  3 integer: state
  4 integer: natom

  Returns:
  1 list of list of floats: gradient vector (natom x 3) in atomic units'''

  ilines=0
  grad=[]
  multfound=False
  statefound=False
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
        if containsstring('SA-MC GRADIENT FOR STATE',out[jlines]):
          line=out[jlines].replace('.',' ').replace('E',' ').split()
          if state==int(line[5]):
            statefound=True
          break
        jlines+=1
      if multfound and statefound:
        jlines+=4
        for i in range(natom):
          line=out[jlines+i].split()
          for j in range(3):
            try:
              line[j+1]=float(line[j+1])
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
  print 'Gradient of state %i in mult %i not found!' % (state,mult)
  sys.exit(21)

# ======================================================================= #
def getnacana(out,mult,state1,state2,natom):
  '''Searches a MOLPRO output file for an analytical non-adiabatic coupling vector from SA-MCSCF. 

  Arguments:
  1 list of strings: MOLPRO output
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
  sys.exit(22)

# ======================================================================= #
def getmrcioverlap(out,mult,state1,state2):
  '''Searches a MOLPRO output for a single MRCI overlap (from a CI trans calculation).

  Arguments:
  1 list of strings: MOLPRO output
  2 integer: mult
  3 integer: state1
  4 integer: state2

  Returns:
  1 float: MRCI overlap (THIS MATRIX IS NOT SYMMETRIC!)'''

  ilines=0
  while ilines<len(out):
    if containsstring('Ket wavefunction restored from record .*\.3',out[ilines]):
      line=out[ilines].replace('.',' ').split()
      if mult==int(line[5])-6000:
        break
    ilines+=1
  while not containsstring('\*\*\*',out[ilines]):
    if containsstring('!MRCI overlap',out[ilines]):
      braket=out[ilines].replace('<',' ').replace('>',' ').replace('|',' ').replace('.',' ').split()
      s1=int(braket[2])
      s2=int(braket[4])
      # overlap matrix is NOT symmetric! 
      if s1==state1 and s2==state2:
        return float(out[ilines].split()[3])
    ilines+=1
  return 0.

# ======================================================================= #
def getnacnum(out,mult,state1,state2):
  '''Searches a MOLPRO output for a single non-adiabatic coupling matrix element from a DDR calculation.

  Arguments:
  1 list of strings: MOLPRO output
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
  sys.exit(23)

# ======================================================================= #
def getsmate(out,mult,state1,state2):
  '''Searches a MOLPRO output for an element of the total adiabatic-diabatic transformation matrix.

  Arguments:
  1 list of strings: MOLPRO output
  2 integer: mult
  3 integer: state1
  4 integer: state2

  Returns:
  1 float: Adiabatic-Diabatic transformation matrix element (MATRIX IS NOT SYMMETRIC!)'''

  ilines=0
  multfound=False
  while ilines<len(out):
    if containsstring('Construct non-adiabatic coupling elements by finite difference method', out[ilines]):
      jlines=ilines
      while not containsstring('\*\*\*',out[jlines]):
        if containsstring('Transition density \(R\|R\)',out[jlines]):
          line=out[jlines].replace('.',' ').replace('-',' ').split()
          if mult==int(line[4])-8100:
            multfound=True
          if multfound:
            klines=jlines
            while not containsstring('\*\*\*',out[klines]):
              if containsstring('STATE OVERLAP MATRIX\(TOT\)',out[klines]):
                break
              klines+=1
            return float(out[klines+1+state1].split()[state2-1])
          else:
            multfound=False
            ilines+=1
        jlines+=1
    ilines+=1
  print 'Overlap of states %i - %i in mult %i not found!' % (state1,state2,mult)
  sys.exit(24)

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
      string+='%s %s ' % (eformat(QMout['h'][i][j].real,12,3),eformat(QMout['h'][i][j].imag,12,3))
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
        string+='%s %s ' % (eformat(QMout['dm'][xyz][i][j].real,12,3),eformat(QMout['dm'][xyz][i][j].imag,12,3))
      string+='\n'
    string+=''
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
        string+='%s %s ' % (eformat(QMout['angular'][xyz][i][j].real,12,3),eformat(QMout['angular'][xyz][i][j].imag,12,3))
      string+='\n'
    string+=''
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
        string+='%s ' % (eformat(QMout['grad'][i][atom][xyz],12,3))
      string+='\n'
    string+=''
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
      string+='%s %s ' % (eformat(QMout['nacdt'][i][j].real,12,3),eformat(QMout['nacdt'][i][j].imag,12,3))
    string+='\n'
  string+=''
  # also write wavefunction phases
  string+='! %i Wavefunction phases (%i, complex)\n%i\n' % (7,nmstates,nmstates)
  for i in range(nmstates):
    string+='%s %s\n' % (eformat(QMout['phases'][i],12,3),eformat(0.,12,3))
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
          string+='%s ' % (eformat(QMout['nacdr'][i][j][atom][xyz],12,3))
        string+='\n'
      string+=''
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
      string+='%s %s ' % (eformat(QMout['overlap'][i][j].real,12,3),eformat(QMout['overlap'][i][j].imag,12,3))
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

  string='! 8 Runtime\n%s\n' % (eformat(QMout['runtime'],12,3))
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
      sys.exit(25)
  else:
    try:
      os.makedirs(SCRATCHDIR)
    except OSError:
      print 'Can not create SCRATCHDIR=%s\n' % (SCRATCHDIR)
      sys.exit(26)

# ======================================================================= #
def removequotes(string):
  if string.startswith("'") and string.endswith("'"):
    return string[1:-1]
  elif string.startswith('"') and string.endswith('"'):
    return string[1:-1]
  else:
    return string

# ======================================================================= #
def getsh2prokey(sh2col,key):
  i=-1
  while True:
    i+=1
    try:
      line=sh2col[i]
    except IndexError:
      return ['','']
    line=re.sub('#.*$','',line)
    if line=='\n':
      continue
    line=line.split(None,1)
    if key in line[0].lower():
      return line

# ======================================================================= #     OK
def readQMin(QMinfilename):
  '''Reads the time-step dependent information from QMinfilename. This file contains all information from the current SHARC job: geometry, velocity, number of states, requested quantities along with additional information. The routine also checks this input and obtains a number of environment variables necessary to run MOLPRO.

  Steps are:
  - open and read QMinfilename
  - Obtain natom, comment, geometry (, velocity)
  - parse remaining keywords from QMinfile
  - check keywords for consistency, calculate nstates, nmstates
  - obtain environment variables for path to MOLPRO and scratch directory, and for error handling

  Arguments:
  1 string: name of the QMin file

  Returns:
  1 dictionary: QMin'''

  # read QMinfile
  try:
    QMinfile=open(QMinfilename,'r')
  except IOError:
    print 'QM input file "%s" not found!' % (QMinfilename)
    sys.exit(27)
  QMinlines=QMinfile.readlines()
  QMinfile.close()
  QMin={}

  # Get natom
  try:
    natom=int(QMinlines[0])
  except ValueError:
    print 'first line must contain the number of atoms!'
    sys.exit(28)
  QMin['natom']=natom
  if len(QMinlines)<natom+4:
    print 'Input file must contain at least:\nnatom\ncomment\ngeometry\nkeyword "states"\nat least one task'
    sys.exit(29)

  # Save Comment line
  QMin['comment']=QMinlines[1]

  # Get geometry and possibly velocity (for backup-analytical non-adiabatic couplings)
  QMin['geo']=[]
  QMin['veloc']=[]
  hasveloc=True
  for i in range(2,natom+2):
    if not containsstring('[a-zA-Z][a-zA-Z]?[0-9]*.*[-]?[0-9]+[.][0-9]*.*[-]?[0-9]+[.][0-9]*.*[-]?[0-9]+[.][0-9]*', QMinlines[i]):
      print 'Input file does not comply to xyz file format! Maybe natom is just wrong.'
      sys.exit(30)
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
  i=natom+2
  while i<len(QMinlines):
    line=QMinlines[i]
    line=re.sub('#.*$','',line)
    if len(line.split())==0:
      i+=1
      continue
    key=line.lower().split()[0]
    if 'savedir' in key:
      args=line.split()[1:]
    else:
      args=line.lower().split()[1:]
    if not key in QMin:
      if len(args)>=1 and key=='nacdr' and args[0]=='select':
        # go through all following lines until "end", write pairs of numbers in array
        i+=1
        line=QMinlines[i]
        nacpairs=[]
        while not containsstring('end',line.lower()):
          fields=line.split()
          try:
            nacpairs.append([int(fields[0]),int(fields[1])])
          except ValueError:
            print '"nacdr select" is followed by pairs of state indices, each pair on a new line!'
            sys.exit(31)
          i+=1
          try:
            line=QMinlines[i]
          except IndexError:
            print '"nacdr select" has to be completed with an "end" on another line!'
            sys.exit(32)
        QMin['nacdr']=['select',nacpairs]
      else:
        QMin[key]=args
    else:
      print 'Repeated keyword %s in input file! Check your input!' % (key)
      sys.exit(33)
    i+=1

  # Check for necessary keywords:
  if not 'states' in QMin:
    print 'Number of states not given in QM input file %s!' % (QMinfilename)
    sys.exit(34)
  possibletasks=['h','soc','dm','grad','nacdr','nacdt','overlap','angular']
  if not any([i in QMin for i in possibletasks]):
  #if not 'h' in QMin and not 'soc' in QMin and not 'nacdr' in QMin and not 'dm' in QMin and not 'grad' in QMin and not 'angular' in QMin:
    print 'No tasks found! Tasks are "h", "soc", "dm", "grad", "nacdr", "nacdt", "overlap" and "angular".'
    sys.exit(35)

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
  # higher multiplicities than octets are not supported
  if len(QMin['states'])>8:
    print 'Higher multiplicities than octets are not supported!'
    sys.exit(36)
  # Substitute aliases and remove "h" if "soc" is present
  if 'h' in QMin and 'soc' in QMin:
    QMin=removekey(QMin,'h')

  if ('grad' in QMin or 'nacdr' in QMin) and ('nacdt' in QMin or 'overlap' in QMin):
    print 'It is not allowed to do a gradient/analytical NAC calculation and a numerical NAC/overlap matrix calculation in the same job!'
    sys.exit(37)

  if 'dmdr' in QMin:
    print 'Dipole moment gradients not available!'
    sys.exit(38)

  # Check for correct gradient list
  if 'grad' in QMin:
    if len(QMin['grad'])==0 or QMin['grad'][0]=='all':
      pass
    else:
      for i in range(len(QMin['grad'])):
        try:
          QMin['grad'][i]=int(QMin['grad'][i])
        except ValueError:
          print 'Arguments to keyword "grad" must be "all" or a list of integers!'
          sys.exit(39)
        if QMin['grad'][i]>nmstates:
          print 'State for requested gradient does not correspond to any state in QM input file state list!'
          sys.exit(40)

  # Check for correct nac list
  if 'nacdr' in QMin and len(QMin['nacdr'])==2:
    nacpairs=QMin['nacdr'][1]
    for i in range(len(nacpairs)):
      if nacpairs[i][0]>nmstates or nacpairs[i][1]>nmstates:
        print 'State for requested non-adiabatic couplings does not correspond to any state in QM input file state list!'
        sys.exit(41)

  # open SH2PRO.inp
  sh2prof=open('SH2PRO.inp','r')
  sh2pro=sh2prof.readlines()
  sh2prof.close()

  # Set up environment variables: MOLPRO exe path and scratch directory, default accuracy for cp
  PWD=os.getcwd()
  QMin['pwd']=PWD

  line=getsh2prokey(sh2pro,'molpro')
  if line[0]:
    MOLPRO=line[1]
  else:
    MOLPRO=os.getenv('MOLPRO')
    if not MOLPRO:
      print 'Either set $MOLPRO or give path to MOLPRO in SH2COL.inp!'
      sys.exit(42)
  f=MOLPRO.split()
  if len(f)>1:
    MOLPRO=f[0]
    options=' '.join(f[1:])
  else:
    options=' '
  MOLPRO=os.path.expandvars(MOLPRO)
  MOLPRO=os.path.expanduser(MOLPRO)
  MOLPRO=removequotes(MOLPRO).strip()
  if containsstring(';',MOLPRO):
    print "MOLPRO string contains a semicolon. Do you probably want to execute another command after MOLPRO? I can't do that for you..."
    sys.exit(43)
  if os.path.isdir(MOLPRO):
    MOLPRO+='/molpro'
  QMin['qmexe']=MOLPRO
  QMin['qmexe_options']=options

  line=getsh2prokey(sh2pro,'scratchdir')
  if line[0]:
    SCRATCHDIR=line[1]
  else:
    print 'Please set in SH2PRO.inp a path to a suitable scratch directory!'
    sys.exit(44)
  SCRATCHDIR=os.path.expandvars(SCRATCHDIR)
  SCRATCHDIR=os.path.expanduser(SCRATCHDIR)
  SCRATCHDIR=removequotes(SCRATCHDIR).strip()
  if containsstring(';',SCRATCHDIR):
    print "SCRATCHDIR contains a semicolon. Do you probably want to execute another command after MOLPRO? I can't do that for you..."
    sys.exit(45)
  checkscratch(SCRATCHDIR)
  QMin['scratchdir']=SCRATCHDIR


  # savedir
  if not 'savedir' in QMin:
    QMin['savedir']=getsh2prokey(sh2pro,'savedir')[1].strip()
    if QMin['savedir']=='':
      QMin['savedir']=QMin['pwd']
  else:
    QMin['savedir']=QMin['savedir'][0]


  # Set default gradient accuracies and get accuracies from environment
  QMin['gradaccudefault']=1e-7
  QMin['gradaccumax']=1e-2
  try:
    line=getsh2prokey(sh2pro,'gradaccudefault')
    if line[0]:
      QMin['gradaccudefault']=float(line[1])
    line=getsh2prokey(sh2pro,'gradaccumax')
    if line[0]:
      QMin['gradaccumax']=float(line[1])
  except ValueError:
    print 'Gradient accuracy-related environment variables do not evaluate to numerical values!'
    sys.exit(46)

  # Set CHECKNACS from environment, if true, also try to get related thresholds
  QMin['CHECKNACS']=False
  QMin['CORRECTNACS']=False
  QMin['CHECKNACS_MRCIO']=0.85
  QMin['CHECKNACS_EDIFF']=0.0001
  try:
    line=getsh2prokey(sh2pro,'checknacs')
    if line[0] and 'true' in line[1].lower():
      QMin['CHECKNACS']=True
      line=getsh2prokey(sh2pro,'correctnacs')
      if line[0] and 'true' in line[1].lower():
        QMin['CORRECTNACS']=True
      line=getsh2prokey(sh2pro,'checknacs_mrcio')
      if line[0]:
        QMin['CHECKNACS_MRCIO']=float(line[1])
      line=getsh2prokey(sh2pro,'checknacs_ediff')
      if line[0]:
        QMin['CHECKNACS_EDIFF']=float(line[1])
  except ValueError:
    print 'Non-adiabatic coupling-check-related environment variables do not evaluate to numerical values!'
    sys.exit(47)

  ## Retrieve the old geometry from geom.xyz in the case backwards NAC are needed
  #try:
    #oldgeom=open('geom.xyz','r')
    #data=oldgeom.readlines()
    #oldgeom.close()
    #oldnatom=int(data[0])
    #isoldgeo=True
    #if oldnatom==natom:
      #QMin['oldgeo']=[]
      #for i in range(2,natom+2):
        #fields=data[i].split()
        #if fields[0]!=QMin['geo'][i-2][0]:
          #isoldgeo=False
        #for j in range(1,4):
          #fields[j]=float(fields[j])
        #QMin['oldgeo'].append(fields)
      #if not isoldgeo:
        #QMin=removekey(QMin,'oldgeo')
  #except IOError:
    #pass
  return QMin

# ======================================================================= #
def gettasks(QMin):
  '''Sets up a list of list specifying the kind and order of MOLPRO calculations.

  Each of the lists elements is a list, with a keyword as the first element and a number of additional information depending on the task.

  The list is set up according to a number of task keywords in QMin and the states specifications. These are:
  - h             Calculate the non-relativistic hamiltonian
  - soc           Calculate the spin-orbit hamiltonian
  - dm            Calculate the non-relativistic dipole moment matrices
  - grad          Calculate non-relativistic SA-MCSCF gradients for the specified states
          * all           Calculate gradients for all states in "states"
          * list of int   Calculate only the gradients of these states (nmstates scheme indices)
  - nac           Calculate the non-adiabatic couplings
          * num           Use the MOLPRO DDR program to obtain the matrix < i |d/dt| j >
          * ana           Use MOLPRO CPMCSCF to obtain the matrix of vectors < i |d/dR| j > 
          * smat          Use MOLPRO DDR to obtain the transformation matrix < i(t) | j(t+dt) >
          * numfromana    Use MOLPRO CPMCSCF to obtain v * < i |d/dR| j >

  From this general requests, the specific MOLPRO tasks are created.
  Tasks are:
  - samestep               Use old wavefunction files, do not obtain new orbitals
  - mcscf                 Dont use old wavefunction files, write a new geometry, do a MCSCF calculation to obtain orbitals
  - mcscf:pspace          Like mcscf, but do not move the old wavefunctions files and include pspace threshold in input file
          * 1 float: pspace threshold
  - ci                    Recalculate the MCSCF wavefunction in the MRCI module for all states of mult
          * 1 integer: mult
  - cihlsmat              Calculate the SOC matrix with the AMFI approximation for the given multiplicities
          * list of integer: multiplicities
  - cpgrad                Solve the z-vector equations for the gradient of the specified state
          * 1 integer: mult
          * 2 integer: state
          * 3 float: accuracy
  - forcegrad             Calculate the gradient for this state
          * 1 integer: mult
          * 2 integer: state
  - citrans               Calculate the transition moments between the last step and the current step for the given mult
          * 1 integer: mult
  - ddr                   Calculate the NAC matrix element for the specified states
          * 1 integer: mult
          * 2 integer: state1
          * 3 integer: state2
  - cpnac                 Solve the z-vector equations for the NAC vector between the given states
          * 1 integer: mult
          * 2 integer: state1
          * 3 integer: state2
          * 4 float: accuracy
  - forcenac              Calculate the NAC vector for the given states
          * 1 integer: mult
          * 2 integer: state1
          * 3 integer: state2
  - casdiab               Calculate diabatic orbitals (which maximise the overlap to the last step orbitals)
  - cidiab                Calculate the transition moments for the current step and between current and last step
          * 1 integer: mult
  - ddrdiab               Calculate the adiabatic-diabatic transformation matrix for all states in mult
          * 1 integer: mult
          * 2 integer: states

  Arguments:
  1 dictionary: QMin

  Returns:
  1 list of lists: tasks'''

  states=QMin['states']
  nstates=QMin['nstates']
  nmstates=QMin['nmstates']
  # Currently implemented keywords: soc, dm, grad, nac, samestep
  tasks=[]
  # calculate new orbitals if no samestep
  # appends "mcscf"
  if 'samestep' in QMin:
    tasks.append(['samestep'])
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
    if len(QMin['grad'])==0 or QMin['grad'][0]=='all':
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
      sys.exit(48)
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
  if 'overlap' in QMin:
    tasks.append(['casdiab'])
    for mult in itmult(states):
      tasks.append(['cidiab',mult,states[mult-1]])
      tasks.append(['ddrdiab',mult,states[mult-1]])
  # Case numfromana
  if 'nacdtfromdr' in QMin:
    for mult,i1,i2 in ittwostates(states):
      tasks.append(['cpnac',[mult,i1,i2,QMin['gradaccudefault']]])
      tasks.append(['forcenac',[mult,i1,i2]])
  return tasks

# ======================================================================= #
def writeMOLPROinput(tasks, QMin):
  '''Prepares all files for the next MOLPRO run as specified by the tasks list. Creates the geometry file, moves/copies wavefunction files and writes the MOLPRO input file based on the template file.

  The routine accomplishes:
  - writes geometry file "geom.xyz"
  - opens file "MOLPRO.template"
  - copies title and memory specs from template
  - sets up MOLPRO wavefunction files:
          * if new orbitals are needed, renames the old wavefunction files
          * if samestep/mcscf:pspace is requested, does not rename files
          * checks whether old wavefunctions exist, if NACs are needed
          * writes the corresponding file units into MOLPRO input
  - copies global options (basis set, DK, etc.) from template to input
  - sets up MOLPRO geometry input (no reorientation, correct units, no symmetry)
  - reads and parses the casscf block of the template to obtain the active space and SA information, checks for consistency
  - finally, creates input for all tasks in the list

  Arguments:
  1 list of lists: tasks list
  2 dictionary: QMin'''

  # set up the geometry file ======================================================================================== #
  geofile=open('geom.xyz','w')
  geofile.write('%i\n' % (QMin['natom']))
  geofile.write('Geometry for: '+QMin['comment'])
  for i in range(QMin['natom']):
    line=QMin['geo'][i][0]
    for j in range(3):
      line+=' %15.9f' % QMin['geo'][i][j+1]
    line+='\n'
    geofile.write(line)
  geofile.close()

  # use the MOLPRO template file: currently only supports templates for casscf and global options ==================== #
  try:
    templatefile=open('MOLPRO.template','r')
  except IOError:
    print 'Need MOLPRO template file "MOLPRO.template"!'
    sys.exit(49)
  template=templatefile.readlines()
  templatefile.close()

  # open the MOLPRO input file ======================================================================================= #
  inp=open('MOLPRO.inp','w')

  # get title and memory from template =============================================================================== #
  itemp=0
  while itemp<len(template):
    # ignore comments and blank lines
    if template[itemp].lstrip()=='' or template[itemp].lstrip()[0]=='!':
      itemp+=1
      continue
    # print title
    if containsstring('\*\*\*',template[itemp]):
      inp.write(template[itemp])
    # print memory
    if containsstring('memory',template[itemp]):
      inp.write(template[itemp])
    itemp+=1
  inp.write('\n')

  # set up MOLPRO file units, depending on samestep or orbital calculation ============================================= #
  if tasks[0]==['mcscf']:
    inp.write('file,1,./integrals,new\n')
    inp.write('file,2,./wf.current,new\n')
    # move wf.last to wf.prelast and wf.current to wf.last
    exist=os.path.exists('%s/wf.current' % (QMin['savedir']))
    if exist:
      try:
        os.rename('%s/wf.last' % (QMin['savedir']),'%s/wf.prelast' % (QMin['savedir']))
      except OSError:
        pass
      try:
        os.rename('%s/wf.current' % (QMin['savedir']),'%s/wf.last' % (QMin['savedir']))
      except OSError:
        pass
  elif tasks[0][0]=='mcscf:pspace':
    inp.write('file,1,./integrals,new\n')
    inp.write('file,2,./wf.current,new\n')
  elif tasks[0]==['samestep']:
    # check if wf file is actually there
    exist=os.path.isfile('%s/wf.current' % (QMin['savedir']))
    if not exist:
      exist=os.path.isfile('%s/wf.current' % (QMin['scratchdir']))
    if not exist:
      print 'Samestep requested, but no wf.current found!'
      sys.exit(50)
    inp.write('file,1,./integrals\n')
    inp.write('file,2,./wf.current\n')
  else:
    print 'Tasks should start with either mcscf or samestep!'
    sys.exit(51)
  # check if wf file from last step is actually there
  # noinit will be used in the casscf block below
  noinit=True
  exist=os.path.exists('%s/wf.last' % (QMin['savedir']))
  if not exist:
    exist=os.path.isfile('%s/wf.last' % (QMin['scratchdir']))
  if exist:
    inp.write('file,3,./wf.last\n\n')
    noinit=False
  else:
    if 'nacdt' in QMin or 'overlap' in QMin:
      print 'No wf.last, but "nacdt" or "overlap" is requested!'
      sys.exit(52)
    exist2=os.path.exists('wf.init')
    if exist2:
      inp.write('file,3,./wf.init\n\n')
      noinit=False
  #if 'grad' in QMin or 'nacdr' in QMin:
    #inp.write('file,9,./wf.gradient,new\n\n')

  # if first task is mcscf: get global options, write geometry input ================================================== #
  if tasks[0]==['mcscf']:
    # scan the template file for global options (i.e. anything before the first program card)
    itemp=0
    while itemp<len(template):
      # ignore comments
      if template[itemp].lstrip()=='' or template[itemp].lstrip()[0]=='!':
        itemp+=1
        continue
      # ignore title and memory
      elif containsstring('\*\*\*',template[itemp]) or containsstring('memory',template[itemp]):
        itemp+=1
        continue
      # stop at the first program card
      elif template[itemp].lstrip()[0]=='{':
        break
      else:
        inp.write(template[itemp])
      itemp+=1
    inp.write('\n')
    # write the geometry input block
    string='nosym\norient,noorient\n'
    if 'unit' in QMin:
      if QMin['unit'][0]=='angstrom':
        string+='angstrom\n'
      elif QMin['unit'][0]=='bohr':
        string+='bohr\n'
      else:
        print 'Dont know input unit %s!' % (QMin['unit'][0])
    else:
      string+='bohr\n'
    string+='geometry={\ninclude geom.xyz\n}\n\n'
    inp.write(string)

  string=''
  # search for tasks expec
  for itask in range(len(tasks)):
    if tasks[itask][0]=='expec':
      string+='gexpec,%s\n' % (tasks[itask][1])
  inp.write(string)

  string='gprint,orbitals,civectors;\n\n'
  inp.write(string)

  # look up casscf block to get active space and wavefunction definition
  # these information are needed no matter whether samestep or mcscf is requested
  itemp=0
  while itemp<len(template):
    if template[itemp].lstrip()=='' or template[itemp].lstrip()[0]=='!':
      itemp+=1
      continue
    if template[itemp].lstrip()[0]=='{' and containsstring('multi|casscf|mcscf',template[itemp]):
      break
    itemp+=1
  if itemp==len(template):
    print 'No casscf block in template file!\nNeed a casscf block in the template file stating the active space and SA-related information!'
    sys.exit(53)
  # get active space block (from program invocation to before first wf card)
  k=template[itemp].find('{')
  ASblock=template[itemp][k+1:]
  while True:
    itemp+=1
    if containsstring('wf',template[itemp]):
      break
    ASblock+=template[itemp]
  # put maxit=40 on the first line, if maxit not given
  if not containsstring('maxit',ASblock):
    k=min(ASblock.find(','),ASblock.find('\n'))
    ASblock=ASblock[:k]+',maxit=40'+ASblock[k:]
  # put convergenve thresholds on the first line, if not given
  if not containsstring('energy',ASblock):
    k=min(ASblock.find(','),ASblock.find('\n'))
    ASblock=ASblock[:k]+',energy=0.1e-7'+ASblock[k:]
  # put convergenve thresholds on the first line, if not given
  if not containsstring('gradient',ASblock):
    k=min(ASblock.find(','),ASblock.find('\n'))
    ASblock=ASblock[:k]+',gradient=0.1e-6'+ASblock[k:]
  # put convergenve thresholds on the first line, if not given
  if not containsstring('step',ASblock):
    k=min(ASblock.find(','),ASblock.find('\n'))
    ASblock=ASblock[:k]+',step=0.1e-2'+ASblock[k:]
  # parse the ASblock to ASdata
  ASdata={}
  AStemp=ASblock.splitlines()
  for i in range(len(AStemp)):
    line=AStemp[i].replace(',',' ').split()
    ASdata[line[0]]=line[1:]
  for i in ASdata:
    if i=='frozen' or i=='closed' or i=='occ':
      ASdata[i]=int(ASdata[i][0])
  if not 'occ' in ASdata or not 'closed' in ASdata:
    print 'Did not find keywords occ or closed in template!\nPlease provide active space information in template CASSCF block!'
    sys.exit(54)
  # get wf cards block including weight, select, etc... (from first wf card to before cpmcscf)
  WFblock=template[itemp]
  while True:
    itemp+=1
    if containsstring('cpmcscf',template[itemp]):
      break
    if containsstring('}',template[itemp]):
      k=template[itemp].find('}')
      WFblock+=template[itemp][:k]
      break
    WFblock+=template[itemp]
  #parse the WFblock to WFdata
  WFdata=[]
  WFtemp=WFblock.splitlines()
  for i in range(len(WFtemp)):
    temp=WFtemp[i].replace(',',' ').split()
    if containsstring('wf',WFtemp[i]):
      if i>0:
        WFdata.append(block)
      block={}
      block['wf']=temp[1:]
      try:
        for j in range(3):
          block['wf'][j]=int(block['wf'][j])
      except ValueError:
        print 'wf card in template file is wrong!'
        sys.exit(55)
      except IndexError:
        print 'wf card in template file needs three entries (electron number, symmetry, multiplicity)!'
        sys.exit(56)
    elif containsstring('state',WFtemp[i]):
      try:
        block['state']=int(temp[1])
      except ValueError:
        print 'state card in template file is wrong!'
        sys.exit(57)
    elif containsstring('weight',WFtemp[i]):
      try:
        block['weight']=[]
        for j in range(len(temp)-1):
          block['weight'].append(int(temp[j+1]))
      except ValueError:
        print 'weight card in template file is wrong!'
        sys.exit(58)
    else:
      block[temp[0]]=temp[1:]
  WFdata.append(block)
  # check for consistent electron numbers and multiplicities
  for i in range(len(WFdata)):
    if (WFdata[i]['wf'][0]+WFdata[i]['wf'][2])%2!=0:
      print 'Electron number and multiplicity inconsistent in CASSCF template block!',WFdata[i]
      sys.exit(59)
  # check if state card is there
  for i in range(len(WFdata)):
    if not 'state' in WFdata[i]:
      print 'No state card given in CASSCF block!'
      sys.exit(60)
  # check if weighting is over sufficient number of states
  for i in range(len(WFdata)):
    if 'weight' in WFdata[i]:
      if len(WFdata[i]['weight'])<WFdata[i]['state']:
        print 'Weighting is over too few states compared to state card!'
        sys.exit(61)

  # ======================= Here starts parsing of the tasks step by step ================= #
  for itask in range(len(tasks)):
    task=tasks[itask]
    string=''
    # samestep: do nothing ============================================================================================== #
    if task[0]=='samestep':
      pass
    # expec: everything already taken care of ========================================================================== #
    elif task[0]=='expec':
      pass
    # mcscf: create a casscf block including maxiter, ASblock, orbital records, WFblock ================================ #
    elif task[0]=='mcscf':
      # print { to start casscf block
      string+='{'+ASblock
      # print start card
      if not noinit:
        string+='\nstart,2140.3'
      # print orbital card, WFblock, bracket
      string+='\norbital,2140.2\n\n'+WFblock+'\n};\n\n'
    # mcscf:pspace: create the same casscf block as above, but include a pspace statement (convergence helper...) ====== #
    elif task[0]=='mcscf:pspace':
      # print { to start casscf block
      string+='{'+ASblock
      # print start card
      if not noinit:
        string+='\nstart,2140.3'
      # print orbital card, WFblock, bracket
      string+='\norbital,2140.2\npspace,%.2f\n\n' % (task[1])
      string+=WFblock+'\n};\n\n'
    # ci: maxiter, orbital, save, noexc, core and wf, state from QMin ================================================== #
    elif task[0]=='ci':
      # print header
      string+='{ci\nmaxiter,250,1000\norbital,2140.2\nsave,%i.2\nnoexc\ncore,%i\n' % (6000+task[1],ASdata['closed'])
      # wf block from WFdata corresponding to current multiplicity
      foundmult=False
      for i in range(len(WFdata)):
        if task[1]==WFdata[i]['wf'][2]+1:
          foundmult=True
          break
      if foundmult:
        nelec=WFdata[i]['wf'][0]
      else:
        nelec=WFdata[1]['wf'][0]
      string+='wf,%i,%i,%i\nstate,%i\n};\n\n' % (nelec,1,task[1]-1,task[2])
    # same as above, but with nstati statement (convergence helper) ==================================================== #
    elif task[0]=='ci:nstati':
      print 'ci:nstati is not yet implemented!'
      sys.exit(62)
    # same as above, but with pspace statement (convergence helper) ==================================================== #
    elif task[0]=='ci:pspace':
      # print header
      string+='{ci\nmaxiter,250,1000\norbital,2140.2\nsave,%i.2\nnoexc\npspace,%i\ncore,%i\n' % (6000+task[1],task[3],ASdata['closed'])
      # wf block from WFdata corresponding to current multiplicity
      foundmult=False
      for i in range(len(WFdata)):
        if task[1]==WFdata[i]['wf'][2]+1:
          foundmult=True
          break
      if foundmult:
        nelec=WFdata[i]['wf'][0]
      else:
        nelec=WFdata[1]['wf'][0]
      string+='wf,%i,%i,%i\nstate,%i\n};\n\n' % (nelec,1,task[1]-1,task[2])
    # make spin orbit calculation including all given multiplicities =================================================== #
    elif task[0]=='cihlsmat':
      string+='{ci\nhlsmat,amfi'
      for i in range(len(task)-1):
        string+=',%i.2' % (6000+task[i+1])
      string+='\nprint,hls=1\n};\n\n'
    # make a casscf cp equation calculation, samestep the orbitals, cpmcscf cards ======================================= #
    elif task[0]=='cpgrad':
      string+='{'+ASblock+'start,2140.2\ndont,orbital\n'+WFblock+'\nprint,micro\n'
      for i in range(len(task)-1):
        # check whether the state averaging contains enough states for cp
        statethere=False
        for j in range(len(WFdata)):
          if WFdata[j]['wf'][2]+1==task[i+1][0]:
            if WFdata[j]['state']>=task[i+1][1]:
              statethere=True
        if not statethere:
          print 'cpmcscf for state %i only possible for SA>=%i!\nPlease increase the number of states in the SA-CASSCF information in the template file!' % (task[i+1][1],task[i+1][1])
          sys.exit(63)
        # build string
        string+='cpmcscf,grad,state=%i.1,ms2=%i,record=%i.1,accu=%18.15f\n' % (task[i+1][1],task[i+1][0]-1,5000+task[i+1][0]*100+task[i+1][1],task[i+1][2])
      string+='};\n\n'
    # forcegrad: samc record is as above =============================================================================== #
    elif task[0]=='forcegrad':
      string+='{force\nsamc,%i.1\n};\n\n' % (5000+task[1][0]*100+task[1][1])
    # citrans: transition density matrix =============================================================================== #
    elif task[0]=='citrans':
      string+='{ci\ntrans,%i.2,%i.3\ndm,%i.2\n};\n\n' % (6000+task[1],6000+task[1],8000+task[1])
    # ddr: dm record from above + states =============================================================================== #
    elif task[0]=='ddr':
      string+='{ddr,-%s,2140.2,2140.3,%i.2\nstate,%i.1,%i.1\n};\n\n' % (QMin['dt'][0],8000+task[1],task[2],task[3])
    # cpnac: make a cpmcscf nacme calculation ========================================================================== #
    elif task[0]=='cpnac':
      string+='{'+ASblock+'start,2140.2\ndont,orbital\n'+WFblock+'\nprint,micro\n'
      for i in range(len(task)-1):
        # check whether the state averaging contains enough states for cp
        statethere=False
        for j in range(len(WFdata)):
          if WFdata[j]['wf'][2]+1==task[i+1][0]:
            if WFdata[j]['state']>=task[i+1][1] and WFdata[j]['state']>=task[i+1][2]:
              statethere=True
        if not statethere:
          print 'cpmcscf for states %i and %i only possible for SA>=%i,%i!\nPlease increase the number of states in the SA-CASSCF information in the template file!' % (task[i+1][1],task[i+1][2],task[i+1][1],task[i+1][2])
          sys.exit(64)
        # build string
        string+='cpmcscf,nacm,state1=%i.1,state2=%i.1,ms2=%i,record=%i.1,accu=%18.15f\n' % (task[i+1][1],task[i+1][2],task[i+1][0]-1,5020+task[i+1][0]*100+10*task[i+1][1]+task[i+1][2],task[i+1][3])
      string+='};\n\n'
    # forcenac: evaluate the cp nacme ================================================================================== #
    elif task[0]=='forcenac':
      string+='{force\nsamc,%i.1\n};\n\n' % (5020+task[i+1][0]*100+10*task[i+1][1]+task[i+1][2])
    # casdiab: diabatize orbitals ====================================================================================== #
    elif task[0]=='casdiab':
      string+='{'+ASblock+'noextra\nstart,2140.2\norbital,2180.2\ndont,orbital\n'+WFblock+'\ndiab,2140.3,method=1\n};\n\n'
    # cidiab: transition density matrices ============================================================================== #
    elif task[0]=='cidiab':
      string+='{ci\nmaxiter,250,1000\norbital,2180.2\nsave,%i.2\nnoexc\ncore,%i\n' % (6100+task[1],ASdata['closed'])
      foundmult=False
      for i in range(len(WFdata)):
        if task[1]==WFdata[i]['wf'][2]+1:
          foundmult=True
          break
      if foundmult:
        nelec=WFdata[i]['wf'][0]
      else:
        nelec=WFdata[1]['wf'][0]
      string+='wf,%i,%i,%i\nstate,%i\n};\n\n' % (nelec,1,task[1]-1,task[2])
      string+='{ci\ntrans,%i.2,%i.2\ndm,%i.2\n};\n\n' % (6100+task[1],6100+task[1],8100+task[1])
      string+='{ci\ntrans,%i.2,%i.3\ndm,%i.2\n};\n\n' % (6100+task[1],6000+task[1],8200+task[1])
    # ddrdiab: overlap matrices ======================================================================================== #
    elif task[0]=='ddrdiab':
      string+='{ddr\norbital,2180.2,2140.3\ndensity,%i.2,%i.2\nmixing' % (8100+task[1],8200+task[1])
      for i in range(task[2]):
        string+=',%i.1' % (i+1)
      string+='\n};\n\n'
    else: # ============================================================================================================ #
      print 'Unknown task keyword %s found in writeMOLPROinput!' % task[0]
      sys.exit(65)
    inp.write(string)
  return

# ======================================================================= #
def runMOLPRO(QMin):
  '''Calls MOLPRO in a shell with the SCRATCHDIR directory as integral directory. 

  Arguments:
  1 dictionary: QMin

  Returns:
  1 integer: MOLPRO exit code'''

  string='%s MOLPRO.inp %s -W%s -I%s -d%s' % (QMin['qmexe'],QMin['qmexe_options'],QMin['savedir'],QMin['scratchdir'],QMin['scratchdir'])
  if PRINT:
    print datetime.datetime.now()
    print '===> Running MOLPRO:\n\n%s\n\nError Code:' % (string)
    sys.stdout.flush()
  try:
    runerror=sp.call(string,shell=True) # TODO: Why is the shell necessary here?
    if PRINT:
      print '%s\n\n' % (runerror)
  except OSError:
    print 'MOLPRO call have had some serious problems:',OSError
    sys.exit(66)
  return runerror

# ======================================================================= #
def redotasks(tasks,QMin):
  '''Screens the MOLPRO output file for error messages and reconstructs the tasks list. The new list contains all remaining tasks which have not been accomplished. The task which caused the crash is redone with altered parameters to ensure convergence.

  Currently, the script can deal with the following MOLPRO errors:
  - EXCESSIVE GRADIENT IN CI:
          This error occurs sometimes if the initial guess for the CI vectors in the MCSCF calculation is bad. Usually, this can be dealt with by including more CSFs in the primary configuration space. 
          If this error occurs, the script will restart MOLPRO with a pspace threshold of 1. If this does not lead to success, the threshold is increased further. If the calculation still crashes with a threshold of 9, the script returns with exit code 1.
  - NO CONVERGENCE IN CPMCSCF:
          This error in the calculation of gradients and non-adiabatic coupling vectors occurs if the active space contains strongly doubly occupied/empty orbitals and the associated orbital rotation gradients are very small.
          If this error occurs, the corresponding calculation is started with a looser convergence criterium. How the criterium is altered can be changed using environment variables GRADACCUDEFAULT, GRADACCUMAX, GRADACCUSTEP

  Arguments:
  1 list of lists: task list
  2 dictionary: QMin

  Returns:
  1 list of lists: new task list'''

  newtasks=[]
  outfile=open('MOLPRO.out','r')
  out=outfile.readlines()
  outfile.close()
  ilines=0
  for itask in range(len(tasks)):
    task=tasks[itask]
    # samestep: pass
    if task[0]=='samestep':
      pass
    # expec: pass
    elif task[0]=='expec':
      pass
    # mcscf: excessive CI gradient error, to be implemented
    elif task[0]=='mcscf':
      ilines,data=nextblock(out,'MULTI',1)
      out=out[ilines:]
      idata=0
      while idata<len(data):
        if containsstring('EXCESSIVE GRADIENT IN CI',data[idata]):
          if PRINT:
            print '=> Excessive CI gradient error: increasing p-space threshold...\n\n'
          newtask=['mcscf:pspace',1.]
          newtasks.append(newtask)
          for j in range(len(tasks)-itask-1):
            newtasks.append(tasks[j+itask+1])
          return newtasks
        idata+=1
    elif task[0]=='mcscf:pspace':
      ilines,data=nextblock(out,'MULTI',1)
      out=out[ilines:]
      idata=0
      while idata<len(data):
        if containsstring('EXCESSIVE GRADIENT IN CI',data[idata]):
          if PRINT:
            print '=> Excessive CI gradient error: increasing p-space threshold...\n\n'
          newtask=['mcscf:pspace',task[1]*4.]
          if task[1]>100.:
            print 'Excessive gradient in CI error unsolvable in MCSCF!'
            sys.exit(67)
          newtasks.append(newtask)
          for j in range(len(tasks)-itask-1):
            newtasks.append(tasks[j+itask+1])
          return newtasks
        idata+=1
    # ci: excessive CI gradient error, to be implemented
    elif task[0]=='ci':
      ilines,data=nextblock(out,'CI',1)
      out=out[ilines:]
      idata=0
      while idata<len(data):
        if containsstring('NO CONVERGENCE IN REFERENCE CI',data[idata]):
          if PRINT:
            print '=> No convergence in reference CI error: increasing p-space threshold...\n\n'
          newtask=['ci:pspace',task[1],task[2],1.]
          newtasks.append(['samestep'])
          newtasks.append(newtask)
          for j in range(len(tasks)-itask-1):
            newtasks.append(tasks[j+itask+1])
          return newtasks
        idata+=1
    elif task[0]=='ci:pspace':
      ilines,data=nextblock(out,'CI',1)
      out=out[ilines:]
      idata=0
      while idata<len(data):
        if containsstring('NO CONVERGENCE IN REFERENCE CI',data[idata]):
          if PRINT:
            print '=> No convergence in reference CI error: increasing p-space threshold...\n\n'
          newtask=['ci:pspace',task[1],task[2],task[3]*4.]
          if task[3]>100.:
            print 'No convergence in reference CI error unsolvable in MRCI!'
            sys.exit(68)
          newtasks.append(['samestep'])
          newtasks.append(newtask)
          for j in range(len(tasks)-itask-1):
            newtasks.append(tasks[j+itask+1])
          return newtasks
        idata+=1
    elif task[0]=='ci:nstati':
      ilines,data=nextblock(out,'CI',1)
      out=out[ilines:]
    # cihlsmat: currently no errors known
    elif task[0]=='cihlsmat':
      ilines,data=nextblock(out,'CI',1)
      out=out[ilines:]
    # cpgrad: convergence not reached error
    elif task[0]=='cpgrad':
      ilines,data=nextblock(out,'MULTI',1)
      out=out[ilines:]
      idata=0
      for i in range(len(task)-1):
        while idata<len(data):
          if containsstring('Solving MCSCF z-vector',data[idata]):
            break
          idata+=1
        # check whether the state number from task is identical to state on the current line
        if not task[i+1][1]==int(data[idata].split()[6][:-3]):
          print 'Missing a z-vector calculation for state %i mult %i!' % (task[i+1][1],task[i+1][0])
          sys.exit(69)
        idata+=2
        conv=1e6
        while containsstring('ITERATI',data[idata]) or containsstring('VECTORS REACHED',data[idata]):
          if containsstring('VECTORS REACHED',data[idata]):
            idata+=1
            continue
          if float(data[idata].split()[-1])<conv:
            conv=float(data[idata].split()[-1])
          idata+=1
        idata+=1
        if containsstring('Convergence reached',data[idata]):
          conv=0.
        if conv<task[i+1][2]:
          continue
        else:
          if PRINT:
            print '=> No convergence in CPMCSCF: decreasing accuracy to %f\n\n' % (conv*1.2)
          newtask=['cpgrad']
          newtask.append([task[i+1][0],task[i+1][1],1.2*conv])
          if newtask[1][2]>QMin['gradaccumax']:
            print 'Could not converge gradient: ',newtask
            sys.exit(70)
          for j in range(len(task)-i-2):
            newtask.append(task[j+i+2])
          newtasks.append(['samestep'])
          newtasks.append(newtask)
          for j in range(len(tasks)-itask-1):
            newtasks.append(tasks[j+itask+1])
          return newtasks
    # forcegrad: currently no errors known
    elif task[0]=='forcegrad':
      ilines,data=nextblock(out,'FORCE',1)
      out=out[ilines:]
    # cpnac: same as cpgrad
    elif task[0]=='cpnac':
      ilines,data=nextblock(out,'MULTI',1)
      out=out[ilines:]
      idata=0
      for i in range(len(task)-1):
        while idata<len(data):
          if containsstring('SOLVING CP-MCSCF NACM',data[idata]):
            break
          idata+=1
        # check whether the state number from task is identical to state on the current line
        if not task[i+1][2]==int(data[idata].replace('.',' ').replace('-',' ').split()[4]) or not task[i+1][1]==int(data[idata].replace('.',' ').replace('-',' ').split()[6]):
          print 'Missing a z-vector calculation for states %i,%i mult %i!' % (task[i+1][1],task[i+1][2],task[i+1][0])
          sys.exit(71)
        idata+=2
        conv=1e6
        while containsstring('ITERATI',data[idata]) or containsstring('VECTORS REACHED',data[idata]):
          if containsstring('VECTORS REACHED',data[idata]):
            idata+=1
            continue
          if float(data[idata].split()[-1])<conv:
            conv=float(data[idata].split()[-1])
          idata+=1
        if conv<task[i+1][3]:
          continue
        else:
          if PRINT:
            print '=> No convergence in CPMCSCF: decreasing accuracy to %f\n\n' % (conv*1.2)
          newtask=['cpnac']
          newtask.append([task[i+1][0],task[i+1][1],task[i+1][2],1.2*conv])
          if newtask[1][3]>QMin['gradaccumax']:
            print 'Could not converge gradient: ',newtask
            sys.exit(72)
          for j in range(len(task)-i-2):
            newtask.append(task[j+i+2])
          newtasks.append(['samestep'])
          newtasks.append(newtask)
          for j in range(len(tasks)-itask-1):
            newtasks.append(tasks[j+itask+1])
          return newtasks
    # forcenac: currently no errors known
    elif task[0]=='forcenac':
      ilines,data=nextblock(out,'FORCE',1)
      out=out[ilines:]
    # citrans: currently no errors known
    elif task[0]=='citrans':
      ilines,data=nextblock(out,'CI',1)
      out=out[ilines:]
    # ddr: currently no errors known. !!! DDR does not create a program block !!!
    elif task[0]=='ddr':
      pass
    # casdiag: currently no errors known
    elif task[0]=='casdiab':
      ilines,data=nextblock(out,'MULTI',1)
      out=out[ilines:]
    # cidiag: currently no errors known
    elif task[0]=='cidiab':
      ilines,data=nextblock(out,'CI',1)
      out=out[ilines:]
    # ddrdiag: currently no errors known !!! DDR does not create a program block !!!
    elif task[0]=='ddrdiab':
      pass
    else:
      print 'Unknown task keyword %s found in redotasks!' % task[0]
      sys.exit(73)
  return newtasks

# ======================================================================= #
def catMOLPROoutput(outcounter):
  '''Reads all MOLPRO output files from the current time step and concatenates them for the extraction of the requested quantities.

  Arguments:
  1 integer: number of output files

  Returns:
  1 list of strings: Concatenation of all MOLPRO output files'''

  if PRINT:
    print '===> Processing output from:\n'
  out=[]
  for i in range(outcounter):
    if PRINT:
      print 'MOLPRO%04i.out' % (i+1)
    outfile=open('MOLPRO%04i.out' % (i+1),'r')
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
  1 list of strings: Concatenated MOLPRO output
  2 dictionary: QMin

  Returns:
  1 dictionary: QMout'''

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
      h[istate][istate]=complex(getcienergy(out,mult,state))
    QMout['h']=h
  # SOC: get SOC matrix and construct hamiltonian, returns a matrix(nmstates,nmstates)
  if 'soc' in QMin:
    # soc: matrix is not diagonal, two nested loop
    soc=makecmatrix(nmstates,nmstates)
    for istate in range(nmstates):
      for jstate in range(nmstates):
        soc[istate][jstate]=getsocme(out,istate+1,jstate+1,states)
    QMout['h']=soc
  # DM: get vector of three dipole matrices, three nested loops, returns a list of three matrices(nmstates,nmstates)
  if 'dm' in QMin:
    dm=[]
    for xyz in range(3):
      dm.append(makecmatrix(nmstates,nmstates))
      for mult,state1,state2 in ittwostatesfull(states):
        for istate,jstate in MultStateToIstateJstate(mult,state1,state2,states):
          dm[xyz][istate-1][jstate-1]=complex(getcidm(out,mult,state1,state2,xyz))
    QMout['dm']=dm
  # Grad: for argument all single loop, otherwise a bit more complex, returns a list of nmstates vectors
  if 'grad' in QMin:
    grad=[]
    if QMin['grad']==['all'] or QMin['grad']==[]:
      for istate in range(nmstates):
        mult,state=IstateToMultState(istate+1,states)
        grad.append(getgrad(out,mult,state,natom))
    else:
      for istate in range(nmstates):
        gradatom=[]
        for iatom in range(natom):
          gradatom.append([0.,0.,0.])
        grad.append(gradatom)
      for iarg in range(len(QMin['grad'])):
        mult,state=IstateToMultState(QMin['grad'][iarg],states)
        for istate in MultStateToIstate(mult,state,states):
          grad[istate-1]=getgrad(out,mult,state,natom)
    QMout['grad']=grad
  # NAC: case of keyword "num": returns a matrix(nmstates,nmstates)
  # and also collects the mrci overlaps for later error evaluation
  if 'nacdt' in QMin:
    nac=makecmatrix(nmstates,nmstates)
    mrcioverlap=makermatrix(nmstates,nmstates)
    for mult,state1,state2 in ittwostatesfull(states):
      for istate,jstate in MultStateToIstateJstate(mult,state1,state2,states):
        nac[istate-1][jstate-1]=complex(getnacnum(out,mult,state1,state2))
        mrcioverlap[istate-1][jstate-1]=getmrcioverlap(out,mult,state1,state2)
    QMout['nacdt']=nac
    QMout['mrcioverlap']=mrcioverlap
  # NAC: case of keyword "ana": returns a matrix(nmstates,nmstates) of vectors
  if 'nacdr' in QMin:
    grad=[]
    for i in range(natom):
      grad.append([0.,0.,0.])
    nac=[ [ grad for i in range(nmstates) ] for j in range(nmstates) ]
    if len(QMin['nacdr'])==2 and QMin['nacdr'][0]=='select':
      nacpairs=QMin['nacdr'][1]
      for i in range(len(nacpairs)):
        m1,i1=IstateToMultState(nacpairs[i][0],states)
        m2,i2=IstateToMultState(nacpairs[i][1],states)
        if m1==m2:
          for istate,jstate in MultStateToIstateJstate(m1,i1,i2,states):
            nac[istate-1][jstate-1]=getnacana(out,m1,i1,i2,natom)
        m1,i1=IstateToMultState(nacpairs[i][1],states)
        m2,i2=IstateToMultState(nacpairs[i][0],states)
        if m1==m2:
          for istate,jstate in MultStateToIstateJstate(m1,i1,i2,states):
            nac[istate-1][jstate-1]=getnacana(out,m1,i1,i2,natom)
    else:
      for mult,state1,state2 in ittwostatesfull(states):
        for istate,jstate in MultStateToIstateJstate(mult,state1,state2,states):
          nac[istate-1][jstate-1]=getnacana(out,mult,state1,state2,natom)
    QMout['nacdr']=nac
  # NAC: case of keyword "smat": returns a matrix(nmstates,nmstates)
  if 'overlap' in QMin:
    nac=makecmatrix(nmstates,nmstates)
    mrcioverlap=makermatrix(nmstates,nmstates)
    for mult,state1,state2 in ittwostatesfull(states):
      for istate,jstate in MultStateToIstateJstate(mult,state1,state2,states):
        nac[istate-1][jstate-1]=complex(getsmate(out,mult,state1,state2))
        mrcioverlap[istate-1][jstate-1]=getmrcioverlap(out,mult,state1,state2)
    QMout['overlap']=nac
    QMout['mrcioverlap']=mrcioverlap
  # NAC: case of numfromana
  if 'nacdtfromdr' in QMin:
    grad=[]
    for i in range(natom):
      grad.append([0.,0.,0.])
    nac=[ [ grad for i in range(nmstates) ] for j in range(nmstates) ]
    for mult,state1,state2 in ittwostatesfull(states):
      for istate,jstate in MultStateToIstateJstate(mult,state1,state2,states):
        nac[istate-1][jstate-1]=getnacana(out,mult,state1,state2,natom)
    QMout['nacdt']=nac
  if 'angular' in QMin:
    ang=[]
    for xyz in range(3):
      ang.append(makecmatrix(nmstates,nmstates))
      for mult,state1,state2 in ittwostatesfull(states):
        for istate,jstate in MultStateToIstateJstate(mult,state1,state2,states):
          ang[xyz][istate-1][jstate-1]=complex(getciang(out,mult,state1,state2,xyz))
    QMout['angular']=ang
  return QMout

# ======================================================================= #
def mrcioverlapsok(QMin,QMout):
  '''Checks for all diagonal elements of the MRCI overlaps whether their absolute value is above the relevant threshold.

  Arguments:
  1 dictionary: QMin
  2 dictionary: QMout

  Returns:
  1 Boolean'''

  ok=True
  mrcioverlap=QMout['mrcioverlap']
  states=QMin['states']
  nmstates=QMin['nmstates']
  h=QMout['h']
  for istate in range(nmstates):
    if abs(mrcioverlap[istate][istate])<QMin['CHECKNACS_MRCIO']:
      ok=False
  return ok

# ======================================================================= #
def setnacszero(QMin,QMout):
  '''Sets non-adiabatic coupling elements to zero, if there corresponding MRCI overlaps are bad and the two coupled states are too far separated.

  Arguments:
  1 dictionary: QMin
  2 dictionary: QMout

  Returns:
  1 list of list of complex: nac matrix'''

  if PRINT:
    print '===> Checking non-adiabatic couplings:\n'
  mrcioverlap=QMout['mrcioverlap']
  states=QMin['states']
  nmstates=QMin['nmstates']
  h=QMout['h']
  nac=QMout['nacdt']
  for istate in range(nmstates):
    if abs(mrcioverlap[istate][istate])<QMin['CHECKNACS_MRCIO']:
      if PRINT:
        print '=> MRCI overlap of state \t%i is bad:' % (istate)
      for jstate in range(nmstates):
        if abs(h[istate][istate]-h[jstate][jstate])>QMin['CHECKNACS_EDIFF']:
          nac[istate][jstate]=complex(0.)
          nac[jstate][istate]=complex(0.)
          if PRINT:
            print '- setting nac[\t%i][\t%i]=-nac[\t%i][\t%i]=0.' % (istate+1,jstate+1,jstate+1,istate+1)
  return nac

# ======================================================================= #
def redoNacjob(QMin):
  '''Plans a completely new calculation to obtain CPMCSCF non-adiabatic couplings, in the case that the numerical couplings are corrupted.

  Arguments:
  1 dictionary: QMin

  Returns:
  1 dictionary: a new QMin, which requests nac ana'''

  QMin2={}
  # needed: geo, gradaccu..., natom, nstates, nmstates, states, pwd, qmexe, scratchdir, unit
  necessary=['comment','geo','gradaccudefault','gradaccumax',
             'natom','nmstates','nstates','pwd','qmexe','scratchdir','unit','states']
  for i in necessary:
    QMin2[i]=QMin[i]
  # only task: analytical couplings
  QMin2['samestep']=[]
  QMin2['nacdr']=[]
  return QMin2

# ======================================================================= #
def contractNACveloc(QMin,QMout,QMout2):
  '''Contracts the matrix of vectorial non-adiabatic couplings with the velocity vector to obtain < i | d/dt| j >.

  < i | d/dt| j > = sum_atom sum_cart v_atom_cart * < i | d/dR_atom_cart| j >

  Arguments:
  1 dictionary: QMin, containing veloc
  2 dictionary: the QMout containing soc, dm, grad
  3 dictionary: the new QMout with the analytical couplings

  Returns:
  1 dictionary: QMout, including everything from the old QMout and the new NAC matrix'''

  # calculates the scalar product of the analytical couplings and the velocity,
  # and puts the resulting matrix into QMout
  veloc=QMin['veloc']
  nmstates=QMin['nmstates']
  natom=QMin['natom']
  nacdr=QMout2['nacdr']
  nacdt=makecmatrix(nmstates,nmstates)
  for istate in range(nmstates):
    for jstate in range(nmstates):
      scal=complex(0.)
      for iatom in range(natom):
        for ixyz in range(3):
          scal+=veloc[iatom][ixyz]*nacdr[istate][jstate][iatom][ixyz]
      nacdt[istate][jstate]=scal
  QMout['nacdt']=nacdt
  return QMout

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
    outfile=open(outfilename,'w')
    outfile.write(string)
    outfile.close()
  except IOError:
    print 'Could not write QM output!'
    sys.exit(74)
  return

# ======================================================================= #
def cycleMOLPRO(QMin,Tasks):
  '''Iteratively writes MOLPRO input, calls MOLPRO (via runMOLPRO) and redoes the tasks list until the tasks list is empty. Renames the MOLPRO output files after each run. 

  Arguments:
  1 dictionary: QMin
  2 list of lists: task list

  Returns:
  1 integer: number of MOLPRO output files'''

  # Loop: write molpro input, run molpro, read molpro output, decide: ready or rewrite the Tasks array
  # Run until no jobs other than a bare restart are necessary
  outcounter=0
  while Tasks!=[]:
    writeMOLPROinput(Tasks, QMin)
    runerror=runMOLPRO(QMin)
    Tasks=redotasks(Tasks,QMin)
    printtasks(Tasks)
    outcounter+=1
    os.rename('MOLPRO.out','MOLPRO%04i.out' % (outcounter))
  if runerror!=0:
    print 'MOLPRO failed with unknown error!'
    sys.exit(75)
  if PRINT:
    string='  '+'='*40+'\n'
    string+='||'+' '*40+'||\n'
    string+='||'+' '*10+'All Tasks completed!'+' '*10+'||\n'
    string+='||'+' '*40+'||\n'
    string+='  '+'='*40+'\n\n'
    print string
  return outcounter

# ======================================================================= #
def checknac(QMin,QMout):
  '''Checks the results from DDR calculations for correctness. Obtains uncorrupted couplings via nac ana if possible. It also obtains wavefunction phases, even if CHECKNACS is disabled.

  In MOLPRO, the calculation of non-adiabatic couplings by means of the DDR procedure is very efficient. However, in the case of strong orbital mixing caused by intruder states this procedure yields highly incorrect values without any error message. In this routine, an intruder state is detected by means of the MRCI overlaps and the problem probably solved by calculating the couplings analytically. To this end, a new QMin dictionary is created, specifying the calculation of these couplings. After the calculation is finished, the coupling matrix is obtained from the scalar product of the velocity and the vector couplings.

  This check is engaged via the environment variable CHECKNACS. It only checks the results for "nac num" and "nac smat". In the former case, a correct matrix is constructed analytically, if velocities are availible, in the latter case an error message is printed and the dynamics aborted.

  Arguments:
  1 dictionary: QMin
  2 dictionary: QMout

  Returns:
  1 dictionary: QMout (including 'phases' and possibly with a corrected 'nac' matrix)'''

  #only if NACS are to be checked
  if QMin['CHECKNACS']:
    # in the case of numeric couplings, which are bad
    if 'nacdt' in QMin and not mrcioverlapsok(QMin,QMout):
      print 'MRCI overlaps seem to be bad. Most probably an intruder state messed up the active space...'
      if QMin['CORRECTNACS']:
        if 'veloc' in QMin:
          print 'Trying to obtain non-corrupted non-adiabatic couplings from "nac ana"...\n'
          # Generate a new QMin dictionary containing the new job, set up the tasks
          QMin_redoNac=redoNacjob(QMin)
          Tasks_redoNac=gettasks(QMin_redoNac)
          printtasks(Tasks_redoNac)
          # Run Molpro with this job until success
          outcounter=cycleMOLPRO(QMin_redoNac,Tasks_redoNac)
          # Extract analytical non-adiabatic couplings
          out_redoNac=catMOLPROoutput(outcounter)
          QMout_redoNac=getQMout(out_redoNac,QMin_redoNac)
          # Build the d/dt matrix from the couplings and the velocities
          QMout=contractNACveloc(QMin,QMout,QMout_redoNac)
        else:
          print 'No velocities availible. Aborting the dynamics because of corrupted non-adiabatic couplings!\n'
          sys.exit(76)
      else:
        print 'Screening couplings for bad values and set these to zero...\n'
        QMout['nacdt']=setnacszero(QMin,QMout)
    if 'overlap' in QMin and not mrcioverlapsok(QMin,QMout):
      print 'MRCI overlaps seem to be bad. Most probably an intruder state messed up the active space...'
      print 'Aborting the dynamics because of corrupted overlap matrix!\n'
      sys.exit(77)
  # finally, obtain the wavefunction phases from the mrcioverlaps
  if 'nacdt' in QMin:
    QMout['phases']=getphases(QMin,QMout)
    # "overcorrect" the NACs, so that SHARC can correct the phase
    for i in range(QMin['nmstates']):
      for j in range(QMin['nmstates']):
        QMout['nacdt'][i][j]/=(QMout['phases'][i]*QMout['phases'][j])
  return QMout

# ======================================================================= #
def cleanupSCRATCH(SCRATCHDIR):
  ''''''
  if PRINT:
    print '===> Removing SCRATCHDIR=%s\n' % (SCRATCHDIR)
  for data in os.listdir(SCRATCHDIR):
    path=SCRATCHDIR+'/'+data
    try:
      if DEBUG or PRINT:
        print 'rm %s' % (path)
      os.remove(path)
    except OSError:
      print 'Could not remove file from SCRATCHDIR: %s' % (path)
  try:
    if DEBUG or PRINT:
      print 'rm %s\n\n' % (SCRATCHDIR)
    os.rmdir(SCRATCHDIR)
  except OSError:
    print 'Could not remove SCRATCHDIR=%s' % (SCRATCHDIR)

# ========================== Main Code =============================== #
def main():
  '''This script realises an interface between the semi-classical dynamics code SHARC and the quantum chemistry program MOLPRO 2012. It allows the automatised calculation of non-relativistic and spin-orbit Hamiltonians, Dipole moments, gradients and non-adiabatic couplings at the CASSCF level of theory for an arbitrary number of states of different multiplicities. It also includes a small number of MOLPRO error handling capabilities (restarting non-converged calculations etc.).

  Input is realised through two files and a number of environment variables.

  QM.in:
    This file contains all information which are known to SHARC and which are independent of the used quantum chemistry code. This includes the current geometry and velocity, the number of states/multiplicities, the time step and the kind of quantities to be calculated.

  MOLPRO.template:
    This file is a minimal MOLPRO input file containing all molecule-specific parameters, like memory requirement, basis set, Douglas-Kroll-Hess transformation, active space and state-averaging. 

  Environment variables:
    Additional information, which are necessary to run MOLPRO, but which do not actually belong in a MOLPRO input file. 
    The necessary variables are:
      * QMEXE: is the path to the MOLPRO executable
      * SCRATCHDIR: is the path to a scratch directory for fast I/O Operations.
    Some optional variables are concerned with MOLPRO error handling (defaults in parenthesis):
      * GRADACCUDEFAULT: default accuracy for MOLPRO CPMCSCF (1e-7)
      * GRADACCUMAX: loosest allowed accuracy for MOLPRO CPMCSCF (1e-2)
      * GRADACCUSTEP: factor for decreasing the accuracy for MOLPRO CPMCSCF (1e-1)
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
    print 'PRINT or DEBUG environment variables do not evaluate to boolean values!'
    sys.exit(78)

  # Process Command line arguments
  if len(sys.argv)!=2:
    print 'Usage:\n./SHARC_MOLPRO.py <QMin>\n'
    print 'version:',version
    print 'date:',versiondate
    print 'changelog:\n',changelogstring
    sys.exit(79)
  QMinfilename=sys.argv[1]

  # Print header
  printheader()

  # Read QMinfile
  QMin=readQMin(QMinfilename)
  printQMin(QMin)

  # Process Tasks
  Tasks=gettasks(QMin)
  printtasks(Tasks)

  # Run MOLPRO until all jobs are done
  outcounter=cycleMOLPRO(QMin,Tasks)

  # Parse MOLPRO Output
  out=catMOLPROoutput(outcounter)
  QMout=getQMout(out,QMin)
  printQMout(QMin,QMout)

  # Calculate scalar product of velocity and NACs
  if 'nacdtfromdr' in QMin:
    QMout=contractNACveloc(QMin,QMout,QMout)
    QMout['phases']=getphases(QMin,QMout)
    #printQMout(QMin,QMout)

  # Check non-adiabatic couplings
  if 'nacdt' in QMin or 'overlap' in QMin:
    QMout=checknac(QMin,QMout)
    #printQMout(QMin,QMout)

  # Remove Scratchfiles from SCRATCHDIR
  cleanupSCRATCH(QMin['scratchdir'])

  # Measure time
  runtime=measuretime()
  QMout['runtime']=runtime

  # Write QMout
  writeQMout(QMin,QMout,QMinfilename)

  if PRINT or DEBUG:
    print '#================ END ================#'

if __name__ == '__main__':
    main()
