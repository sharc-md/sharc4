#!/usr/bin/env python2

#******************************************
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
#******************************************

#!/usr/bin/env python2

#    ====================================================================
#||                                                                       ||
#||                General Remarks                                        ||
#||                                                                       ||
#    ====================================================================
#
# This script ... 
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

# Modules:
# Operating system, isfile and related routines, move files, create directories
import os
# External Calls to MOLPRO
import subprocess as sp
# Command line arguments
import sys
# shell utilities like copy
import shutil
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
# gethostname routine
from socket import gethostname
# reading binary files
import struct
import copy
import ast
import string

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

version='0.1'
versiondate=datetime.date(2020,11,11)


changelogstring='''
14.01.2020:     Initial version 0.1
QM/MM single point calculation:
 - QM/MM setup from COBRAMM 
 - QM energy and gradient from SHARC interfaces
 - MM energy from AMBER
 - QM/MM energy from COMBRAMM (subtractive scheme + electrostatic embedding)

- File needed for COBRAMM: 
 - real.top
 - model-H.top
 - real_layers.xyz

'''

# ======================================================================= #
# holds the system time when the script was started
starttime=datetime.datetime.now()

# global variables for printing (PRINT gives formatted output, DEBUG gives raw output)
DEBUG=False
PRINT=True

NUMBERS = {'H':  1, 'He': 2,
'Li': 3, 'Be': 4, 'B':  5, 'C':  6,  'N': 7,  'O': 8, 'F':  9, 'Ne':10,
'Na':11, 'Mg':12, 'Al':13, 'Si':14,  'P':15,  'S':16, 'Cl':17, 'Ar':18,
'K': 19, 'Ca':20, 
'Sc':21, 'Ti':22, 'V':23, 'Cr':24, 'Mn':25, 'Fe':26, 'Co':27, 'Ni':28, 'Cu':29, 'Zn':30,
'Ga':31, 'Ge':32, 'As':33, 'Se':34, 'Br':35, 'Kr':36,
'Rb':37, 'Sr':38,
'Y':39,  'Zr':40, 'Nb':41, 'Mo':42, 'Tc':43, 'Ru':44, 'Rh':45, 'Pd':46, 'Ag':47, 'Cd':48,
'In':49, 'Sn':50, 'Sb':51, 'Te':52,  'I':53, 'Xe':54,
'Cs':55, 'Ba':56,
'La':57, 
'Ce':58, 'Pr':59, 'Nd':60, 'Pm':61, 'Sm':62, 'Eu':63, 'Gd':64, 'Tb':65, 'Dy':66, 'Ho':67, 'Er':68, 'Tm':69, 'Yb':70, 'Lu':71,
'Hf':72, 'Ta':73,  'W':74, 'Re':75, 'Os':76, 'Ir':77, 'Pt':78, 'Au':79, 'Hg':80,
'Tl':81, 'Pb':82, 'Bi':83, 'Po':84, 'At':85, 'Rn':86, 
'Fr':87, 'Ra':88,
'Ac':89, 
'Th':90, 'Pa':91,  'U':92, 'Np':93, 'Pu':94, 'Am':95, 'Cm':96, 'Bk':97, 'Cf':98, 'Es':99,'Fm':100,'Md':101,'No':102,'Lr':103,
        'Rf':104,'Db':105,'Sg':106,'Bh':107,'Hs':108,'Mt':109,'Ds':110,'Rg':111,'Cn':112,
'Nh':113,'Fl':114,'Mc':115,'Lv':116,'Ts':117,'Og':118
}

INTERFACES= [
     'MOLPRO',
     'COLUMBUS',
     'ANALYTICAL',
     'MOLCAS',
     'ADF',
     'TURBOMOLE',
     'GAUSSIAN',
     'ORCA',
     'BAGEL'
  ]

# conversion factors
au2a=0.529177211
rcm_to_Eh=4.556335e-6
kcal_to_Eh=0.0015936011

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
    sys.exit(12)
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
    sys.exit(13)

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

def link(PATH,NAME,crucial=True,force=True):
   # do not create broken links
   if not os.path.exists(PATH) and crucial:
       print 'Source %s does not exist, cannot create link!' % (PATH)
       sys.exit(95)
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
                   sys.exit(96)
               else:
                   return
   elif os.path.exists(NAME):
       # NAME is not a link. The interface will not overwrite files/directories with links, even with force=True
       print '%s exists, cannot create a link of the same name!' % (NAME)
       if crucial:
           sys.exit(97)
       else:
           return
   os.symlink(PATH, NAME)



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
  string+='||'+' '*27+'SHARC - COBRAMM - Interface'+' '*26+'||\n'
  string+='||'+' '*80+'||\n'
  string+='||'+' '*29+'Author: '+' Davide Avagliano '+' '*20+'||\n'
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


def printQMin(QMin):      ##### cambiare qua per il log che dipende da interfaccia

  if DEBUG:
    pprint.pprint(QMin)
  if not PRINT:
    return
  print '==> QM/MM Job description for:\n%s' % (QMin['comment'])

  string='Tasks: QM/MM calculation \n'
  
  string+='QM: SHARC/%s interface \n' % (QMin['template']['interface'])  
  string+='MM: COMBRAMM/AMBER interface \n'
  string+='QM/MM: COBRAMM'
  print string

  string='%s QM Job description in ./QM.log file \nCOBRAMM QM/MM Job detail in ./cobramm.log file' % (QMin['template']['interface'])
  print string





  string='Found Geo'
  if 'veloc' in QMin:
    string+=' and Veloc! '
  else:
    string+='! '
  string+='NAtom is %i.\n' % (QMin['natom'])
  print string

  string='\nGeometry in Bohrs:\n'
  if DEBUG:
    for i in range(QMin['natom']):
      string+='%2s ' % (QMin['geo'][i][0])
      for j in range(3):
        string+='% 7.4f ' % (QMin['geo'][i][j+1])
      string+='\n'
  else:
    for i in range(min(QMin['natom'],5)):
      string+='%2s ' % (QMin['geo'][i][0])
      for j in range(3):
        string+='% 7.4f ' % (QMin['geo'][i][j+1])
      string+='\n'
    if QMin['natom']>5:
      string+='..     ...     ...     ...\n'
      string+='%2s ' % (QMin['geo'][-1][0])
      for j in range(3):
        string+='% 7.4f ' % (QMin['geo'][-1][j+1])
      string+='\n'
  print string

  if 'veloc' in QMin and DEBUG:
    string=''
    for i in range(QMin['natom']):
      string+='%s ' % (QMin['geo'][i][0])
      for j in range(3):
        string+='% 7.4f ' % (QMin['veloc'][i][j])
      string+='\n'
    print string

 # if 'grad' in QMin:
#    string='Gradients requested:   '
 #   for i in range(1,QMin['nmstates']+1):
  #    if i in QMin['grad']:
 #       string+='X '
 #     else:
 #       string+='. '
 #   string+='\n'
 #   print string

  #if 'overlap' in QMin:
    #string='Overlaps:\n'
    #for i in range(1,QMin['nmstates']+1):
      #for j in range(1,QMin['nmstates']+1):
        #if [i,j] in QMin['overlap'] or [j,i] in QMin['overlap']:
          #string+='X '
        #else:
          #string+='. '
      #string+='\n'
    #print string

  for i in QMin:
    if not any( [i==j for j in ['geo','veloc','comment','LD_LIBRARY_PATH', 'grad','template','force_field'] ] ):
      if not any( [i==j for j in ['ionlist','ionmap'] ] ) or DEBUG:
        string=i+': '
        string+=str(QMin[i])
        print string
    else:
        string=i+': ...'
        print string
  print '\n'
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
# =========================================== output extraction ================================= #
# =============================================================================================== #
# =============================================================================================== #

# estrapolare cobramm log
def get_COBRAMMout(QMin):

    if PRINT:
         print '-----  Summary of QM/MM excited states calculation -----'

         with open (QMin['scratchdir']+'/QMMM/cobramm.log', 'r') as cobralog:
           states=QMin['states'][0]
           numstates=int(states)
      #nstates=QMin['nstates']
      #nmstates=QMin['nmstates']
      #natom=QMin['natom']
    
           line_number=0
           lines=cobralog.readlines()
           #print lines
           for line in lines:
             line_number+=1
             if 'QM/MM ENERGIES' in line:
               print "found QM/MM!!!!\n\n"
               print inizio,fine
               break
             inizio=line_number
             inizio+=-1
             fine=line_number
             fine+=5+numstates*12
           for i in range(inizio,fine):
             print lines[i]
    #return QMout
# =============================================================================================== #
# =============================================================================================== #
# =========================================== QMout writing ===================================== #
# =============================================================================================== #
# =============================================================================================== #

#  written by cobramm


# =============================================================================================== #
# =============================================================================================== #
# =========================================== SUBROUTINES TO readQMin =========================== #
# =============================================================================================== #
# =============================================================================================== #
def checkscratch(SCRATCHDIR):
  '''Checks whether SCRATCHDIR is a file or directory. If a file, it quits with exit code 1, if its a directory, it passes. If SCRATCHDIR does not exist, tries to create it.

  Arguments:
  1 string: path to SCRATCHDIR'''

  exist=os.path.exists(SCRATCHDIR)
  if exist:
    isfile=os.path.isfile(SCRATCHDIR)
    if isfile:
      print '$SCRATCHDIR=%s exists and is a file!' % (SCRATCHDIR)
      sys.exit(42)
  else:
    try:
      os.makedirs(SCRATCHDIR)
    except OSError:
      print 'Can not create SCRATCHDIR=%s\n' % (SCRATCHDIR)
      sys.exit(43)

# ======================================================================= #
def removequotes(string):
  if string.startswith("'") and string.endswith("'"):
    return string[1:-1]
  elif string.startswith('"') and string.endswith('"'):
    return string[1:-1]
  else:
    return string

# ======================================================================= #   ?
def getsh2cbmkey(sh2cbm,key):
  i=-1
  while True:
    i+=1
    try:
      line=re.sub('#.*$','',sh2cbm[i])
    except IndexError:
      break
    line=line.strip().split(None,1)
    if line==[]:
      continue
    if key.lower() in line[0].lower():
      return line
  return ['','']

# ======================================================================= #    ?
def get_sh2cbm_environ(sh2cbm,key,environ=True,crucial=True):
  line=getsh2cbmkey(sh2cbm,key)
  if line[0]:
    LINE=line[1]
  else:
    if environ:
      LINE=os.getenv(key.upper())
      if not LINE:
        print 'Either set $%s or give path to %s in SH2CBM.inp!' % (key.upper(),key.upper())
        if crucial:
          sys.exit(44)
        else:
          return None
    else:
      print 'Give path to %s in SH2CBM.inp!' % (key.upper())
      if crucial:
        sys.exit(45)
      else:
        return None
  LINE=os.path.expandvars(LINE)
  LINE=os.path.expanduser(LINE)
  LINE=os.path.abspath(LINE)
  LINE=removequotes(LINE).strip()
  if containsstring(';',LINE):
    print "$%s contains a semicolon. Do you probably want to execute another command after %s? I can't do that for you..." % (key.upper(),key.upper())
    sys.exit(46)
  return LINE

# ======================================================================= #      ??
def get_pairs(QMinlines,i):
  nacpairs=[]
  while True:
    i+=1
    try:
      line=QMinlines[i].lower()
    except IndexError:
      print '"keyword select" has to be completed with an "end" on another line!'
      sys.exit(47)
    if 'end' in line:
      break
    fields=line.split()
    try:
      nacpairs.append([int(fields[0]),int(fields[1])])
    except ValueError:
      print '"nacdr select" is followed by pairs of state indices, each pair on a new line!'
      sys.exit(48)
  return nacpairs,i

# =============================================================================================== #
# =============================================================================================== #
# =========================================== readQMin and gettasks ============================= #
# =============================================================================================== #
# =============================================================================================== #


  



# ======================================================================= #     OK
def readQMin(QMinfilename):
  '''Reads the time-step dependent information from QMinfilename. This file contains all information from the current SHARC job: geometry, velocity, number of states, requested quantities along with additional information. The routine also checks this input and obtains a number of environment variables necessary to run COLUMBUS.


  Steps are:
  - open and read QMinfilename
  - Obtain natom, comment, geometry (, velocity)

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
    sys.exit(49)
  QMin['natom']=natom
  if len(QMinlines)<natom+4:
    print 'Input file must contain at least:\nnatom\ncomment\ngeometry\nkeyword "states"\nat least one task'
    sys.exit(50)

  # Save Comment line
  QMin['comment']=QMinlines[1]

  # Get geometry and possibly velocity (for backup-analytical non-adiabatic couplings)
  QMin['geo']=[]
  QMin['veloc']=[]
  hasveloc=True
  for i in range(2,natom+2):
    if not containsstring('[a-zA-Z][a-zA-Z]?[0-9]*.*[-]?[0-9]+[.][0-9]*.*[-]?[0-9]+[.][0-9]*.*[-]?[0-9]+[.][0-9]*', QMinlines[i]):
      print 'Input file does not comply to xyz file format! Maybe natom is just wrong.'
      sys.exit(51)
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
    if 'savedirmm' in key:
      args=line.split()[1:]
    else:
      args=line.lower().split()[1:]
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
      sys.exit(52)
  else:
    factor=1./au2a

  for iatom in range(len(QMin['geo'])):
    for ixyz in range(3):
      QMin['geo'][iatom][ixyz+1]*=factor





  # ------------------------------------------ Resources: COBRAMM and AMBER  ----------------------------------


  # environment setup

  QMin['pwd']=os.getcwd()

  # open COBRAMM.resources
  filename='COBRAMM.resources'
  if os.path.isfile(filename):
    sh2cbm=readfile(filename)
  else:
    print 'HINT: reading resources from SH2CBM.inp'
    sh2cbm=readfile('SH2CBM.inp')

  # ncpus for SMP-parallel turbomole and wfoverlap
  # this comes before the turbomole path determination
  QMin['ncpu']=1
  line=getsh2cbmkey(sh2cbm,'ncpu')
  if line[0]:
    try:
      QMin['ncpu']=int(line[1])
      QMin['ncpu']=max(1,QMin['ncpu'])
    except ValueError:
      print 'Number of CPUs does not evaluate to numerical value!'
      sys.exit(66)
  os.environ['OMP_NUM_THREADS']=str(QMin['ncpu'])
  if QMin['ncpu']>1:
    os.environ['PARA_ARCH']='SMP'
    os.environ['PARNODES']=str(QMin['ncpu'])

  # set COBRAMM paths

  QMin['cobrammdir']=get_sh2cbm_environ(sh2cbm,'cobrammdir')
  os.environ['COBRAMM_PATH']=QMin['cobrammdir']
  os.environ['PATH']='%s:' % (QMin['cobrammdir']) +os.environ['PATH'] 

  # set AMBER paths
  
  QMin['amberdir']=get_sh2cbm_environ(sh2cbm,'amberdir')
  os.environ['AMBERHOME']=QMin['amberdir']
  os.environ['PATH']='%s:' % (QMin['amberdir']) +os.environ['PATH']
  os.environ['LD_LIBRARY_PATH']='%s:' % (QMin['amberdir'])+os.environ['LD_LIBRARY_PATH']

  # Set up scratchdir
  line=get_sh2cbm_environ(sh2cbm,'scratchdir',False,False)
  if line==None:
    line=QMin['pwd']+'/SCRATCHDIR/'
  line=os.path.expandvars(line)
  line=os.path.expanduser(line)
  line=os.path.abspath(line)
  #checkscratch(line)
  QMin['scratchdir']=line


  # Set up savedirmm
  if 'savedirmm' in QMin:
    # savedirmm may be read from QM.in file
    line=QMin['savedirmm'][0]
  else:
    line=get_sh2cbm_environ(sh2cbm,'savedirmm',False,False)
    if line==None:
      line=QMin['pwd']+'/SAVEDIRQMMM/'
  line=os.path.expandvars(line)
  line=os.path.expanduser(line)
  line=os.path.abspath(line)
  if 'init' in QMin:
    checkscratch(line)
  QMin['savedirmm']=line
  #link(QMin['savedirmm'],os.path.join(QMin['pwd'],'SAVE'),False,False)  #new 18.09.20

  # debug keyword in SH2CBM
  line=getsh2cbmkey(sh2cbm,'debug')
  if line[0]:
    if len(line)<=1 or 'true' in line[1].lower():
      global DEBUG
      DEBUG=True

  line=getsh2cbmkey(sh2cbm,'no_print')
  if line[0]:
    if len(line)<=1 or 'true' in line[1].lower():
      global PRINT
      PRINT=False


  # memory for only COBRAMM and AMBER, no QM
  QMin['memory']=100
  line=getsh2cbmkey(sh2cbm,'memory')
  if line[0]:
    try:
      QMin['memory']=int(line[1])
      QMin['memory']=max(100,QMin['memory'])
    except ValueError:
      print 'Run memory does not evaluate to numerical value!'
      sys.exit(67)
  else:
    print 'WARNING: Please set mm_memory in COBRAMM.resources (in MB)! Using 100 MB default value!'

  # memory QM
#  QMin['QMmemory']=100
#  line=getsh2cbmkey(sh2cbm,'qm_memory')
#  if line[0]:
#    try:
#      QMin['QMmemory']=int(line[1])
#      QMin['QMmemory']=max(100,QMin['qm_memory'])
#    except ValueError:
#      print 'Run memory does not evaluate to numerical value!'
#      sys.exit(67)
#  else:
#    print 'WARNING: Please set qm_memory in COBRAMM.resources (in MB)! Using 100 MB default value!'




  # -------------------------------------- COBRAMM Template ----------------------------------

  strings =['interface','jobtype',]
  #QMin['template']['jobtype']='sp'
  #QMin['template']['cut']='12'

  # open template
  template=readfile('COBRAMM.template')

  QMin['template']={}

  for line in template:
    line=re.sub('#.*$','',line).lower().split()
    if len(line)==0:
      continue
    elif line[0] in strings:
      QMin['template'][line[0]]=line[1]
    elif line[0] in sander:
      QMin['template'][line[0]]=line[1]

  
  necessary=['interface']
  for i in necessary:
	if not i in QMin['template']:
	  print 'Key %s missing in template file!' % (i)
          sys.exit(73)
 
 # make interface name correct
  for interface in INTERFACES:
    if QMin['template']['interface'].lower()==interface.lower():
      QMin['template']['interface']=interface
      break

  # checks:

  # find job type for cobramm
  allowed_jobtype=['sp','optxg','ts','irc','ci','freqxg']
  for t in allowed_jobtype:
    if QMin['template']['jobtype']==t:
      QMin['jobtype']=t
      break
  else:
    print 'Unknown job type "%s" for COBRAMM given in COBRAMM.template' % (QMin['template']['jobtype'])
    sys.exit(79)




# --------------------------------------------- logic checks ----------------------------------



  # Check the save directory
  try:
    ls=os.listdir(QMin['savedirmm'])
    err=0
  except OSError:
    err=1
  if 'init' in QMin:
    err=0
  elif 'overlap' in QMin:
    if 'newstep' in QMin:
      if not 'real.crd' in ls:
        print 'File "real.crd" missing in SAVEDIRQMMM!'
        err+=1
#      if not 'real.top' in ls:
#        print 'File "real.top" missing in SAVEDIR!'
#        err+=1
#      if not 'model-H.top' in ls:
#        print 'File "model-H.top" missing in SAVEDIR!'
#        err+=1
#      if not 'real_layers.xzy' in ls:
#        print 'File "real_layers.xyz" missing in SAVEDIR!'
#        err+=1

      for imult,nstates in enumerate(QMin['states']):
       if nstates<1:
        continue
       # if not 'dets.%i' % (imult+1) in ls:
       #   print 'File "dets.%i.old" missing in SAVEDIR!' % (imult+1)
       #   err+=1
       elif 'samestep' in QMin or 'restart' in QMin:
        if not 'real.crd.old' in ls:
         print 'File "real.crd" missing in SAVEDIRQMMM!'
         err+=1
       if err>0:  #        print '%i files missing in SAVEDIRQMMM=%s' % (err,QMin['savedirmm'])
        print '%i files missing in SAVEDIRQMMM=%s' % (err,QMin['savedirmm'])
        sys.exit(88)

  if PRINT:
    printQMin(QMin)

  return QMin



# =============================================================================================== #
# =============================================================================================== #
# =========================================== gettasks and setup routines ======================= #
# =============================================================================================== #
# =============================================================================================== #


def gettasks(QMin):
  ''''''

  tasks=[]
  # During initialization, create all temporary directories
  # and link them appropriately
  tasks.append(['mkdir', QMin['scratchdir']])
  tasks.append(['link', QMin['scratchdir'],QMin['pwd']+'/SCRATCH',False])
  tasks.append(['mkdir',QMin['scratchdir']+'/QMMM'])
  #tasks.append(['mkdir',QMin['scratchdir']+'/QM'])
  #if 'overlap' in QMin:
    #tasks.append(['mkdir',QMin['scratchdir']+'/QMMM/SAVE/'])
  #  tasks.append(['movetosave'])
  if not 'samestep' in QMin and not 'init' in QMin and not 'restart' in QMin:
    tasks.append(['movetoold'])
 
  if 'backupmm' in QMin:
    tasks.append(['mkdir',QMin['savedirmm']+'/backupmm/'])


    # calls: 
  #jobs=get_jobs(QMin)
  #for ijob,job in enumerate(jobs):
    #tasks.append(['prep_layers',job])
  tasks.append(['write_input'])
  tasks.append(['copyfile'])
  #tasks.append(['write_geom'])
  tasks.append(['writecrd'])
 #tasks.append(['write_geom',QMin['scratchdir']+'/QMMM'])
  tasks.append(['run_cobramm'])
  tasks.append(['save_data'])
  tasks.append(['getcobrammout'])
  tasks.append(['getqmout'])
  #if ijob==0:
  #  tasks.append(['save_data'])

  if 'backupmm' in QMin:
    tasks.append(['backupdata',QMin['backup']])

  if 'cleanup' in QMin:
    tasks.append(['cleanup',QMin['savedirmm']])
  #if not DEBUG:
    #tasks.append(['cleanup',QMin['scratchdir']])

  return tasks



# =============================================================================================== #
# =============================================================================================== #
# =========================================== SUBROUTINES TO RUNEVERYTING ======================= #
# =============================================================================================== #
# =============================================================================================== #

def mkdir(DIR):
    # mkdir the DIR, or clean it if it exists
    if os.path.exists(DIR):
        if os.path.isfile(DIR):
            print '%s exists and is a file!' % (DIR)
            sys.exit(89)
        elif os.path.isdir(DIR):
            if DEBUG:
                print 'Remake\t%s' % DIR
            shutil.rmtree(DIR)
            os.makedirs(DIR)
    else:
        try:
            if DEBUG:
                print 'Make\t%s' % DIR
            os.makedirs(DIR)
        except OSError:
            print 'Can not create %s\n' % (DIR)
            sys.exit(90)

# ======================================================================= #
def link(PATH,NAME,crucial=True,force=True):
  # do not create broken links
  if not os.path.exists(PATH):
    print 'Source %s does not exist, cannot create link!' % (PATH)
    sys.exit(91)
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
          sys.exit(92)
        else:
          return
  elif os.path.exists(NAME):
    # NAME is not a link. The interface will not overwrite files/directories with links, even with force=True
    print '%s exists, cannot create a link of the same name!' % (NAME)
    if crucial:
      sys.exit(93)
    else:
      return
  os.symlink(PATH, NAME)



 # ======================================================================= #

def write_input(QMin):
  string=''
  string+='!keyword \n type=%s qm-type=sharc '  % (QMin['template']['jobtype'])
  string+='nproc=%i ' % (QMin['ncpu'])
  #string+='geomem= %i' % (QMin['QMmemory'])
  string+='sharc-qm=%s \n' % (QMin['template']['interface'])
  string+='?keyword \n \n'
  string+='!sander\n mm\n &cntrl\n imin   = 1,\n maxcyc = 0,\n ntb    = 0,\n igb    = 0,\n ntr    = 0,\n ibelly = 1,\n ntxo   = 1,\n cut    = 10 ,' 
  string+='\n/'
  string+='\n?sander\n'
  filename=os.path.join(QMin['scratchdir']+'/QMMM','cobram.command')
  writefile(filename, string)
  print 'command written'
 ##return string


# ======================================================================= #
def shorten_DIR(string):
    maxlen=40
    front=12
    if len(string)>maxlen:
        return string[0:front]+'...'+string[-(maxlen-3-front):]
    else:
        return string+' '*(maxlen-len(string))

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
def movetoold(QMin):
  # rename all files in savedir
  saveable=['real.crd']
  savedirmm=QMin['savedirmm']
  ls=os.listdir(savedirmm)
  if ls==[]:
    return
  for f in ls:
    f2=savedirmm+'/'+f
    if os.path.isfile(f2):
      if any( [ i in f for i in saveable ] ):
        if not 'old' in f:
          fdest=f2+'.old'
          shutil.move(f2,fdest)
  #if 'overlap' in QMin:
  #  saved=QMin['pwd']+'/SAVE'
  #  ls_save=saved #os.listdir(QMin['pwd']+'/SAVE/')
  #  print ls_save
  #  for files in ls_save:
  #      ftomove=saved+'/'+files
  #      fin=QMin['scratchdir']+'/QMMM/SAVE/'+files
  #      shutil.copy(ftomove,fin)
# ======================================================================  #

# copia file per cobramm
def copy_file(QMin):
  currentdir=os.getcwd()
  intername=string.upper(QMin['template']['interface'])
  interface_templ='%s.template' % intername
  interface_res= '%s.resources' % intername
  tocopy=['real.top', 'model-H.top', 'real_layers.xyz']
  moretocopy=[interface_templ, interface_res]
  for files in tocopy:
    fromfile=os.path.join(currentdir, files)
    tofile=os.path.join(QMin['scratchdir'], 'QMMM', files)
    shutil.copy(fromfile,tofile)
  for morefiles in moretocopy:
    morefromfile=os.path.join(currentdir, morefiles)
    moretofile=os.path.join(QMin['scratchdir'], 'QMMM', morefiles)
    shutil.copy(morefromfile,moretofile)
  fromqm=os.path.join(currentdir, 'QM.in')
  toqmmm=os.path.join(QMin['scratchdir'], 'QMMM', 'QMMM.in')
  shutil.copy(fromqm,toqmmm)
  

# ======================================================================= #
def save_data(QMin):
  # copy files to savedir
  saveable=['real.crd']
  for i in saveable:
    fromfile=os.path.join(QMin['scratchdir']+'/QMMM',i)
    tofile  =os.path.join(QMin['savedirmm'],i)
    shutil.copy(fromfile,tofile)
  #if 'init' in QMin:
  #   #link(QMin['savedirmm'],os.path.join(QMin['pwd'],'SAVE'),False,False)  #new 18.09.20
  #   ls=os.listdir(QMin['scratchdir']+'/QMMM/SAVEDIRQMMM/')
  #   for files in ls:
  #      ftomove=os.path.join(QMin['scratchdir']+'/QMMM/SAVEDIRQMMM/'+files)
  #      fin=os.path.join(QMin['savedirmm']+'/'+files)
  #      shutil.copy(ftomove,fin)

# ======================================================================= #
def copy_qmfiles(QMin):
  log='QM.log'
  filelog=os.path.join(QMin['scratchdir'], 'QMMM', log)
  outlog=os.path.join(QMin['pwd'], log)
  shutil.copy(filelog,outlog)

  oldout='QMMM.out'
  newout='QM.out'
  oldfileout=os.path.join(QMin['scratchdir'], 'QMMM', oldout)
  newfileout=os.path.join(QMin['pwd'],newout)
  shutil.copy(oldfileout,newfileout)



# ========================================================================  #

def write_crd(QMin):
  factor=au2a
  fname=QMin['scratchdir']+'/QMMM/real.crd'
  string='crd format \n %i\n' % (QMin['natom'])
  counter=1
  for atom in QMin['geo']:
    for xyz in range(1,4):
      string+='%12.7f' % (atom[xyz]*factor)
    if counter %2 == 0:
      string+='\n'
    counter+=1
  writefile(fname,string)
  print 'real.crd written'



# ======================================================================= #
def runProgram(string,workdir):
  prevdir=os.getcwd()
  if DEBUG:
    print workdir
  os.chdir(workdir)
  if PRINT or DEBUG:
    starttime=datetime.datetime.now()
    sys.stdout.write('%s\n\t%s' % (string,starttime)) 
    sys.stdout.flush()
  try:
    runerror=sp.call(string,shell=True)
  except OSError:
    print 'Call have had some serious problems:',OSError
    sys.exit(81)
  if PRINT or DEBUG:
    endtime=datetime.datetime.now()
    sys.stdout.write('\t%s\t\tRuntime: %s\t\tError Code: %i\n\n' % (endtime,endtime-starttime,runerror))
  os.chdir(prevdir)
  return runerror





# ======================================================================= #
def run_cobramm(QMin):
  workdir=os.path.join(QMin['scratchdir'],'QMMM')
  string='cobram.py > cobramm.log'
  runerror=runProgram(string,workdir)
  print 'COBRAMM QM/MM setup and calculation started:'
  if runerror!=0:
    print 'COBRAMM calculation crashed! Error code=%i' % (runerror)
    sys.exit(99)

  return

# ======================================================================= #
def backupdata(backupdir,QMin):
  # save all files in savedir, except which have 'old' in their name
  ls=os.listdir(QMin['savedirmm'])
  for f in ls:
    ff=QMin['savedirmm']+'/'+f
    if os.path.isfile(ff) and not 'old' in ff:
      fdest=backupdir+'/'+f
      shutil.copy(ff,fdest)



# ======================================================================= #
def runeverything(tasks, QMin):
  
  if PRINT or DEBUG:
    print '=============> Entering RUN section <=============\n\n'

  QMout={}
  for task in tasks:
    if DEBUG:
      print task
    if task[0]=='copyfile':
      copy_file(QMin)
    #if task[0]=='save_data':
    #  movetoold(QMin)
    if task[0]=='mkdir':
      mkdir(task[1])
    if task[0]=='link':
      if len(task)==4:
        link(task[1],task[2],task[3])
      else:
        link(task[1],task[2])
    #if task[0]=='movetosave':
    #  movetoold(QMin)
    #if task[0]=='movetoold':
    #  save_data(QMin)
    #if task[0]=='backupdata':
      #backupdata(task[1],QMin)
    if task[0]=='cleanup':
      cleandir(task[1])
    if task[0]=='writecrd':
      write_crd(QMin)
    if task[0]=='save_data':
      save_data(QMin)
    if task[0]=='movetoold':
      movetoold(QMin)
    if task[0]=='write_input':
      write_input(QMin)
    if task[0]=='run_cobramm':
      run_cobramm(QMin)
    if task[0]=='getcobrammout':
       get_COBRAMMout(QMin)   # da scrivere ancora
    if task[0]=='getqmout':
       copy_qmfiles(QMin)  


  return QMin,QMout



# ======================================================================= #
# ======================================================================= #
# ======================================================================= #






# ========================== Main Code =============================== #
def main():

  # Retrieve PRINT and DEBUG
  try:
    envPRINT=os.getenv('SH2CBM_PRINT')
    if envPRINT and envPRINT.lower()=='false':
      global PRINT
      PRINT=False
    envDEBUG=os.getenv('SH2CBM_DEBUG')
    if envDEBUG and envDEBUG.lower()=='true':
      global DEBUG
      DEBUG=True
  except ValueError:
    print 'PRINT or DEBUG environment variables do not evaluate to logical values!'
    sys.exit(110)

  # Process Command line arguments
  if len(sys.argv)!=2:
    print 'Usage:\n./SHARC_COMBRAMM.py <QMin>\n'
    print 'version:',version
    print 'date:',versiondate
    print 'changelog:\n',changelogstring
    sys.exit(111)
  QMinfilename=sys.argv[1]

  # Print header
  printheader()

  # Read QMinfile
  QMin=readQMin(QMinfilename)

  # Process Tasks
  Tasks=gettasks(QMin)
  if DEBUG:
    pprint.pprint(Tasks)

  # do all runs
  QMin,QMout=runeverything(Tasks,QMin)

  #printQMout(QMin,QMout)

  # Measure time
  #runtime=measuretime()
  #QMout['runtime']=runtime

  # Write QMout
  #writeQMout(QMin,QMout,QMinfilename)

  if PRINT or DEBUG:
    print datetime.datetime.now()
    print '#================ END ================#'

if __name__ == '__main__':
    main()
