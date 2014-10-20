#!/usr/bin/env python2

# Interactive script for the setup of initial condition excitation calculations for SHARC
# 
# usage: python setup_init.py

import copy
import math
import sys
import re
import os
import stat
import shutil
import datetime
from optparse import OptionParser
import readline
import time

# =========================================================
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

# some constants
DEBUG = False
CM_TO_HARTREE = 1./219474.6     #4.556335252e-6 # conversion factor from cm-1 to Hartree
HARTREE_TO_EV = 27.211396132    # conversion factor from Hartree to eV
U_TO_AMU = 1./5.4857990943e-4            # conversion from g/mol to amu
BOHR_TO_ANG=0.529177211
PI = math.pi

version='1.0'
versionneeded=[0.2, 1.0]
versiondate=datetime.date(2014,10,8)


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

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def try_read(l,index,typefunc,default):
  try:
    return typefunc(l[index])
  except IndexError:
    return typefunc(default)
  except ValueError:
    print 'Could not initialize object!'
    quit(1)

# ======================================================================================================================

class ATOM:
  def __init__(self,symb='??',num=0.,coord=[0.,0.,0.],m=0.,veloc=[0.,0.,0.]):
    self.symb  = symb
    self.num   = num
    self.coord = coord
    self.mass  = m
    self.veloc = veloc
    self.Ekin=0.5*self.mass * sum( [ self.veloc[i]**2 for i in range(3) ] )

  def init_from_str(self,initstring=''):
    f=initstring.split()
    self.symb  =   try_read(f,0,str,  '??')
    self.num   =   try_read(f,1,float,0.)
    self.coord = [ try_read(f,i,float,0.) for i in range(2,5) ]
    self.mass  =   try_read(f,5,float,0.)*U_TO_AMU
    self.veloc = [ try_read(f,i,float,0.) for i in range(6,9) ]
    self.Ekin=0.5*self.mass * sum( [ self.veloc[i]**2 for i in range(3) ] )

  def __str__(self):
    s ='%2s % 5.1f '               % (self.symb, self.num)
    s+='% 12.8f % 12.8f % 12.8f '  % tuple(self.coord)
    s+='% 12.8f '                  % (self.mass/U_TO_AMU)
    s+='% 12.8f % 12.8f % 12.8f'   % tuple(self.veloc)
    return s

  def EKIN(self):
    self.Ekin=0.5*self.mass * sum( [ self.veloc[i]**2 for i in range(3) ] )
    return self.Ekin

  def geomstring(self):
    s='  %2s % 5.1f % 12.8f % 12.8f % 12.8f % 12.8f' % (self.symb,self.num,self.coord[0],self.coord[1],self.coord[2],self.mass/U_TO_AMU)
    return s

  def velocstring(self):
    s=' '*11+'% 12.8f % 12.8f % 12.8f' % tuple(self.veloc)
    return s

# ======================================================================================================================

class STATE:
  def __init__(self,i=0,e=0.,eref=0.,dip=[0.,0.,0.]):
    self.i       = i
    self.e       = e.real
    self.eref    = eref.real
    self.dip     = dip
    self.Excited = False
    self.Eexc    = self.e-self.eref
    self.Fosc    = (2./3.*self.Eexc*sum( [i*i.conjugate() for i in self.dip] ) ).real
    if self.Eexc==0.:
      self.Prob  = 0.
    else:
      self.Prob  = self.Fosc/self.Eexc**2

  def init_from_str(self,initstring):
    f=initstring.split()
    self.i       =   try_read(f,0,int,  0 )
    self.e       =   try_read(f,1,float,0.)
    self.eref    =   try_read(f,2,float,0.)
    self.dip     = [ try_read(f,i,float,0.) for i in range(3,6) ]
    self.Excited =   try_read(f,2,bool, False)
    self.Eexc    = self.e-self.eref
    self.Fosc    = (2./3.*self.Eexc*sum( [i*i.conjugate() for i in self.dip] ) ).real
    if self.Eexc==0.:
      self.Prob  = 0.
    else:
      self.Prob  = self.Fosc/self.Eexc**2

  def __str__(self):
    s ='%03i % 18.10f % 18.10f ' % (self.i,self.e,self.eref)
    for i in range(3):
      s+='% 12.8f % 12.8f ' % (self.dip[i].real,self.dip[i].imag)
    s+='% 12.8f % 12.8f %s' % (self.Eexc*HARTREE_TO_EV,self.Fosc,self.excited)
    return s

  def Excite(self,max_Prob,erange):
    try:
      Prob=self.Prob/max_Prob
    except ZeroDivisionError:
      Prob=-1.
    if not (erange[0] <= self.Eexc <= erange[1]):
      Prob=-1.
    self.excited=(random.random() < Prob)

# ======================================================================================================================

class INITCOND:
  def __init__(self,atomlist=[],eref=0.,epot_harm=0.):
    self.atomlist=atomlist
    self.eref=eref
    self.Epot_harm=epot_harm
    self.natom=len(atomlist)
    self.Ekin=sum( [atom.Ekin for atom in self.atomlist] )
    self.statelist=[]
    self.nstate=0
    self.Epot=epot_harm

  def addstates(self,statelist):
    self.statelist=statelist
    self.nstate=len(statelist)
    self.Epot=self.statelist[0].e-self.eref

  def init_from_file(self,f,eref,index):
    while True: 
      line=f.readline()
      if 'Index     %i' % (index) in line:
        break
      if line=='\n':
        continue
      if line=='':
        print 'Initial condition %i not found in file %s' % (index,f.name)
        quit(1)
    f.readline()        # skip one line, where "Atoms" stands
    atomlist=[]
    while True:
      line=f.readline()
      if 'States' in line:
        break
      atom=ATOM()
      atom.init_from_str(line)
      atomlist.append(atom)
    statelist=[]
    while True:
      line=f.readline()
      if 'Ekin' in line:
        break
      state=STATE()
      state.init_from_str(line)
      statelist.append(state)
    epot_harm=0.
    while not line=='\n' and not line=='':
      line=f.readline()
      if 'epot_harm' in line.lower():
        epot_harm=float(line.split()[1])
        break
    self.atomlist=atomlist
    self.eref=eref
    self.Epot_harm=epot_harm
    self.natom=len(atomlist)
    self.Ekin=sum( [atom.Ekin for atom in self.atomlist] )
    self.statelist=statelist
    self.nstate=len(statelist)
    if self.nstate>0:
      self.Epot=self.statelist[0].e-self.eref
    else:
      self.Epot=epot_harm

  def __str__(self):
    s='Atoms\n'
    for atom in self.atomlist:
      s+=str(atom)+'\n'
    s+='States\n'
    for state in self.statelist:
      s+=str(state)+'\n'
    s+='Ekin      % 16.12f a.u.\n' % (self.Ekin)
    s+='Epot_harm % 16.12f a.u.\n' % (self.Epot_harm)
    s+='Epot      % 16.12f a.u.\n' % (self.Epot)
    s+='Etot_harm % 16.12f a.u.\n' % (self.Epot_harm+self.Ekin)
    s+='Etot      % 16.12f a.u.\n' % (self.Epot+self.Ekin)
    s+='\n\n'
    return s

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def check_initcond_version(string,must_be_excited=False):
  if not 'sharc initial conditions file' in string.lower():
    return False
  f=string.split()
  for i,field in enumerate(f):
    if 'version' in field.lower():
      try:
        v=float(f[i+1])
        if not v in versionneeded:
          return False
      except IndexError:
        return False
  if must_be_excited:
    if not 'excited' in string.lower():
      return False
  return True


# ======================================================================================================================

def centerstring(string,n,pad=' '):
  l=len(string)
  if l>=n:
    return string
  else:
    return  pad*((n-l+1)/2)+string+pad*((n-l)/2)

# ======================================================================================================================

def displaywelcome():
  print 'Script for setup of initial conditions started...\n'
  string='\n'
  string+='  '+'='*80+'\n'
  string+='||'+centerstring('',80)+'||\n'
  string+='||'+centerstring('Setup initial conditions for SHARC dynamics',80)+'||\n'
  string+='||'+centerstring('',80)+'||\n'
  string+='||'+centerstring('Author: Sebastian Mai',80)+'||\n'
  string+='||'+centerstring('',80)+'||\n'
  string+='||'+centerstring('Version:'+version,80)+'||\n'
  string+='||'+centerstring(versiondate.strftime("%d.%m.%y"),80)+'||\n'
  string+='||'+centerstring('',80)+'||\n'
  string+='  '+'='*80+'\n\n'
  string+='''
This script automatizes the setup of excited-state calculations for initial conditions 
for SHARC dynamics. 
  '''
  print string

# ======================================================================================================================

def open_keystrokes():
  global KEYSTROKES
  KEYSTROKES=open('KEYSTROKES.tmp','w')

def close_keystrokes():
  KEYSTROKES.close()
  shutil.move('KEYSTROKES.tmp','KEYSTROKES.setup_init')

# ===================================

def question(question,typefunc,default=None,autocomplete=True):
  if typefunc==int or typefunc==float:
    if not default==None and not isinstance(default,list):
      print 'Default to int or float question must be list!'
      quit(1)
  if typefunc==str and autocomplete:
    readline.set_completer_delims(' \t\n;')
    readline.parse_and_bind("tab: complete")    # activate autocomplete
  else:
    readline.parse_and_bind("tab: ")            # deactivate autocomplete

  while True:
    s=question
    if default!=None:
      if typefunc==bool or typefunc==str:
        s+= ' [%s]' % (str(default))
      elif typefunc==int or typefunc==float:
        s+= ' ['
        for i in default:
          s+=str(i)+' '
        s=s[:-1]+']'
    if typefunc==str and autocomplete:
      s+=' (autocomplete enabled)'
    s+=' '

    line=raw_input(s)
    line=re.sub('#.*$','',line).strip()
    if not typefunc==str:
      line=line.lower()

    if line=='' or line=='\n':
      if default!=None:
        KEYSTROKES.write(line+' '*(40-len(line))+' #'+s+'\n')
        return default
      else:
        continue

    if typefunc==bool:
      posresponse=['y','yes','true', 'ja',  'si','yea','yeah','aye','sure','definitely']
      negresponse=['n','no', 'false','nein',     'nope']
      if line in posresponse:
        KEYSTROKES.write(line+' '*(40-len(line))+' #'+s+'\n')
        return True
      elif line in negresponse:
        KEYSTROKES.write(line+' '*(40-len(line))+' #'+s+'\n')
        return False
      else:
        print 'I didn''t understand you.'
        continue

    if typefunc==str:
      KEYSTROKES.write(line+' '*(40-len(line))+' #'+s+'\n')
      return line

    if typefunc==int or typefunc==float:
      # int and float will be returned as a list
      f=line.split()
      try:
        for i in range(len(f)):
          f[i]=typefunc(f[i])
        KEYSTROKES.write(line+' '*(40-len(line))+' #'+s+'\n')
        return f
      except ValueError:
        if typefunc==int:
          i=1
        elif typefunc==float:
          i=2
        print 'Please enter a %s' % ( ['string','integer','float'][i] )
        continue



# ======================================================================================================================

def get_general():
  '''This routine questions from the user some general information:
  - initconds file
  - number of states
  - number of initial conditions
  - interface to use'''
  INFOS={}

  print centerstring('Initial conditions file',60,'-')+'\n'
  # open the initconds file
  try:
    initfile='initconds'
    initf=open(initfile)
    line=initf.readline()
    if check_initcond_version(line):
      print 'Initial conditions file "initconds" detected. Do you want to use this?'
      if not question('Use file "initconds"?',bool,True):
        initf.close()
        raise IOError
    else:
      initf.close()
      raise IOError
  except IOError:
    print '\nIf you do not have an initial conditions file, prepare one with wigner.py!\n'
    print 'Please enter the filename of the initial conditions file.'
    while True:
      initfile=question('Initial conditions filename:',str,'initconds')
      initfile=os.path.expanduser(os.path.expandvars(initfile))
      if os.path.isdir(initfile):
        print 'Is a directory: %s' % (initfile)
        continue
      if not os.path.isfile(initfile):
        print 'File does not exist: %s' % (initfile)
        continue
      try:
        initf=open(initfile,'r')
      except IOError:
        print 'Could not open: %s' % (initfile)
        continue
      line=initf.readline()
      if check_initcond_version(line):
        break
      else:
        print 'File does not contain initial conditions!'
        continue
  # read the header
  ninit=int(initf.readline().split()[1])
  natom=int(initf.readline().split()[1])
  INFOS['ninit']=ninit
  INFOS['natom']=natom
  initf.seek(0)                 # rewind the initf file
  INFOS['initf']=initf
  print '\nFile "%s" contains %i initial conditions.' % (initfile,ninit)
  print 'Number of atoms is %i\n' % (natom)



  print centerstring('Range of initial conditions',60,'-')
  print '\nPlease enter the range of initial conditions for which an excited-state calculation should be performed as two integers separated by space.'
  while True:
    irange=question('Initial condition range:',int,[1,ninit])
    if len(irange)!=2:
      print 'Enter two numbers separated by spaces!'
      continue
    if irange[0]>irange[1]:
      print 'Range empty!'
      continue
    if irange[0]==irange[1]==0:
      print 'Only preparing calculation at equilibrium geometry!'
      break
    if irange[1]>ninit:
      print 'There are only %i initial conditions in file %s!' % (ninit,initfile)
      continue
    if irange[0]<=0:
      print 'Only positive indices allowed!'
      continue
    break
  print '\nScript will use initial conditions %i to %i (%i in total).\n' % (irange[0],irange[1],irange[1]-irange[0]+1)
  INFOS['irange']=irange



  print centerstring('Number of states',60,'-')
  print '\nPlease enter the number of states as a list of integers\ne.g. 3 0 3 for three singlets, zero doublets and three triplets.'
  while True:
    states=question('Number of states:',int)
    if len(states)==0:
      continue
    if any(i<0 for i in states):
      print 'Number of states must be positive!'
      continue
    break
  print ''
  nstates=0
  for mult,i in enumerate(states):
    nstates+=(mult+1)*i
  print 'Number of states: '+str(states)
  print 'Total number of states: %i\n' % (nstates)
  soc=question('Spin-Orbit calculation?',bool,True)
  if soc:
    if len(states)>1:
      print 'Will calculate spin-orbit matrix.'
    else:
      print 'Not calculating spin-orbit matrix, only singlets specified.'
      soc=False
  print ''
  INFOS['states']=states
  INFOS['nstates']=nstates
  INFOS['soc']=soc



  print centerstring('Choose the quantum chemistry interface',60,'-')
  print '\nPlease specify the quantum chemistry interface (enter any of the following numbers):'
  for i in Interfaces:
    print '%i\t%s' % (i, Interfaces[i]['description'])
  print ''
  while True:
    num=question('Interface number:',int)[0]
    if num in Interfaces:
      break
    else:
      print 'Please input one of the following: %s!' % ([i for i in Interfaces])
  INFOS['interface']=num



  INFOS['cwd']=os.getcwd()
  print ''
  return INFOS

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

Interfaces={
  1: {'script':          'SHARC_MOLPRO.py',
      'description':     'MOLPRO (only CASSCF)',
      'get_routine':     'get_MOLPRO',
      'prepare_routine': 'prepare_MOLPRO',
      'couplings':       [1,2,3]
     },
  2: {'script':          'SHARC_COLUMBUS.py',
      'description':     'COLUMBUS (CASSCF, RASSCF and MRCISD), using SEWARD integrals',
      'get_routine':     'get_COLUMBUS',
      'prepare_routine': 'prepare_COLUMBUS',
      'couplings':       [3]
     },
  3: {'script':          'SHARC_Analytical.py',
      'description':     'Analytical PESs',
      'get_routine':     'get_Analytical',
      'prepare_routine': 'prepare_Analytical',
      'couplings':       [3]
     },
  4: {'script':          'SHARC_MOLCAS.py',
      'description':     'MOLCAS (only CASSCF)',
      'get_routine':     'get_MOLCAS',
      'prepare_routine': 'prepare_MOLCAS',
      'couplings':       [3]
     }
  #4: {'script':          'SHARC_MOLCAS_QMMM.py',
      #'description':     'MOLCAS (with QM/MM)',
      #'get_routine':     'get_MOLCAS_QMMM',
      #'prepare_routine': 'prepare_MOLCAS_QMMM',
      #'couplings':       []
     #},
  }




def checktemplate_MOLPRO(filename):
  necessary=['memory','basis','closed','occ','wf','state']
  try:
    f=open(filename)
    data=f.readlines()
    f.close()
  except IOError:
    print 'Could not open template file %s' % (filename)
    return False
  i=0
  for l in data:
    if necessary[i] in l:
      i+=1
      if i+1==len(necessary):
        return True
  print 'The template %s seems to be incomplete! It should contain: ' % (filename) +str(necessary)
  return False

# =================================================

def get_MOLPRO(INFOS):
  '''This routine asks for all questions specific to MOLPRO:
  - path to molpro
  - scratch directory
  - MOLPRO.template
  - wf.init
  '''

  string='\n  '+'='*80+'\n'
  string+='||'+centerstring('MOLPRO Interface setup',80)+'||\n'
  string+='  '+'='*80+'\n\n'
  print string

  print centerstring('Path to MOLPRO',60,'-')+'\n'
  path=os.getenv('MOLPRO')
  path=os.path.expanduser(os.path.expandvars(path))
  if not path=='':
    if not path.endswith('/molpro'):
      path='$MOLPRO/molpro'
    else:
      path='$MOLPRO/'
  else:
    path=None
  #if path!='':
    #print 'Environment variable $MOLPRO detected:\n$MOLPRO=%s\n' % (path)
    #if question('Do you want to use this MOLPRO installation?',bool,True):
      #INFOS['molpro']=path
  #if not 'molpro' in INFOS:
  print '\nPlease specify path to MOLPRO directory (SHELL variables and ~ can be used, will be expanded when interface is started).\n'
  INFOS['molpro']=question('Path to MOLPRO executable:',str,path)
  print ''


  print centerstring('Scratch directory',60,'-')+'\n'
  print 'Please specify an appropriate scratch directory. This will be used to temporally store the integrals. The scratch directory will be deleted after the calculation. Remember that this script cannot check whether the path is valid, since you may run the calculations on a different machine. The path will not be expanded by this script.'
  INFOS['scratchdir']=question('Path to scratch directory:',str)
  print ''


  print centerstring('MOLPRO input template file',60,'-')+'\n'
  print '''Please specify the path to the MOLPRO.template file. This file must be a valid MOLPRO input file for a CASSCF calculation. It should contain the following settings:
- memory settings
- Basis set (possibly also Douglas-Kroll settings etc.)
- CASSCF calculation with:
  * Number of frozen, closed and occupied orbitals
  * wf and state cards for the specification of the wavefunction
MOLPRO.template files can easily be created using molpro_input.py (Open a second shell if you need to create one now).

The MOLPRO interface will generate the remaining MOLPRO input automatically.
'''
  if os.path.isfile('MOLPRO.template'):
    if checktemplate_MOLPRO('MOLPRO.template'):
      print 'Valid file "MOLPRO.template" detected. '
      usethisone=question('Use this template file?',bool,True)
      if usethisone:
        INFOS['molpro.template']='MOLPRO.template'
  if not 'molpro.template' in INFOS:
    while True:
      filename=question('Template filename:',str)
      if not os.path.isfile(filename):
        print 'File %s does not exist!' % (filename)
        continue
      if checktemplate_MOLPRO(filename):
        break
    INFOS['molpro.template']=filename
  print ''


  print centerstring('Initial wavefunction: MO Guess',60,'-')+'\n'
  print '''Please specify the path to a MOLPRO wavefunction file containing suitable starting MOs for the CASSCF calculation. Please note that this script cannot check whether the wavefunction file and the Input template are consistent!

If you optimized your geometry with MOLPRO/CASSCF you can reuse the "wf" file from the optimization.
'''
  if question('Do you have an initial wavefunction file?',bool,True):
    while True:
      filename=question('Initial wavefunction file:',str,'wf.init')
      if os.path.isfile(filename):
        break
      else:
        print 'File not found!'
    INFOS['molpro.guess']=filename
  else:
    print 'WARNING: Remember that CASSCF calculations may run very long and/or yield wrong results without proper starting MOs.'
    time.sleep(2)
    INFOS['molpro.guess']=False

  return INFOS



# ======================================================================================================================

def checktemplate_COLUMBUS(TEMPLATE, mult):
  '''Checks whether TEMPLATE is a file or directory. If a file or does not exist, it quits with exit code 1, if it is a directory, it checks whether all important input files are there. Does not check for all input files, since runc does this, too.

  Arguments:
  1 string: path to TEMPLATE

  returns whether input is for isc keyword or socinr keyword
  and returns the DRT of the given multiplicity'''

  exist=os.path.exists(TEMPLATE)
  if exist:
    isfile=os.path.isfile(TEMPLATE)
    if isfile:
      #print 'TEMPLATE=%s exists and is a file!' % (TEMPLATE)
      return None,None
    necessary=['control.run','mcscfin','molcas.input','tranin','propin']
    lof=os.listdir(TEMPLATE)
    for i in necessary:
      if not i in lof:
        #print 'Did not find input file %s! Did you prepare the input according to the instructions?' % (i)
        return None,None
    cidrtinthere=False
    ciudginthere=False
    for i in lof:
      if 'cidrtin' in i:
        cidrtinthere=True
      if 'ciudgin' in i:
        ciudginthere=True
    if not cidrtinthere or not ciudginthere:
      #print 'Did not find input file %s.*! Did you prepare the input according to the instructions?' % (i)
      return None,None
  else:
    #print 'Directory %s does not exist!' % (TEMPLATE)
    return None,None

  # check cidrtin and cidrtin* for the multiplicity
  try:
    cidrtin=open(TEMPLATE+'/cidrtin')
    line=cidrtin.readline().split()
    if line[0].lower()=='y':
      maxmult=int(cidrtin.readline().split()[0])
      cidrtin.readline()
      nelec=int(cidrtin.readline().split()[0])
      if mult<=maxmult and (mult+nelec)%2!=0:
        return 1, (mult+1)/2    # socinr=1, single=-1, isc=0
      else:
        return None,None
    else:
      mult2=int(cidrtin.readline().split()[0])
      if mult!=mult2:
        #print 'Multiplicity %i cannot be treated in directory %s (single DRT)!'  % (mult,TEMPLATE)
        return None,None
      return -1,1
  except IOError:
    # find out in which DRT the requested multiplicity is
    for i in range(1,9):        # COLUMBUS can treat at most 8 DRTs
      try:
        cidrtin=open(TEMPLATE+'/cidrtin.%i' % i)
      except IOError:
        return None,None
      cidrtin.readline()
      mult2=int(cidrtin.readline().split()[0])
      if mult==mult2:
        return 0,i
      cidrtin.close()

# =================================================

def get_COLUMBUS(INFOS):
  '''This routine asks for all questions specific to COLUMBUS:
  - path to COLUMBUS
  - scratchdir
  - path to template directory
  - mocoef
  - memory
  '''

  string='\n  '+'='*80+'\n'
  string+='||'+centerstring('COLUMBUS Interface setup',80)+'||\n'
  string+='  '+'='*80+'\n\n'
  print string


  print centerstring('Path to COLUMBUS',60,'-')+'\n'
  path=os.getenv('COLUMBUS')
  if path=='':
    path=None
  else:
    path='$COLUMBUS/'
  #path=os.path.expanduser(os.path.expandvars(path))
  #if path!='':
    #print 'Environment variable $COLUMBUS detected:\n$COLUMBUS=%s\n' % (path)
    #if question('Do you want to use this COLUMBUS installation?',bool,True):
      #INFOS['columbus']=path
  #if not 'columbus' in INFOS:
  print '\nPlease specify path to COLUMBUS directory (SHELL variables and ~ can be used, will be expanded when interface is started).\n'
  INFOS['columbus']=question('Path to COLUMBUS:',str,path)
  print ''


  print centerstring('Scratch directory',60,'-')+'\n'
  print 'Please specify an appropriate scratch directory. This will be used to temporally store all COLUMBUS files. The scratch directory will be deleted after the calculation. Remember that this script cannot check whether the path is valid, since you may run the calculations on a different machine. The path will not be expanded by this script.'
  INFOS['scratchdir']=question('Path to scratch directory:',str)
  print ''


  print centerstring('COLUMBUS input template directory',60,'-')+'\n'
  print '''Please specify the path to the COLUMBUS template directory. 
The directory must contain subdirectories with complete COLUMBUS input file sets for the following steps:
- Integrals with SEWARD/MOLCAS
- SCF
- MCSCF
- SO-MRCI (even if no Spin-Orbit couplings will be calculated)
The COLUMBUS interface will generate the remaining COLUMBUS input automatically, depending on the number of states.

In order to setup the COLUMBUS input, use COLUMBUS' input facility colinp. For further information, see the Spin-orbit tutorial for COLUMBUS [1].

[1] http://www.univie.ac.at/columbus/docs_COL70/tutorial-SO.pdf
'''
  while True:
    path=question('Path to templates:',str)
    path=os.path.expanduser(os.path.expandvars(path))
    path=os.path.abspath(path)
    if not os.path.isdir(path):
      print 'Directory %s does not exist!' % (path)
      continue

    content=os.listdir(path)
    multmap={}
    allOK=True
    for mult in range(1,1+len(INFOS['states'])):
      if INFOS['states'][mult-1]==0:
        continue
      found=False
      for d in content:
        template=path+'/'+d
        socitype,drt=checktemplate_COLUMBUS(template,mult)
        if socitype==None:
          continue
        if not d[-1]=='/':
          d+='/'
        multmap[mult]=d
        found=True
        break
      if not found:
        print 'No input directory for multiplicity %i!' % (mult)
        allOK=False
        continue
    if allOK:
      break
  print '\nAccepted path: %s\n' % (path)

  print '''Check whether the jobs are assigned correctly to the multiplicities. Use the following commands:
  mult job        make <mult> use the input in <job>
  show            show the mapping of multiplicities to jobs
  end             confirm this mapping
'''
  for i in multmap:
    print '%i ==> %s' % (i,multmap[i])
  while True:
    line=question('Adjust job mapping:',str,'end',False)
    if 'show' in line.lower():
      for i in multmap:
        print '%i ==> %s' % (i,multmap[i])
      continue
    elif 'end' in line.lower():
      break
    else:
      f=line.split()
      try:
        m=int(f[0])
        j=f[1]
      except (ValueError,IndexError):
        continue
      if not m in multmap:
        print 'Multiplicity %i not necessary!' % (m)
        continue
      if not os.path.isdir(path+'/'+j):
        print 'No template subdirectory %s!' % (j)
        continue
      if not j[-1]=='/':
        j+='/'
      multmap[m]=j
  print ''

  mocoefmap={}
  for job in set([ multmap[i] for i in multmap]):
    mocoefmap[job]=multmap[1]
  print '''Check whether the mocoeffiles are assigned correctly to the jobs. Use the following commands:
  job mocoefjob   make <job> use the mocoeffiles from <mocoefjob>
  show            show the mapping of multiplicities to jobs
  end             confirm this mapping
'''
  width=max([ len(i) for i in mocoefmap] )
  for i in mocoefmap:
    print '%s' % (i) +' '*(width-len(i))+ ' <== %s' % (mocoefmap[i])
  while True:
    line=question('Adjust mocoef mapping:',str,'end',False)
    if 'show' in line.lower():
      for i in mocoefmap:
        print '%s <== %s' % (i,mocoefmap[i])
      continue
    elif 'end' in line.lower():
      break
    else:
      f=line.split()
      try:
        j=f[0]
        m=f[1]
      except (ValueError,IndexError):
        continue
      if not m[-1]=='/':
        m+='/'
      if not j[-1]=='/':
        j+='/'
      mocoefmap[j]=m
  print ''

  INFOS['columbus.template']=path
  INFOS['columbus.multmap']=multmap
  INFOS['columbus.mocoefmap']=mocoefmap

  INFOS['columbus.copy_template']=question('Do you want to copy the template directory to each trajectory (Otherwise it will be linked)?',bool,False)
  if INFOS['columbus.copy_template']:
    INFOS['columbus.copy_template_from']=INFOS['columbus.template']
    INFOS['columbus.template']='./COLUMBUS.template/'


  print centerstring('Initial wavefunction: MO Guess',60,'-')+'\n'
  print '''Please specify the path to a COLUMBUS mocoef file containing suitable starting MOs for the CASSCF calculation.
'''
  init=question('Do you have an initial mocoef file?',bool,True)
  if init:
    while True:
      line=question('Mocoef filename:',str,'mocoef_mc.init')
      line=os.path.expanduser(os.path.expandvars(line))
      if os.path.isfile(line):
          break
      else:
        print 'File not found!'
        continue
    INFOS['columbus.guess']=line
  else:
    print 'WARNING: Remember that CASSCF calculations may run very long and/or yield wrong results without proper starting MOs.'
    time.sleep(2)
    INFOS['columbus.guess']=False
  print ''


  print centerstring('COLUMBUS Memory usage',60,'-')+'\n'
  print '''Please specify the amount of memory available to COLUMBUS (in MB). For calculations including moderately-sized CASSCF calculations and less than 150 basis functions, around 2000 MB should be sufficient.
'''
  INFOS['columbus.mem']=abs(question('COLUMBUS memory:',int)[0])


  # Ionization
  print '\n'+centerstring('Ionization probability by Dyson norms',60,'-')+'\n'
  INFOS['ion']=question('Dyson norms?',bool,False)
  if INFOS['ion']:
    INFOS['columbus.dysonpath']=question('Path to dyson executable:',str)
    INFOS['columbus.civecpath']=question('Path to civecconsolidate executable:',str,'$COLUMBUS/civecconsolidate')
    INFOS['columbus.dysonthres']=abs(question('c2 threshold for Dyson:',float,[1e-4])[0])

  return INFOS

# ======================================================================================================================

def check_Analytical_block(data,identifier,nstates,eMsg):
  iline=-1
  while True:
    iline+=1
    if iline==len(data):
      if eMsg:
        print 'No matrix %s defined!' % (identifier)
      return False
    line=re.sub('#.*$','',data[iline]).split()
    if line==[]:
      continue
    ident=identifier.split()
    fits=True
    for i,el in enumerate(ident):
      if not el.lower() in line[i].lower():
        fits=False
        break
    if fits:
      break
  strings=data[iline+1:iline+1+nstates]
  for i,el in enumerate(strings):
    a=el.strip().split(',')
    if len(a)<i+1:
      if eMsg:
        print '%s matrix is not a lower triangular matrix with n=%i!' % (identifier,nstates)
      return False
  return True

# ======================================================================================================================

def checktemplate_Analytical(filename,req_nstates,eMsg=True):
  f=open(filename)
  data=f.readlines()
  f.close()

  # check whether first two lines are positive integers
  try:
    natom=int(data[0])
    nstates=int(data[1])
  except ValueError:
    if eMsg:
      print 'First two lines must contain natom and nstates!'
    return False
  if natom<1 or nstates<1:
    if eMsg:
      print 'natom and nstates must be positive!'
    return False
  if nstates!=req_nstates:
    if eMsg:
      print 'Template file is for %i states!' % (nstates)
    return False

  # check the next natom lines
  variables=set()
  for i in range(2,2+natom):
    line=data[i]
    match=re.match('\s*[a-zA-Z]*\s+[a-zA-Z0][a-zA-Z0-9_]*\s+[a-zA-Z0][a-zA-Z0-9_]*\s+[a-zA-Z0][a-zA-Z0-9_]*',line)
    if not match:
      if eMsg:
        print 'Line %i malformatted!' % (i+1)
      return False
    else:
      a=line.split()
      for j in range(3):
        match=re.match('\s*[a-zA-Z][a-zA-Z0-9_]*',a[j+1])
        if match:
          variables.add(a[j+1])

  # check variable blocks
  iline=-1
  while True:
    iline+=1
    if iline==len(data):
      break
    line=re.sub('#.*$','',data[iline]).split()
    if line==[]:
      continue
    if 'variables' in line[0].lower():
      while True:
        iline+=1
        if iline==len(data):
          if eMsg:
            print 'Non-terminated variables block!'
          return False
        line=re.sub('#.*$','',data[iline]).split()
        if line==[]:
          continue
        if 'end' in line[0].lower():
          break
        match=re.match('[a-zA-Z][a-zA-Z0-9_]*',line[0])
        if not match:
          if eMsg:
            print 'Invalid variable name: %s' % (line[0])
          return False
        try:
          a=float(line[1])
        except ValueError:
          if eMsg:
            print 'Non-numeric value for variable %s' % (line[0])
          return False
        except IndexError:
          if eMsg:
            print 'No value for variable %s' % (line[0])
          return False

  # check hamiltonian block
  line='hamiltonian'
  a=check_Analytical_block(data,line,nstates,eMsg)
  if not a:
    return False

  # check derivatives of each variable
  for v in variables:
    line='derivatives %s' % (v)
    a=check_Analytical_block(data,line,nstates,eMsg)
    if not a:
      return False

  return True

# ======================================================================================================================

def get_Analytical(INFOS):

  string='\n  '+'='*80+'\n'
  string+='||'+centerstring('Analytical PES Interface setup',80)+'||\n'
  string+='  '+'='*80+'\n\n'
  print string

  if os.path.isfile('SH2Ana.inp'):
    if checktemplate_Analytical('SH2Ana.inp',INFOS['nstates'],eMsg=True):
      print 'Valid file "SH2Ana.inp" detected. '
      usethisone=question('Use this template file?',bool,True)
      if usethisone:
        INFOS['analytical.template']='SH2Ana.inp'
  if not 'analytical.template' in INFOS:
    while True:
      filename=question('Template filename:',str)
      if not os.path.isfile(filename):
        print 'File %s does not exist!' % (filename)
        continue
      if checktemplate_Analytical(filename,INFOS['nstates']):
        break
    INFOS['analytical.template']=filename
  print ''

  return INFOS


# ======================================================================================================================

def checktemplate_MOLCAS(filename,INFOS):
  necessary=['basis','ras2','nactel','inactive']
  try:
    f=open(filename)
    data=f.readlines()
    f.close()
  except IOError:
    print 'Could not open template file %s' % (filename)
    return False
  valid=[]
  for i in necessary:
    for l in data:
      if i in l:
        valid.append(True)
        break
    else:
      valid.append(False)
  if not all(valid):
    print 'The template %s seems to be incomplete! It should contain: ' % (filename) +str(necessary)
    return False
  for mult,state in enumerate(INFOS['states']):
    if state<=0:
      continue
    valid=[]
    for l in data:
      if 'spin' in l.lower():
        f=l.split()
        if int(f[1])==mult+1:
          valid.append(True)
          break
    else:
      valid.append(False)
  if not all(valid):
    string='The template %s seems to be incomplete! It should contain the keyword "spin" for ' % (filename)
    for mult,state in enumerate(INFOS['states']):
      if state<=0:
        continue
      string+='%s, ' % (IToMult[mult+1])
    string=string[:-2]+'!'
    print string
    return False
  return True

# =================================================

def get_MOLCAS(INFOS):
  '''This routine asks for all questions specific to MOLPRO:
  - path to molpro
  - scratch directory
  - MOLPRO.template
  - wf.init
  '''

  string='\n  '+'='*80+'\n'
  string+='||'+centerstring('MOLCAS Interface setup',80)+'||\n'
  string+='  '+'='*80+'\n\n'
  print string

  print centerstring('Path to MOLCAS',60,'-')+'\n'
  path=os.getenv('MOLCAS')
  #path=os.path.expanduser(os.path.expandvars(path))
  if path=='':
    path=None
  else:
    path='$MOLCAS/'
      #print 'Environment variable $MOLCAS detected:\n$MOLCAS=%s\n' % (path)
      #if question('Do you want to use this MOLCAS installation?',bool,True):
        #INFOS['molcas']=path
    #if not 'molcas' in INFOS:
  print '\nPlease specify path to MOLCAS directory (SHELL variables and ~ can be used, will be expanded when interface is started).\n'
  INFOS['molcas']=question('Path to MOLCAS:',str,path)
  print ''


  print centerstring('Scratch directory',60,'-')+'\n'
  print 'Please specify an appropriate scratch directory. This will be used to temporally store the integrals. The scratch directory will be deleted after the calculation. Remember that this script cannot check whether the path is valid, since you may run the calculations on a different machine. The path will not be expanded by this script.'
  INFOS['scratchdir']=question('Path to scratch directory:',str)
  print ''


  print centerstring('MOLCAS input template file',60,'-')+'\n'
  print '''Please specify the path to the MOLcas.template file. This file must contain the following settings:
  
basis <Basis set>
ras2 <Number of active orbitals>
nactel <Number of active electrons>
inactive <Number of doubly occupied orbitals>
spin <Multiplicity (1=S)> roots <Number of roots for this multiplicity>  (repeat this line for each multiplicity)

The MOLCAS interface will generate the appropriate MOLCAS input automatically.
'''
  if os.path.isfile('MOLCAS.template'):
    if checktemplate_MOLCAS('MOLCAS.template',INFOS):
      print 'Valid file "MOLCAS.template" detected. '
      usethisone=question('Use this template file?',bool,True)
      if usethisone:
        INFOS['molcas.template']='MOLCAS.template'
  if not 'molcas.template' in INFOS:
    while True:
      filename=question('Template filename:',str)
      if not os.path.isfile(filename):
        print 'File %s does not exist!' % (filename)
        continue
      if checktemplate_MOLCAS(filename,INFOS):
        break
    INFOS['molcas.template']=filename
  print ''


  print centerstring('Initial wavefunction: MO Guess',60,'-')+'\n'
  print '''Please specify the path to a MOLCAS JobIph file containing suitable starting MOs for the CASSCF calculation. Please note that this script cannot check whether the wavefunction file and the Input template are consistent!
'''
  string='Do you have initial wavefunction files for '
  for mult,state in enumerate(INFOS['states']):
    if state<=0:
      continue
    string+='%s, ' % (IToMult[mult+1])
  string=string[:-2]+'?'
  if question(string,bool,True):
    INFOS['molcas.guess']={}
    for mult,state in enumerate(INFOS['states']):
      if state<=0:
        continue
      while True:
        filename=question('Initial wavefunction file for %ss:' % (IToMult[mult+1]),str,'wf.%i.JobIph.old' % (mult+1))
        if os.path.isfile(filename):
          INFOS['molcas.guess'][mult+1]=filename
          break
        else:
          print 'File not found!'
  else:
    print 'WARNING: Remember that CASSCF calculations may run very long and/or yield wrong results without proper starting MOs.'
    time.sleep(2)
    INFOS['molcas.guess']={}


  print centerstring('MOLCAS Memory usage',60,'-')+'\n'
  print '''Please specify the amount of memory available to MOLCAS (in MB). For calculations including moderately-sized CASSCF calculations and less than 150 basis functions, around 2000 MB should be sufficient.
'''
  INFOS['molcas.mem']=abs(question('MOLCAS memory:',int)[0])




  return INFOS

# ======================================================================================================================

def prepare_MOLPRO(INFOS,iconddir):
  # write SH2PRO.inp
  try:
    sh2pro=open('%s/SH2PRO.inp' % (iconddir), 'w')
  except IOError:
    print 'IOError during prepareMOLPRO, iconddir=%s' % (iconddir)
    quit(1)
  string='molpro %s\nscratchdir %s/%s/' % (INFOS['molpro'],INFOS['scratchdir'],iconddir)
  sh2pro.write(string)
  sh2pro.close()

  # copy MOs and template
  cpfrom=INFOS['molpro.template']
  cpto='%s/MOLPRO.template' % (iconddir)
  shutil.copy(cpfrom,cpto)
  if INFOS['molpro.guess']:
    cpfrom=INFOS['molpro.guess']
    cpto='%s/wf.init' % (iconddir)
    shutil.copy(cpfrom,cpto)

  return

# ======================================================================================================================

def prepare_COLUMBUS(INFOS,iconddir):
  # write SH2COL.inp
  try:
    sh2col=open('%s/SH2COL.inp' % (iconddir), 'w')
  except IOError:
    print 'IOError during prepareCOLUMBUS, directory=%i' % (iconddir)
    quit(1)
  string= 'columbus %s\nscratchdir %s/%s/WORK\n' % (INFOS['columbus'],INFOS['scratchdir'],iconddir)
  string+='savedir %s/%s/savedir\ntemplate %s\nmemory %i\nnooverlap\n\n' % (INFOS['scratchdir'],iconddir, INFOS['columbus.template'],INFOS['columbus.mem'])
  for mult in INFOS['columbus.multmap']:
    string+='DIR %i %s\n' % (mult,INFOS['columbus.multmap'][mult])
  string+='\n'
  for job in INFOS['columbus.mocoefmap']:
    string+='MOCOEF %s %s\n' % (job,INFOS['columbus.mocoefmap'][job])
  if INFOS['ion']:
    string+='dyson %s\n' % (INFOS['columbus.dysonpath'])
    string+='civecconsolidate %s\n' % (INFOS['columbus.civecpath'])
    string+='dysonthres %s\n' % (INFOS['columbus.dysonthres'])
  sh2col.write(string)
  sh2col.close()

  # copy MOs and template
  if INFOS['columbus.guess']:
    cpfrom=INFOS['columbus.guess']
    cpto='%s/mocoef_mc.init' % (iconddir)
    shutil.copy(cpfrom,cpto)

  if INFOS['columbus.copy_template']:
    copy_from=INFOS['columbus.copy_template_from']
    copy_to=iconddir+'/COLUMBUS.template/'
    shutil.copytree(copy_from,copy_to)

  return

# ======================================================================================================================

def prepare_Analytical(INFOS,iconddir):
  # copy SH2Ana.inp

  # copy MOs and template
  cpfrom=INFOS['analytical.template']
  cpto='%s/SH2Ana.inp' % (iconddir)
  shutil.copy(cpfrom,cpto)

  return

# ======================================================================================================================

def prepare_MOLCAS(INFOS,iconddir):
  # write SH2PRO.inp
  try:
    sh2cas=open('%s/SH2CAS.inp' % (iconddir), 'w')
  except IOError:
    print 'IOError during prepareMOLCAS, iconddir=%s' % (iconddir)
    quit(1)
  project=iconddir.replace('/','_')
  string='molcas %s\nscratchdir %s/%s/\nmemory %i\nproject %s' % (INFOS['molcas'],INFOS['scratchdir'],iconddir,INFOS['molcas.mem'],project)
  sh2cas.write(string)
  sh2cas.close()

  # copy MOs and template
  cpfrom=INFOS['molcas.template']
  cpto='%s/MOLCAS.template' % (iconddir)
  shutil.copy(cpfrom,cpto)
  if not INFOS['molcas.guess']=={}:
    for i in INFOS['molcas.guess']:
      cpfrom=INFOS['molcas.guess'][i]
      cpto='%s/%s.%i.JobIph.old' % (iconddir,project,i)
      shutil.copy(cpfrom,cpto)

  return





# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def get_runscript_info(INFOS):
  ''''''

  string='\n  '+'='*80+'\n'
  string+='||'+centerstring('Run mode setup',80)+'||\n'
  string+='  '+'='*80+'\n\n'
  print string

  print centerstring('Run script',60,'-')+'\n'
  print '''This script can generate the run scripts for each initial condition in two modes:

  - In the first mode, the calculation is run in subdirectories of the current directory.

  - In the second mode, the input files are transferred to another directory (e.g. a local scratch directory), the calculation is run there, results are copied back and the temporary directory is deleted. Note that this temporary directory is not the same as the scratchdir employed by the interfaces.

Note that in any case this script will setup the input subdirectories in the current working directory. 
'''
  print 'Do you want to use mode 1 \n(actually perform the calculations in subdirectories of: %s)\n' % (INFOS['cwd'])
  here=question('Calculate here?',bool,False)
  if here:
    INFOS['here']=True
  else:
    INFOS['here']=False
    print '\nWhere do you want to perform the calculations? Note that this script cannot check whether the path is valid.'
    INFOS['copydir']=question('Run directory?',str)
  print ''

  print centerstring('Submission script',60,'-')+'\n'
  print '''During the setup, a script for running all initial conditions sequentially in batch mode is generated. Additionally, a queue submission script can be generated for all initial conditions.
'''
  qsub=question('Generate submission script?',bool,False)
  if not qsub:
    INFOS['qsub']=False
  else:
    INFOS['qsub']=True
    print '\nPlease enter a queue submission command, including possibly options to the queueing system,\ne.g. for SGE: "qsub -q queue.q -S /bin/bash -cwd" (Do not type quotes!).'
    INFOS['qsubcommand']=question('Submission command?',str,None,False)
    INFOS['proj']=question('Project Name:',str,None,False)

  print ''
  return INFOS

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def make_directory(iconddir):
  '''Creates a directory'''

  if os.path.isfile(iconddir):
    print '\nWARNING: %s is a file!' % (iconddir)
    return -1
  if os.path.isdir(iconddir):
    if len(os.listdir(iconddir))==0:
      return 0
    else:
      print '\nWARNING: %s/ is not empty!' % (iconddir)
      if not 'overwrite' in globals():
        global overwrite
        overwrite=question('Do you want to overwrite files in this and all following directories? ',bool,False)
      if overwrite:
        return 0
      else:
        return -1
  else:
    try:
      os.mkdir(iconddir)
    except OSError:
      print '\nWARNING: %s cannot be created!' % (iconddir)
      return -1
    return 0

# ======================================================================================================================

def writeQMin(INFOS,iconddir):
  icond=int(iconddir[-6:-1])
  try:
    qmin=open('%s/QM.in' % (iconddir), 'w')
  except IOError:
    print 'IOError during writeQMin, icond=%s' % (iconddir)
    quit(1)
  string='%i\nInitial condition %s\n' % (INFOS['natom'],iconddir)

  if icond>0:
    searchstring='Index     %i' % (icond)
  else:
    searchstring='Equilibrium'
  rewinded=False
  while True:
    try:
      line=INFOS['initf'].readline()
    except EOFError:
      if not rewinded:
        rewinded=True
        INFOS['initf'].seek(0)
      else:
        print 'Could not find Initial condition %i!' % (icond)
        quit(1)
    if searchstring in line:
      break
  if icond>0:
    line=INFOS['initf'].readline()        # skip one line
  for iatom in range(INFOS['natom']):
    line=INFOS['initf'].readline()
    s=line.split()
    string+='%s %s %s %s\n' % (s[0],s[2],s[3],s[4])

  string+='unit bohr\ninit\ncleanup\nstates '
  for i in INFOS['states']:
    string+='%i ' % (i)
  if INFOS['soc']:
    string+='\nSOC\n'
  else:
    string+='\nH\n'
  string+='DM\n'
  if 'ion' in INFOS and INFOS['ion']:
    string+='ion\n'

  qmin.write(string)
  qmin.close()
  return

# ======================================================================================================================

def writeRunscript(INFOS,iconddir):
  '''writes the runscript in each subdirectory'''

  try:
    runscript=open('%s/run.sh' % (iconddir), 'w')
  except IOError:
    print 'IOError during writeRunscript, iconddir=%s' % (iconddir)
    quit(1)
  if 'proj' in INFOS:
    projname='%4s_%5s' % (INFOS['proj'][0:4],iconddir[-6:-1])
  else:
    projname='init_%5s' % (iconddir[-6:-1])

  if INFOS['here']:
    string='''#/bin/bash

#$-N %s

PRIMARY_DIR=%s/%s/

cd $PRIMARY_DIR

$SHARC/%s QM.in >> QM.log 2>> QM.err
''' % (projname,INFOS['cwd'], iconddir, Interfaces[INFOS['interface']]['script'])
  else:
    string='''#/bin/bash

#$-N %s

PRIMARY_DIR=%s/%s/
COPY_DIR=%s/%s/

mkdir -p $COPY_DIR
cp -r $PRIMARY_DIR/* $COPY_DIR
cd $COPY_DIR

$SHARC/%s QM.in >> QM.log 2>> QM.err

cp $COPY_DIR/QM.* $PRIMARY_DIR
rm -r $COPY_DIR
''' % (projname,INFOS['cwd'], iconddir, INFOS['copydir'], iconddir, Interfaces[INFOS['interface']]['script'])

  runscript.write(string)
  runscript.close()
  filename='%s/run.sh' % (iconddir)
  os.chmod(filename, os.stat(filename).st_mode | stat.S_IXUSR)
  return

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def setup_equilibrium(INFOS):
  iconddir='ICOND_%05i/' % (0)
  exists=os.path.isfile(iconddir+'/QM.out')
  if not exists:
    iconddir='ICOND_%05i/' % (0)
    io=make_directory(iconddir)
    if io!=0:
      print 'Skipping initial condition %i!' % (iconddir)
      return

    writeQMin(INFOS,iconddir)
    globals()[Interfaces[ INFOS['interface']]['prepare_routine'] ](INFOS,iconddir)
    writeRunscript(INFOS,iconddir)
  return exists

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def setup_all(INFOS):
  '''This routine sets up the directories for the initial calculations.'''

  string='\n  '+'='*80+'\n'
  string+='||'+centerstring('Setting up directories...',80)+'||\n'
  string+='  '+'='*80+'\n\n'
  print string

  all_run=open('all_run_init.sh','w')
  string='#/bin/bash\n\nCWD=%s\n\n' % (INFOS['cwd'])
  all_run.write(string)
  if INFOS['qsub']:
    all_qsub=open('all_qsub_init.sh','w')
    string='#/bin/bash\n\nCWD=%s\n\n' % (INFOS['cwd'])
    all_qsub.write(string)

  width=50
  ninit=INFOS['irange'][1]-INFOS['irange'][0]+1
  idone=0

  EqExists=setup_equilibrium(INFOS)
  if not EqExists:
    iconddir='ICOND_%05i/' % (0)
    string='cd $CWD/%s/\nbash run.sh\ncd $CWD\necho %s >> DONE\n' % (iconddir,iconddir)
    all_run.write(string)
    if INFOS['qsub']:
      string='cd $CWD/%s/\n%s run.sh\ncd $CWD\n' % (iconddir,INFOS['qsubcommand'])
      all_qsub.write(string)

  if INFOS['irange']!=[0,0]:
    for icond in range(INFOS['irange'][0],INFOS['irange'][1]+1):
      iconddir='ICOND_%05i/' % (icond)
      idone+=1
      done=idone*width/ninit
      sys.stdout.write('\rProgress: ['+'='*done+' '*(width-done)+'] %3i%%' % (done*100/width))
      sys.stdout.flush()

      io=make_directory(iconddir)
      if io!=0:
        print 'Skipping initial condition %i!' % (iconddir)
        continue

      writeQMin(INFOS,iconddir)
      globals()[Interfaces[ INFOS['interface']]['prepare_routine'] ](INFOS,iconddir)
      writeRunscript(INFOS,iconddir)

      string='cd $CWD/%s/\nbash run.sh\ncd $CWD\necho %s >> DONE\n' % (iconddir,iconddir)
      all_run.write(string)
      if INFOS['qsub']:
        string='cd $CWD/%s/\n%s run.sh\ncd $CWD\n' % (iconddir,INFOS['qsubcommand'])
        all_qsub.write(string)

  all_run.close()
  filename='all_run_init.sh'
  os.chmod(filename, os.stat(filename).st_mode | stat.S_IXUSR)
  if INFOS['qsub']:
    all_qsub.close()
    filename='all_qsub_init.sh'
    os.chmod(filename, os.stat(filename).st_mode | stat.S_IXUSR)

  print '\n'


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def main():
  '''Main routine'''

  usage='''
python setup_init.py

This interactive program prepares the initial excited-state calculations for SHARC.
As input it takes the initconds file, number of states and range of initconds.

Afterwards, it asks for the interface used and goes through the preparation depending on the interface.
'''

  description=''
  parser = OptionParser(usage=usage, description=description)

  displaywelcome()
  open_keystrokes()
  
  INFOS=get_general()
  INFOS=globals()[Interfaces[ INFOS['interface']]['get_routine'] ](INFOS)
  INFOS=get_runscript_info(INFOS)

  print '\n'+centerstring('Full input',60,'#')+'\n'
  for item in INFOS:
    print item, ' '*(25-len(item)), INFOS[item]
  print ''
  setup=question('Do you want to setup the specified calculations?',bool,True)
  print ''

  if setup:
    setup_all(INFOS)

  close_keystrokes()


# ======================================================================================================================

if __name__ == '__main__':
  try:
    main()
  except KeyboardInterrupt:
    print '\nCtrl+C makes me a sad SHARC ;-(\n'
    quit(0)
