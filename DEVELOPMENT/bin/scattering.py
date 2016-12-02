#!/usr/bin/env python2

import sys
if sys.version_info[0]!=2:
  sys.stdout.write('The SHARC suite is not compatible with Python 3! Use Python 2 (>2.6)!')
  sys.exit(0)


import copy
import math
import re
import os
import stat
import shutil
import random
import datetime
from optparse import OptionParser
import readline
import time

try:
  import numpy
  NONUMPY=False
except ImportError:
  import subprocess as sp
  NONUMPY=True

# =========================================================0
# compatibility stuff

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
versiondate=datetime.date(2014,12,18)


NUMBERS = {'H':  1, 'He': 2,
'Li': 3, 'Be': 4, 'B':  5, 'C':  6,  'N': 7,  'O': 8, 'F':  9, 'Ne':10,
'Na':11, 'Mg':12, 'Al':13, 'Si':14,  'P':15,  'S':16, 'Cl':17, 'Ar':18,
'K': 19, 'Ca':20, 
'Sc':21, 'Ti':22, 'V': 23, 'Cr':24, 'Mn':25, 'Fe':26, 'Co':27, 'Ni':28, 'Cu':29, 'Zn':30,
'Ga':31, 'Ge':32, 'As':33, 'Se':34, 'Br':35, 'Kr':36, 
'Rb':37, 'Sr':38,
'Y':39,  'Zr':40, 'Nb':41, 'Mo':42, 'Tc':43, 'Ru':44, 'Rh':45, 'Pd':46, 'Ag':47, 'Cd':48,
'In':49, 'Sn':50, 'Sb':51, 'Te':52,  'I':53, 'Xe':54,
'Cs':55, 'Ba':56,
'La':57, 'Hf':72, 'Ta':73,  'W':74, 'Re':75, 'Os':76, 'Ir':77, 'Pt':78, 'Au':79, 'Hg':80,
'Tl':81, 'Pb':82, 'Bi':83, 'Po':84, 'At':85, 'Rn':86
}

# Atomic Weights of the most common isotopes
# Masses are from MOLPRO, except where noted
# Also very comprehensive: http://www.nist.gov/pml/data/comp.cfm
MASSES = {'H' :   1.00782 * U_TO_AMU,
          'He':   4.00260 * U_TO_AMU,
          'Li':   7.01600 * U_TO_AMU,
          'Be':   9.01218 * U_TO_AMU,
          'B' :  11.00931 * U_TO_AMU,
          'C' :  12.00000 * U_TO_AMU,
          'N' :  14.00307 * U_TO_AMU,
          'O' :  15.99491 * U_TO_AMU,
          'F' :  18.99840 * U_TO_AMU,
          'Ne':  19.99244 * U_TO_AMU,
          'Na':  22.98980 * U_TO_AMU,
          'Mg':  23.98504 * U_TO_AMU,
          'Al':  26.98153 * U_TO_AMU,
          'Si':  27.97693 * U_TO_AMU,
          'P' :  30.97376 * U_TO_AMU,
          'S' :  31.97207 * U_TO_AMU,
          'Cl':  34.96885 * U_TO_AMU,
          'Ar':  39.96238 * U_TO_AMU,
          'K' :  38.96371 * U_TO_AMU,
          'Ca':  39.96259 * U_TO_AMU,
          'Sc':  44.95592 * U_TO_AMU,
          'Ti':  47.94795 * U_TO_AMU,
          'V' :  50.94400 * U_TO_AMU,
          'Cr':  51.94050 * U_TO_AMU,
          'Mn':  54.93800 * U_TO_AMU,
          'Fe':  55.93490 * U_TO_AMU,
          'Co':  58.93320 * U_TO_AMU,
          'Ni':  57.93534 * U_TO_AMU,
          'Cu':  62.92960 * U_TO_AMU,
          'Zn':  63.92910 * U_TO_AMU,
          'Ga':  68.92570 * U_TO_AMU,
          'Ge':  73.92190 * U_TO_AMU,
          'As':  74.92160 * U_TO_AMU,
          'Se':  79.91650 * U_TO_AMU,
          'Br':  78.91830 * U_TO_AMU,
          'Kr':  83.80000 * U_TO_AMU,
          'Rb':  84.91170 * U_TO_AMU,
          'Sr':  87.90560 * U_TO_AMU,
          'Y' :  88.90590 * U_TO_AMU,
          'Zr':  89.90430 * U_TO_AMU,
          'Nb':  92.90600 * U_TO_AMU,
          'Mo':  97.90550 * U_TO_AMU,
          'Tc':  98.90620 * U_TO_AMU,
          'Ru': 101.90370 * U_TO_AMU,
          'Rh': 102.90480 * U_TO_AMU,
          'Pd': 105.90320 * U_TO_AMU,
          'Ag': 106.90509 * U_TO_AMU,
          'Cd': 113.90360 * U_TO_AMU,
          'In': 114.90410 * U_TO_AMU,
          'Sn': 119.90220 * U_TO_AMU,   # MOLPRO library is wrong
          'Sb': 120.90380 * U_TO_AMU,
          'Te': 129.90670 * U_TO_AMU,
          'I' : 126.90440 * U_TO_AMU,
          'Xe': 131.90420 * U_TO_AMU,
          'Cs': 132.90510 * U_TO_AMU,
          'Ba': 137.90500 * U_TO_AMU,
          'La': 138.90610 * U_TO_AMU,
          'Hf': 179.94680 * U_TO_AMU,
          'Ta': 180.94800 * U_TO_AMU,
          'W' : 183.95100 * U_TO_AMU,
          'Re': 186.95600 * U_TO_AMU,
          'Os': 190.20000 * U_TO_AMU,
          'Ir': 192.96330 * U_TO_AMU,
          'Pt': 194.96480 * U_TO_AMU,
          'Au': 196.96660 * U_TO_AMU,
          'Hg': 201.97060 * U_TO_AMU,
          'Tl': 204.97450 * U_TO_AMU,
          'Pb': 207.97660 * U_TO_AMU,
          'Bi': 208.98040 * U_TO_AMU,
          'Po': 208.98250 * U_TO_AMU,
          'At': 209.98715 * U_TO_AMU,   # MOLPRO library is wrong
          'Rn': 210.99060 * U_TO_AMU}   # MOLPRO library is wrong

# Isotopes used for the masses
ISOTOPES={'H' : 'H-1' ,
          'He': 'He-4',
          'Li': 'Li-7',
          'Be': 'Be-9*',
          'B' : 'B_11' ,
          'C' : 'C-12' ,
          'N' : 'N-14' ,
          'O' : 'O-16' ,
          'F' : 'F-19*' ,
          'Ne': 'Ne-20',
          'Na': 'Na-23*',
          'Mg': 'Mg-24',
          'Al': 'Al-27*',
          'Si': 'Si-28',
          'P' : 'P-31*' ,
          'S' : 'S-32' ,
          'Cl': 'Cl-35',
          'Ar': 'Ar-40',
          'K' : 'K-39' ,
          'Ca': 'Ca-40',
          'Sc': 'Sc-45*',
          'Ti': 'Ti-48',
          'V' : 'V-51' ,
          'Cr': 'Cr-52',
          'Mn': 'Mn-55*',
          'Fe': 'Fe-56',
          'Co': 'Co-59*',
          'Ni': 'Ni-58',
          'Cu': 'Cu-63',
          'Zn': 'Zn-64',
          'Ga': 'Ga-68',
          'Ge': 'Ge-74',
          'As': 'As-75*',
          'Se': 'Se-80',
          'Br': 'Br-79',
          'Kr': 'Kr-84',
          'Rb': 'Rb-85',
          'Sr': 'Sr-88',
          'Y' : 'Y-89*' ,
          'Zr': 'Zr-90',
          'Nb': 'Nb-93*',
          'Mo': 'Mo-98',
          'Tc': 'Tc-99',
          'Ru': 'Ru-102',
          'Rh': 'Rh-103*',
          'Pd': 'Pd-106',
          'Ag': 'Ag-107',
          'Cd': 'Cd-114',
          'In': 'In-115',
          'Sn': 'Sn-120',
          'Sb': 'Sb-121',
          'Te': 'Te-130',
          'I' : 'I-127*' ,
          'Xe': 'Xe-132',
          'Cs': 'Cs-133*',
          'Ba': 'Ba-138',
          'La': 'La-139',
          'Hf': 'Hf-180',
          'Ta': 'Ta-181',
          'W' : 'W-184' ,
          'Re': 'Re-187',
          'Os': 'Os-192',
          'Ir': 'Ir-193',
          'Pt': 'Pt-195',
          'Au': 'Au-197*',
          'Hg': 'Hg-202',
          'Tl': 'Tl-205',
          'Pb': 'Pb-208',
          'Bi': 'Bi-209*',
          'Po': 'Po-209',
          'At': 'At-210',
          'Rn': 'Rn-211'}

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def try_read(l,index,typefunc,default):
  try:
    if typefunc==bool:
      return 'True'==l[index]
    else:
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
    self.dip     = [ complex( try_read(f,i,float,0.),try_read(f,i+1,float,0.) ) for i in [3,5,7] ]
    self.Excited =   try_read(f,11,bool, False)
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
    s+='% 12.8f % 12.8f %s' % (self.Eexc*HARTREE_TO_EV,self.Fosc,self.Excited)
    return s

  def Excite(self,max_Prob,erange):
    try:
      Prob=self.Prob/max_Prob
    except ZeroDivisionError:
      Prob=-1.
    if not (erange[0] <= self.Eexc <= erange[1]):
      Prob=-1.
    self.Excited=(random.random() < Prob)

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
      #if 'Index     %i' % (index) in line:
      if re.search('Index\s+%i' % (index),line):
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
  string='\n'
  string+='  '+'='*80+'\n'
  string+='||'+centerstring('',80)+'||\n'
  string+='||'+centerstring('Excite initial conditions for SHARC',80)+'||\n'
  string+='||'+centerstring('',80)+'||\n'
  string+='||'+centerstring('Author: Sebastian Mai',80)+'||\n'
  string+='||'+centerstring('',80)+'||\n'
  string+='||'+centerstring('Version:'+version,80)+'||\n'
  string+='||'+centerstring(versiondate.strftime("%d.%m.%y"),80)+'||\n'
  string+='||'+centerstring('',80)+'||\n'
  string+='  '+'='*80+'\n\n'
  string+='''
This script automatizes to read-out the results of initial excited-state calculations for SHARC.
It calculates oscillator strength (in MCH and diagonal basis) and stochastically determines whether
a trajectory is bright or not.
  '''
  print string

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def open_keystrokes():
  global KEYSTROKES
  KEYSTROKES=open('KEYSTROKES.tmp','w')

def close_keystrokes():
  KEYSTROKES.close()
  shutil.move('KEYSTROKES.tmp','KEYSTROKES.scattering')

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
# ======================================================================================================================
# ======================================================================================================================

def get_infos(INFOS):
  '''This routine asks for the paths of the initconds file and ICONDS directory, for energy window and the representation.'''

  print centerstring('Initial conditions file for TARGET',60,'-')+'\n'
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
  INFOS['ninit']=int(initf.readline().split()[1])
  INFOS['natom']=int(initf.readline().split()[1])
  INFOS['repr']=initf.readline().split()[1]
  INFOS['eref']=float(initf.readline().split()[1])
  INFOS['eharm']=float(initf.readline().split()[1])

  return INFOS

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def get_initconds(INFOS):
  ''''''

  if not INFOS['read_QMout'] and not INFOS['make_list']:
    INFOS['initf'].seek(0)
    while True:
      line=INFOS['initf'].readline()
      if 'Repr' in line:
        INFOS['diag']=line.split()[1].lower()=='diag'
        INFOS['repr']=line.split()[1]
      if 'Eref' in line:
        INFOS['eref']=float(line.split()[1])
        break

  initlist=[]
  for icond in range(1,INFOS['ninit']+1):
    initcond=INITCOND()
    initcond.init_from_file(INFOS['initf'],INFOS['eref'],icond)
    initlist.append(initcond)
  print 'Number of initial conditions in file:       %5i' % (INFOS['ninit'])
  return initlist

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def writeoutput(initlist,INFOS):
  outfilename=INFOS['initf'].name+'.excited'
  if os.path.isfile(outfilename):
    overw=question('Overwrite %s? ' % (outfilename),bool,False)
    print ''
    if overw:
      try:
        outf=open(outfilename,'w')
      except IOError:
        print 'Could not open: %s' % (outfilename)
        outf=None
    else:
      outf=None
    if not outf:
      while True:
        outfilename=question('Please enter the output filename: ',str)
        try:
          outf=open(outfilename,'w')
        except IOError:
          print 'Could not open: %s' % (outfilename)
          continue
        break
  else:
    outf=open(outfilename,'w')

  print 'Writing output to %s ...' % (outfilename)

  string='''SHARC Initial conditions file, version %s   <Excited>
Ninit     %i
Natom     %i
Repr      %s
Eref      %18.10f
Eharm     %18.10f
''' % (version,INFOS['ninit'],INFOS['natom'],INFOS['repr'],INFOS['eref'],INFOS['eharm'])
  if INFOS['states']:
    string+='States    '
    for n in INFOS['states']:
      string+='%i ' % (n)
  string+='\n\n\nEquilibrium\n'

  for atom in INFOS['equi']:
    string+=str(atom)+'\n'
  string+='\n\n'

  for i,icond in enumerate(initlist):
    string+= 'Index     %i\n%s' % (i+1, str(icond))
  outf.write(string)
  outf.close()


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def main():
  '''Main routine'''

  usage='''
python scattering.py

This interactive script reads out initconds files and combines them into a single initconds file for scattering initial conditions.
'''
  description=''
  parser = OptionParser(usage=usage, description=description)
  #parser.add_option('--no-excitation', dest='E', action='store_true',default=False,help="Sets all excitations to false.")
  #parser.add_option('--ground-state-only', dest='G', action='store_true',default=False,help="Selects the ground state of all initial conditions, and no excited states (e.g., for dynamics with laser excitation).")
  #(options, args) = parser.parse_args()

  displaywelcome()
  open_keystrokes()

  INFOS={}
  INFOS=get_infos(INFOS)

  print '\n\n'+centerstring('Full input',60,'#')+'\n'
  for item in INFOS:
    if not item=='equi':
      print item, ' '*(25-len(item)), INFOS[item]
  print ''
  go_on=question('Do you want to continue?',bool,True)
  if not go_on:
    quit(0)
  print ''

  initlist=get_initconds(INFOS)



  close_keystrokes()

# ======================================================================================================================

if __name__ == '__main__':
  try:
    main()
  except KeyboardInterrupt:
    print '\nCtrl+C makes me a sad SHARC ;-(\n'
    quit(0)
