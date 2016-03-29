#!/usr/bin/env python2

# Script for the creation of ADF input files
# 
# usage 

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
sys.path.append('/usr/license/adf/adf2014.07/scripting')
from kf import *


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



# some constants
DEBUG = False
CM_TO_HARTREE = 1./219474.6     #4.556335252e-6 # conversion factor from cm-1 to Hartree
HARTREE_TO_EV = 27.211396132    # conversion factor from Hartree to eV
U_TO_AMU = 1./5.4857990943e-4            # conversion from g/mol to amu
BOHR_TO_ANG=0.529177211
PI = math.pi

version='1.0'
versiondate=datetime.date(2015,7,15)


# List of atomic numbers until Rn, with Lanthanoids missing (1-57, 72-86)
NUMBERS = {'H':  1, 'He': 2,
'Li': 3, 'Be': 4, 'B':  5, 'C':  6,  'N': 7,  'O': 8, 'F':  9, 'Ne':10,
'Na':11, 'Mg':12, 'Al':13, 'Si':14,  'P':15,  'S':16, 'Cl':17, 'Ar':18,
'K': 19, 'Ca':20, 
'Sc':21, 'Ti':22, 'V': 23, 'Cr':24, 'Mn':25, 'Fe':26, 'Co':27, 'Ni':28, 'Cd':29, 'Zn':30,
'Ge':31, 'Ga':32, 'As':33, 'Se':34, 'Br':35, 'Kr':36, 
'Rb':37, 'Sr':38,
'Y':39,  'Zr':40, 'Nb':41, 'Mo':42, 'Tc':43, 'Ru':44, 'Rh':45, 'Pd':46, 'Ag':47, 'Cd':48,
'In':49, 'Sn':50, 'Sb':51, 'Te':52,  'I':53, 'Xe':54,
'Cs':55, 'Ba':56,
'La':57, 'Hf':72, 'Ta':73,  'W':74, 'Re':75, 'Os':76, 'Ir':77, 'Pt':78, 'Au':79, 'Hg':80,
'Tl':81, 'Pb':82, 'Bi':83, 'Po':84, 'At':85, 'Rn':86
}

# List of atomic numbers until Rn, with Lanthanoids missing (1-57, 72-86)
Atomsymb = {1: 'H', 2: 'He',
3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar',
19: 'K', 20: 'Ca',
21:'Sc', 22:'Ti', 23:'V', 24:'Cr', 25:'Mn', 26:'Fe', 27:'Co', 28:'Ni', 29:'Cd', 30:'Zn',
31:'Ge', 32:'Ga', 33:'As', 34:'Se', 35:'Br', 36:'Kr',
37:'Rb', 38:'Sr',
39:'Y',  40:'Zr', 41:'Nb', 42:'Mo', 43:'Tc', 44:'Ru', 45:'Rh', 46:'Pd', 47:'Ag', 48:'Cd',
49:'In', 50:'Sn', 51:'Sb', 52:'Te',  53:'I', 54:'Xe',
55:'Cs', 56:'Ba',
57:'La', 72:'Hf', 73:'Ta',  74:'W', 75:'Re', 76:'Os', 77:'Ir', 78:'Pt', 79:'Au', 80:'Hg',
81:'Tl', 82:'Pb', 83:'Bi', 84:'Po', 85:'At', 86:'Rn'
}

#Atomic Radii 
Allingerradii= {'H':   1.350,'He':  1.275,
'Li':  2.125,'Be':  1.858,'B':   1.792,'C':   1.700,'N':   1.608,'O':   1.517,'F':   1.425,'Ne':  1.333,
'Na':  2.250,'Mg':  2.025,'Al':  1.967,'Si':  1.908,'P':   1.850,'S':   1.792,'Cl':  1.725,'Ar':  1.658,
'K':   2.575,'Ca':  2.342,'Sc':  2.175,'Ti':  1.992,'V':   1.908,'Cr':  1.875,'Mn':  1.867,'Fe':  1.858,'Co':  1.858,'Ni':  1.850,'Cu':  1.883,'Zn':  1.908,'Ga':  2.050,'Ge':  2.033,'As':  1.967,'Se':  1.908,'Br':  1.850,'Kr':  1.792,
'Rb':  2.708,'Sr':  2.500,'Y':   2.258,'Zr':  2.117,'Nb':  2.025,'Mo':  1.992,'Tc':  1.967,'Ru':  1.950,'Rh':  1.950,'Pd':  1.975,'Ag':  2.025,'Cd':  2.083,'In':  2.200,'Sn':  2.158,'Sb':  2.100,'Te':  2.033,'I':   1.967,'Xe':  1.900,
'Cs':  2.867,'Ba':  2.558,'La':  2.317,'Ce':  2.283,'Pr':  2.275,'Nd':  2.275,'Pm':  2.267,'Sm':  2.258,'Eu':  2.450,'Gd':  2.258,'Tb':  2.250,'Dy':  2.242,'Ho':  2.225,'Er':  2.225,'Tm':  2.225,'Yb':  2.325,'Lu':  2.208,'Hf':  2.108,'Ta':  2.025,'W':   1.992,'Re':  1.975,'Os':  1.958,'Ir':  1.967,'Pt':  1.992,'Au':  2.025,'Hg':  2.108,'Tl':  2.158,'Pb':  2.283,'Bi':  2.217,'Po':  2.158,'At':  2.092,'Rn':  2.025,'Fr':  3.033,'Ra':  2.725,'Ac':  2.567,'Th':  2.283,'Pa':  2.200,'U':   2.100,'Np':  2.100,'Pu':  2.100
}

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def centerstring(string,n,pad=' '):
  l=len(string)
  if l>=n:
    return string
  else:
    return  pad*((n-l+1)/2)+string+pad*((n-l)/2)

def displaywelcome():
  string='\n'
  string+='  '+'='*80+'\n'
  string+='||'+centerstring('',80)+'||\n'
  string+='||'+centerstring('ADF template file generator',80)+'||\n'
  string+='||'+centerstring('',80)+'||\n'
  string+='||'+centerstring('Author: Andrew Atkins',80)+'||\n'
  string+='||'+centerstring('',80)+'||\n'
  string+='||'+centerstring('Version:'+version,80)+'||\n'
  string+='||'+centerstring(versiondate.strftime("%d.%m.%y"),80)+'||\n'
  string+='||'+centerstring('',80)+'||\n'
  string+='  '+'='*80+'\n\n'
  string+='''
This script allows to quickly create ADF input files. 
It also generates the ADF.template files to be used with the SHARC-ADF interface.
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
  shutil.move('KEYSTROKES.tmp','KEYSTROKES.adf_input')

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

def get_infos():
  '''Asks for the settings of the calculation:
- type (single point, optimization+freq or ADF.template
- Functional
- basis set
- Spin-orbit Coupling
- geometry

specific:
- opt: freq?
- TD-DFT'''

  INFOS={}

  #Type of Calculation
  print centerstring('Type of calculation',60,'-')
  print '''\nThis script generates input for the following types of calculations:  
  1   Single Point calculations
  2   Optimizations and Frequency calculations
  3   ADF.template
Please enter the number corresponding to the type of calculation.
'''
  while True:
    ctype=question('Type of calculation:',int)[0]
    if not ctype in [1,2,3]:
      print 'Enter an integer (1-3)!'
      continue
    break
  INFOS['ctype']=ctype
  freq=False
  print ''
  print 'If alterations to the options given here or more complex options are wished to be used please\nsee the ADF manual at www.scm.com/Doc/Doc2014/ADF/ADFUsersGuide/page1.html' 
  print ''
  if ctype==2:
    freq=question('Frequency calculation?',bool,True)
  INFOS['freq']=freq
  print ''

  path=''
  # Geometry
  if ctype <= 2:
     print centerstring('Geometry',60,'-')
     print '\nPlease specify the geometry file (xyz format, Angstroms):'
     while True:
       path=question('Geometry filename:',str,'geom.xyz')
       try:
         gf=open(path,'r')
       except IOError:
         print 'Could not open: %s' % (path)
         continue
       g=gf.readlines()
       gf.close()
       try:
         natom=int(g[0])
       except ValueError:
         print 'Malformatted: %s' % (path)
         continue
       geom=[]
       fine=True
       for i in range(natom):
         try:
           line=g[i+2].split()
         except IndexError:
           print 'Malformatted: %s' % (path)
           fine=False
         try:
           j=i+1
           if re.match('[\d*]',line[0]) != None:
              line[0]=Atomsymb[int(line[0])]
           atom=[int(j),line[0],float(line[1]),float(line[2]),float(line[3])]
         except (IndexError,ValueError):
           print 'Malformatted: %s' % (path)
           fine=False
           continue
         geom.append(atom)
       else:
         break
     INFOS['geom']=geom
     print ''
     print 'Please input the name of the created .run file (if none stated will use the xyz file name)'
     paths=question('name of the run file:',str,path[:-3]+'run')
     INFOS['path']=paths
     print ''
     print 'Enter the total (net) molecular charge:'
     charge=question('Charge:',float,[0.0])[0]
     INFOS['CHARGE']=charge
     print ''
     print 'Please state the number of unpaired electrons (alpha-beta).'
     Unr=question('Nr of unpaired electrons:',float,[0.0])[0]
     INFOS['Unp_elec']=Unr
     if ctype == 2 :
        print ''
        print 'Please state the number of geometry iterations permitted:'
        iterations = question('Iterations:',int,[150])[0]
        INFOS['GIter']=iterations  
        if INFOS['freq']==True:
           print ''
           print 'Do you wish to perform Analytical frequencies (Only works for some GGA functionals. but should be quicker)?'
           Type=question('Analytical frequencies:',bool,True)
           if Type == True:
             INFOS['Freqtype']='analytical'
           else:
             INFOS['Freqtype']='numerical'
  elif ctype ==3:
     print ''
     print 'Enter the total (net) molecular charge:'
     charge=question('Charge:',float,[0.0])[0]
     INFOS['CHARGE']=charge
     print ''
     print 'Please state the number of unpaired electrons (alpha-beta).'
     Unr=question('Nr of unpaired electrons:',float,[0.0])[0]
     INFOS['Unp_elec']=Unr
     
  print ''
  print 'Please choose the maximum number of SCF iterations'
  SCFIter = question('Max Iterations:',int,[100])[0]
  INFOS['SCFIter']=SCFIter

  # basis set
  print ''
  print '\nPlease enter the basis set. (SZ, DZ, DZP, TZP, TZ2P, QZ4P)'
  print 'If system contains a transition metal TZP or higher is recommended'
  basis=question('Basis set:',str,'TZP',autocomplete=False)
  INFOS['basis']=basis

  #DFT SETUP
  print ''
  print '\n Please choose the type of Functional (LDA, GGA, HYBRID, METAGGA, METAHYBRID, MODEL, RANGE).'
  print '\nGGA is recommended for Dynamics'
  XCtype=question('XCtype:',str,'GGA',autocomplete=False)
  INFOS['XCtype']=XCtype
  print ''
  print '\n Please state the Functional to be used.'
  print '\n Meta, Model and Range-seperated can not be used for dynamics'
  print '''Common functionals and their names in ADF (for more see\nhttp://www.scm.com/Doc/Doc2014/ADF/ADFUsersGuide/page68.html#keyscheme%20XC):
 LDA: VWN, PW92
 GGA: BP86, PBE, mPBE
 Hybrid: B3LYP, PBE0 BHandHLYP
 Meta-GGA: TPSS, M06-L
 Meta-Hybrid: TPSSH, M06
 Model: LB94, SAOP
 Range-separated: CAMY-B3LYP, LCY-PBE
'''
  XCfun=question('Functional:',str,'BP86',autocomplete=False)
  INFOS['XCfun']=XCfun 
  print ''
  print 'Would you like to include Relativistic effects (Scalar)?'
  Rel=question('Include Relativistic effects:',bool,True)
  INFOS['Rel']=Rel
  if Rel == True:
     print''
     print 'Use Spin-orbit Coupling? (Perturbative method)'
     SO=question('Spin-orbit coupling:',bool,False)
     INFOS['SO_COUP']=SO
  print ''
  print 'ZlmFit, STOFit or EXACTDENSITY? (Warning ZlmFit replaced STOFit as default)'
  print 'This is used to fit the Density unless EXACTDENSITY is chosen'
  FIT=question('Fit type',str,'ZlmFit',autocomplete=False)
  INFOS['FIT']=FIT
  if FIT == 'ZlmFit':
     print ''
     print 'Choose Fit Grid Quality (basic,normal,good,verygood,excellent),\na higher grid quality means increased computation time!'
     Fitquality=question('Grid Quality:',str,'normal',autocomplete=False)
     INFOS['FitGrid']=Fitquality
  print ''
  print 'Choose Grid Quality for Integration (Becke: basic,normal,good,verygood,excellent),\na higher grid quality means increased computation time!'
  IntGrid=question('Grid Quality:',str,'good',autocomplete=False)
  INFOS['IntGrid']=IntGrid 
  print ''
  print 'Do you want to run excited state calculations (yes/no)?'
  Exci=question('Excitation Calculation',bool,False)
  INFOS['Exci']=Exci
  if Exci == True:
     print ''
     print 'Singlets, Triplets or Both (ONLYSING, ONLYTRIP or BOTH)?'
     print 'If Spin-orbit coupling only select BOTH'
     Mult=question('Excitation Type:',str,'BOTH',autocomplete=False)
     INFOS['Mult']=Mult
     print ''
     print 'Number of Excitations to Calculate (If both types then you calculate the number for both Singlets and Triplets)'
     NrExci=question('Number of Excitations:',int,[10])[0]
     INFOS['NrExci']=NrExci
     print ''
     print 'Select to use the Tamm-Dancoff Approximation (Recommended as it shoudl improve the description of conical intersections)'
     TDA=question('Use TDA?',bool,True)
     INFOS['TDA']=TDA
     print ''
     print 'Do you want to optimize an excited state or get excited state gradients (In SHARC-dynamics runs do not use in template)?'
     ExcitedGO=question('Gradients/GeoOpt:',bool,False)
     INFOS['ExcGO']=ExcitedGO
     if ExcitedGO == True:
        print ''
        SingTrip=question('Singlet or Triplet:',str,'SINGLET',autocomplete=False)
        INFOS['SINGTRIP']=SingTrip
        print ''
        State=question('Which State:',int,[1])[0]
        INFOS['STATE']=State
        print ''
        Iter=question('Number of Iterations:',int,[0])[0]
        INFOS['Iter']=Iter
  print ''
  print 'Would you like to use COSMO (Does not work with Excited State Geometries)?'
  COSMO=question('Include the Solvent:',bool,False)
  INFOS['COSMO']=COSMO
  if COSMO == True:
     print 'Choose whether to use a simple input with solvent name or User defined'
     User=question('User defined dielectric constant and solvent radius?',bool,False) 
     INFOS['User']=User
     print ''
     if User == True:
        Epsi = question('State the dielecric constant:',float,[0.1])[0]
        print ''
        Rad = question('State the solvent radius:',float,[1.93])[0]
        INFOS['Epsi']=Epsi
        INFOS['Rad']=Rad
     else:
        print 'Here are a few solvent options: infinitedielectric, Water, Acetonitrile, Methanol, Dichloromethane'
        Solvent=question('Which Solvent:',str,'infinitedielectric',autocomplete=False)
        INFOS['SOLVENT']=Solvent
     
  print ''

  return INFOS

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def setup_input(INFOS):
  ''''''
  
  inpf=''
  if INFOS['ctype']==3:
     inpf='ADF.template'
  else:
     inpf=INFOS['path']
  print 'Writing input to %s' % (inpf)
  try:
    inp=open(inpf,'w')
  except IOError:
    print 'Could not open %s for write!' % (inpf)
    quit(1)
  s='ATOMS\n'
  if INFOS['ctype']<=2:
     for n in range(0,int(len(INFOS['geom']))):
         s+='%i %s %4.8f %4.8f %4.8f \n' % (INFOS['geom'][n][0],INFOS['geom'][n][1],INFOS['geom'][n][2],INFOS['geom'][n][3],INFOS['geom'][n][4])
     s+='END\n\n'
     if INFOS['ctype']==2:
        if INFOS['freq']==True:
           if INFOS['Freqtype']!='analytical':
              s+='GEOMETRY \nFrequencies Numdif=2 SCANALL \noptim Delocalized \niterations %s \nEND\n\n' %(INFOS['GIter'])
           else:
              s+='GEOMETRY \n optim Delocalized\n iterations %s\nEND\n\nAnalyticalFreq\nEND\n\n'%(INFOS['GIter'])
        else:
           s+='GEOMETRY \n optim Delocalized\n iterations %s\nEND\n\n' %(INFOS['GIter'])
  else:
     s+='END\n\n'
  s+='SYMMETRY NOSYM\n\n'

  if INFOS['Rel']==True:
     s+='RELATIVISTIC Scalar ZORA \n\n'
     if INFOS['SO_COUP'] == True:
        s+= 'SOPERT\nPRINT SOMATRIX\nGSCORR\n\n'

  s+='SAVE TAPE21 TAPE41\n\n'
  s+='SCF \niterations %i\nEND\n\n' %(INFOS['SCFIter'])

  s+='BASIS \n type %s \n core None \n createoutput None\nEND \n\n' % (INFOS['basis'])
  s+='BeckeGrid\n Quality %s\nEND\n\n' % (INFOS['IntGrid'])

  if INFOS['XCtype'] != 'RANGE':
     s+='XC\n %s %s \nEND\n\n' % (INFOS['XCtype'],INFOS['XCfun'])
  else:
     if INFOS['XCfun'] == 'CAMY-B3LYP':
        s+='XC\nHybrid CAMY-B3LYP\nXCFUN\nRANGESEP\nEND\n\n'
     else:
        s+='XC\nGGA %s\nXCFUN\nRANGESEP\nEND\n\n' % (INFOS['XCfun'])

  if INFOS['CHARGE'] !=0.0 or INFOS['Unp_elec'] != 0.0:
     s+='CHARGE %1.1f %1.1f \n\n' %(INFOS['CHARGE'], INFOS['Unp_elec'])  
     if INFOS['Unp_elec'] != 0.0:
        s+='UNRESTRICTED \n\n'

  if INFOS['FIT'] == 'STOFit':
     s+='STOFIT\n\n'
  elif INFOS['FIT']== 'ZlmFit':
     s+='ZlmFit\nQuality %s\nEND\n\n' % (INFOS['FitGrid'])
  else:
     s+='EXACTDENSITY\n\n'

  if INFOS['Exci'] == True and INFOS['SO_COUP'] == True:
     s+='EXCITATION\nDAVIDSON\nlowest %i\nEND\n\n' % (INFOS['NrExci'])
  elif INFOS['Exci'] == True:
     if INFOS['Mult'] != 'BOTH':
        s+='EXCITATION\n%s DAVIDSON\nlowest %i\nEND\n\n' % (INFOS['Mult'],INFOS['NrExci'])
     else:
        s+='EXCITATION\nDAVIDSON\nlowest %s\nEND\n\n' % (INFOS['NrExci'])
     if INFOS['TDA'] == True and INFOS['Exci'] == True:
        s+='TDA \n\n'
  if INFOS['Exci']==True and INFOS['ExcGO'] == True:
     s+='GEOMETRY\nIterations %s\nEND\n\n' % (INFOS['Iter'])
     s+='EXCITEDGO\n%s\nSTATE A %i\nOUTPUT=4\nEND\n\n' %(INFOS['SINGTRIP'],INFOS['STATE'])
 
  if INFOS['COSMO']== True:
     if INFOS['User']==True:
        s+='SOLVATION\nSurf Esurf\n Solv eps=%2.2f rad=%2.2f cav0=0.0 cav1=0.0067639\nCharged method=CONJ\nC-Mat POT\nSCF VAR ALL\nCSMRSP\nEND\n\n' %(INFOS['Epsi'], INFOS['Rad'])
     else:
        if INFOS['SOLVENT'] == 'infinitedielectric':
           s+='SOLVATION\nSurf Esurf\n Solv eps=0.1 rad=1.93 cav0=0.0 cav1=0.0067639\nCharged method=CONJ\nC-Mat POT\nSCF VAR ALL\nCSMRSP\nEND\n\n'
        else:
           s+='SOLVATION\nSurf Esurf\n Solv name=%s cav0=0.0 cav1=0.0067639\nCharged method=CONJ\nC-Mat POT\nSCF VAR ALL\nCSMRSP\nEND\n\n' %(INFOS['SOLVENT'])


  s+='\n\n'
#  s+='* Infos:\n'
#  s+='* %s@%s\n' % (os.environ['USER'],os.environ['HOSTNAME'])
#  s+='* Date: %s\n' % (datetime.datetime.now())
#  s+='* Current directory: %s\n\n' % (os.getcwd())

  inp.write(s)

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def main():
  '''Main routine'''

  usage='''
python ADF_input.py

This interactive program prepares template files for the SHARC-MOLCAS interface.
'''

  description=''
  parser = OptionParser(usage=usage, description=description)

  displaywelcome()
  open_keystrokes()

  INFOS=get_infos()

  print centerstring('Full input',60,'#')+'\n'
  for item in INFOS:
    print item, ' '*(15-len(item)), INFOS[item]
  print ''

  setup_input(INFOS)
  print '\nFinished\n'

  close_keystrokes()

# ======================================================================================================================

if __name__ == '__main__':
  try:
    main()
  except KeyboardInterrupt:
    print '\nCtrl+C makes me a sad SHARC ;-(\n'
    quit(0)
