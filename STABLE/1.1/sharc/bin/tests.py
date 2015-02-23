#!/usr/bin/env python2

# Script to test whether a correct python version is installed, and to run the test calculations.
# 
# usage 
import sys
if sys.version_info[0]!=2:
  sys.stdout.write('*'*80+'\nThe SHARC suite is not compatible with Python 3! \nUse Python 2 (>2.6)!\n'+'*'*80+'\n')
  sys.exit(1)

import copy
import math
import re
import os
import datetime
from optparse import OptionParser
import readline
import shutil
import subprocess as sp
import filecmp
import time

# =========================================================0
# compatibility stuff

if sys.version_info[1]<6:
  sys.stdout.write('*'*80+'\nThe SHARC suite is not tested to work with Python versions older than 2.6!\nProceed at your own risk.\nSome scripts might work, while other won''t.\nWe recommend you to install a more recent version of Python 2.\n'+'*'*80+'\n')
  time.sleep(5)

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

try:
  import numpy
except ImportError:
  sys.stdout.write('*'*80+'\nThe Python package NumPy was not found.\nPerformance of excite.py might be slightly reduced.\n'+'*'*80+'\n')
  time.sleep(5)






version='1.0'
versiondate=datetime.date(2014,10,8)



INTERFACES=set(['MOLPRO','MOLCAS','COLUMBUS','Analytical','scripts'])


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
  string+='||'+centerstring('SHARC Test suite run script',80)+'||\n'
  string+='||'+centerstring('',80)+'||\n'
  string+='||'+centerstring('Author: Sebastian Mai',80)+'||\n'
  string+='||'+centerstring('',80)+'||\n'
  string+='||'+centerstring('Version:'+version,80)+'||\n'
  string+='||'+centerstring(versiondate.strftime("%d.%m.%y"),80)+'||\n'
  string+='||'+centerstring('',80)+'||\n'
  string+='  '+'='*80+'\n\n'
  string+='''
This script collects a number of environment variables and subsequently runs
the calculations in the SHARC test suite. After the runs, the output is checked 
against the reference outputs.
  '''
  sys.stdout.write(string+'\n')

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def open_keystrokes():
  global KEYSTROKES
  KEYSTROKES=open('KEYSTROKES.tmp','w')

def close_keystrokes():
  KEYSTROKES.close()
  shutil.move('KEYSTROKES.tmp','KEYSTROKES.test_suite')

# ===================================

def question(question,typefunc,default=None,autocomplete=True):
  if typefunc==int or typefunc==float:
    if not default==None and not isinstance(default,list):
      sys.stdout.write('Default to int or float question must be list!\n')
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
        sys.stdout.write('I didn''t understand you.\n')
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
        sys.stdout.write('Please enter a %s\n' % ( ['string','integer','float'][i] ))
        continue

# ======================================================================================================================

def env_or_question(varname,setenv=False):
  path=os.getenv(varname)
  if path!=None and path!='':
    path=os.path.expanduser(os.path.expandvars(path))
    sys.stdout.write('\nEnvironment variable $%s detected:\n$%s=%s\n\n' % (varname,varname,path))
    if question('Do you want to use this?',bool,True):
      return path
  path=question('Please enter the path for $%s:' % (varname),str)
  path=os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
  if setenv:
    os.environ[varname]=path
  return path

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def get_infos():
  INFOS={}

  # pwd
  INFOS['pwd']=os.getcwd()

  # get SHARC path, if not there
  string='\n  '+'='*80+'\n'
  string+='||'+centerstring('SHARC path',80)+'||\n'
  string+='  '+'='*80+'\n'
  sys.stdout.write(string+'\n')
  INFOS['sharc']=env_or_question('SHARC',setenv=True)
  INFOS['sharc']=os.path.abspath(os.path.expanduser(os.path.expandvars(INFOS['sharc'])))

  # check for valid SHARC directory
  ls=os.listdir(INFOS['sharc'])
  necessary=['sharc.x','data_extractor.x','wigner.py','setup_init.py','setup_traj.py']
  for i in necessary:
    if not i in ls:
      sys.stdout.write('Missing directory "%s" in $SHARC path' % (i))
      quit(1)
  ls=os.listdir(INFOS['sharc']+'/..')
  necessary=['tests']
  for i in necessary:
    if not i in ls:
      sys.stdout.write('Missing directory "%s" in $SHARC/.. path' % (i))
      quit(1)

  # get list of available test jobs
  string='\n  '+'='*80+'\n'
  string+='||'+centerstring('Tests to run',80)+'||\n'
  string+='  '+'='*80+'\n'
  sys.stdout.write(string+'\n')
  ls=os.listdir(INFOS['sharc']+'/../tests/INPUT')
  ls.sort()
  testlist=[]
  for i in ls:
    for j in INTERFACES:
      if j in i:
        testlist.append([i,j])
      continue
  sys.stdout.write('Available test simulations:\n')
  for index,i in enumerate(testlist):
    if 'scripts' in i:
      sys.stdout.write('%5i '% (index+1) + i[0]+' '*(25-len(i[0]))+'\n')
    else:
      sys.stdout.write('%5i '% (index+1) + i[0]+' '*(25-len(i[0]))+'via SHARC_%s.py\n' % (i[1]))
  sys.stdout.write('\n')

  # specify the jobs which should be run
  jobs=question('Which jobs should be run (enter the corresponding numbers)?',int,[i+1 for i in range(len(testlist))])
  jobs=list(set(jobs))
  jobs.sort()
  INFOS['joblist']=[ testlist[j-1][0] for j in jobs if 0<j<=len(testlist)]
  INFOS['interfaces']=set([ testlist[j-1][1] for j in jobs])

  # collect environment variables
  string='\n  '+'='*80+'\n'
  string+='||'+centerstring('Environment variables: Paths',80)+'||\n'
  string+='  '+'='*80+'\n'
  sys.stdout.write(string+'\n')
  for interface in INTERFACES:
    if interface in INFOS['interfaces'] and not interface in ['Analytical','scripts']:
      INFOS[interface]=env_or_question(interface,setenv=True)

  # get scratch directory
  string='\n  '+'='*80+'\n'
  string+='||'+centerstring('Scratch directory',80)+'||\n'
  string+='  '+'='*80+'\n'
  sys.stdout.write(string+'\n')
  INFOS['SCRADIR']=env_or_question('SCRADIR',setenv=True)

  return INFOS

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def run_tests(INFOS):
  string='\n  '+'='*80+'\n'
  string+='||'+centerstring('Running test jobs...',80)+'||\n'
  string+='  '+'='*80+'\n'
  sys.stdout.write(string+'\n')

  INFOS['joberrors']=[]

  for job in INFOS['joblist']:
    path=INFOS['sharc']+'/../tests/INPUT/'+job
    newpath=INFOS['pwd']+'/RUNNING_TESTS/'+job
    #os.chdir(path)
    if os.path.isdir(newpath):
      sys.stdout.write('%s *** OVERWRITTEN ***\n' % (newpath))
      shutil.rmtree(newpath)
    shutil.copytree(path,newpath)
    os.chdir(newpath)

    starttime=datetime.datetime.now()
    sys.stdout.write('%s\n\t%s' % (path,starttime))
    sys.stdout.flush()
    command='sh run.sh > run.out 2> run.err'
    #sys.stdout.write(command)
    try:
      outfile=open('run.out','w')
      errfile=open('run.err','w')
      runerror=sp.call(command,shell=True,stdout=outfile,stderr=errfile)
    except OSError:
      sys.stdout.write('Call have had some serious problems:'+str(OSError)+'\n')
      quit(1)
    endtime=datetime.datetime.now()
    sys.stdout.write('\t%s\t\tRuntime: %s\t\tError Code: %i\n\n' % (endtime,endtime-starttime,runerror))
    os.chdir(INFOS['pwd'])

    INFOS['joberrors'].append(runerror)

  return INFOS

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def full_lists(dc):
  same=dc.same_files
  diff=dc.diff_files
  for subd in dc.subdirs:
    el=dc.subdirs[subd]
    s,d=full_lists(el)
    for i in s:
      same.append(subd+'/'+i)
    for i in d:
      diff.append(subd+'/'+i)
  return same,diff

# ======================================================================================================================

def compare_scripts(INFOS,index):
  dc=filecmp.dircmp(INFOS['pwd']+'/RUNNING_TESTS/'+INFOS['joblist'][index],
                    INFOS['sharc']+'/../tests/RESULTS/'+INFOS['joblist'][index])
  same,diff=full_lists(dc)
  ignore_files=['run.sh','all_run_init.sh','runQM.sh','SH2PRO.inp']
  diff=[item for item in diff if not any([f in item for f in ignore_files]) ]
  sys.stdout.write('Differing: '+str(diff)+'\n')
  sys.stdout.write('Same     : '+str(same)+'\n')
  count=len(diff)
  return count

# ======================================================================================================================

def sign(x):
  if x==float('inf'):
    return 1.
  elif x==-float('inf'):
    return -1.
  elif x==0.:
    return 0.
  else:
    return math.copysign(1, x)

# ======================================================================================================================

def compare_trajectories(INFOS,index):
  file1=INFOS['pwd']+'/RUNNING_TESTS/'+INFOS['joblist'][index]+'/output.dat'
  sys.stdout.write(file1+'\n')
  f=open(file1)
  outtext=f.readlines()
  f.close()

  file2=INFOS['sharc']+'/../tests/RESULTS/'+INFOS['joblist'][index]+'/output.dat'
  sys.stdout.write(file2+'\n')
  f=open(file2)
  reftext=f.readlines()
  f.close()

  # return -1 if output.dat files have different lengths
  if len(outtext)!=len(reftext):
    count=-1
    return count


  compare={
  # flag accuracy   check sign?
    -1: [1e-8, True],    # anything in the header
     0: [1e-8, False],   # step
     1: [1e-8, False],   # Hamiltonian
     2: [1e-8, True],    # U matrix
     3: [1e-8, False],   # Dipole matrices
     4: [1e-8, True],    # Overlap matrix
     5: [1e-8, True],    # coefficients
     6: [1e-8, False],   # probabilities
     7: [1e-8, False],   # ekin
     8: [0e+0, False],   # states
     9: [1e+0, False],   # random number
    10: [1e+8, False],   # runtime
    11: [1e-6, True],    # geometry
    12: [1e-6, True],    # velocity
    13: [1e+8, False]    # property matrix
  }

  count=0
  nlines=len(outtext)
  flag=-1

  for i in range(nlines):
    try:
      a=outtext[i]
      b=reftext[i]
    except IndexError:
      break
    if a[0]=='!':
      if not b[0]=='!':
        return -1
      flag=int(a.split()[1])
      continue
    if flag==-1:
      a1=[float(a.split()[0])]
      b1=[float(b.split()[0])]
    else:
      a1=[float(j) for j in a.split()]
      b1=[float(j) for j in b.split()]

    for j,ja in enumerate(a1):
      jb=b1[j]
      d=abs(ja-jb)
      if d>compare[flag][0]:
        count+=1
        sys.stdout.write('*** Value deviation on line %i: %18.12f vs %18.12f\n' % (i, ja,jb))
      if compare[flag][1]:
        if not sign(ja)==sign(jb):
          count+=1
          sys.stdout.write('***  Sign deviation on line %i: %18.12f vs %18.12f\n' % (i, ja,jb))
  return count

# ======================================================================================================================

def run_diff(INFOS):
  string='\n  '+'='*80+'\n'
  string+='||'+centerstring('Test job analysis',80)+'||\n'
  string+='  '+'='*80
  sys.stdout.write(string+'\n')

  INFOS['result']=[]

  for index,job in enumerate(INFOS['joblist']):
    sys.stdout.write('\n'+centerstring(job,60,'-')+'\n')

    if INFOS['joberrors'][index]!=0:
      sys.stdout.write('Job did not finish successfully. Error code: %i\n' % (INFOS['joberrors'][index]))
      INFOS['result'].append('Job did not finish')
      continue

    if 'scripts' in INFOS['joblist'][index]:
      count=compare_scripts(INFOS,index)
    else:
      count=compare_trajectories(INFOS,index)

    if count==0:
      sys.stdout.write('Output and reference are identical.\n')
      INFOS['result'].append('Test SUCCESSFUL.')
    elif count<0:
      sys.stdout.write('Output and Reference have different length!')
      INFOS['result'].append('Different output length.')
    else:
      sys.stdout.write('Output and reference show differences.\n')
      INFOS['result'].append('%i Differences detected.' % count)

  string='\n  '+'='*80+'\n'
  string+='||'+centerstring('Summary',80)+'||\n'
  string+='  '+'='*80
  sys.stdout.write(string+'\n')
  for index,job in enumerate(INFOS['joblist']):
    sys.stdout.write(job+' '*(25-len(job)) + INFOS['result'][index]+'\n')

# ======================================================================================================================

def recursive_overwrite(src, dest, ignore=None):
    if os.path.isdir(src):
        if not os.path.isdir(dest):
            os.makedirs(dest)
        files = os.listdir(src)
        if ignore is not None:
            ignored = ignore(src, files)
        else:
            ignored = set()
        for f in files:
            if f not in ignored:
                recursive_overwrite(os.path.join(src, f), 
                                    os.path.join(dest, f), 
                                    ignore)
    else:
        shutil.copyfile(src, dest)

# ======================================================================================================================

def update_results(INFOS):
  sys.stdout.write('Overwriting reference output files...\n')

  for job in INFOS['joblist']:
    sys.stdout.write('%s\n' % (job))
    path= INFOS['pwd']+'/RUNNING_TESTS/'+job
    path2=INFOS['sharc']+'/../tests/RESULTS/'+job
    ls=os.listdir(path2)
    for i in ls:
      recursive_overwrite(path+'/'+i,path2+'/'+i)

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def main():
  '''Main routine'''

  usage=''
  description=''
  parser = OptionParser(usage=usage, description=description)
  parser.add_option('--update_results', dest='u', action='store_true',default=False,help="")
  (options, args) = parser.parse_args()

  displaywelcome()
  open_keystrokes()

  INFOS=get_infos()

  sys.stdout.write('\n'+centerstring('Full input',60,'#')+'\n\n')
  for item in INFOS:
    sys.stdout.write(str(item)+' '*(15-len(item))+str(INFOS[item])+'\n')
  sys.stdout.write('\n')
  setup=question('Do you want to setup the specified calculations?',bool,True)
  sys.stdout.write('\n')
  if setup:
    INFOS=run_tests(INFOS)
    run_diff(INFOS)

  if options.u:
    update_results(INFOS)

  close_keystrokes()

# ======================================================================================================================

if __name__ == '__main__':
  try:
    main()
  except KeyboardInterrupt:
    sys.stdout.write('\nCtrl+C makes me a sad SHARC ;-(\n')
    quit(0)