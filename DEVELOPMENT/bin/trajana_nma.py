#!/usr/bin/python

"""
author: Felix Plasser
version: 1.0
descr: Script for doing a normal mode analysis. Coordinates are transformed into the normal mode basis.
    Before this they should have been aligned with the script align.sh
    For each time step structure the coordinates in <ref_struc_file> are subtracted. The resulting vector is transformed into the normal mode basis contained in <vibration_file>.
    <abs_list> contains a list of indeces of normal modes where the absolute value has to be taken because of symmetry considerations.
    <ana_ints> contains a nested list of time step intervals for which the analysis is made
    <plot> states wether cross averages against time are plotted and saved into files
    
    Output (all values in Angstrom):
       - nma.txt in each results folder -> direct transformation of the coordinates
       - in the <out_dir> folder:
            - mean_against_time.txt -> cross mean over trajectories against the time
            - std_against_time.txt -> standard deviation of cross averaging
            - total_std.txt -> total std over all time steps and trajectories. representative of how active a normal mode is.
            - cross_av_std.txt -> averaging over trajectories and std over timesteps. representative of coherent motions.
"""

# possible add-ons:
    # phase correction (1/omega d/dx) - then the amplitude could be computed out of any part of a sine wave 
    # comparison to zero-point-vibrations
    # maybe computation with absolute values should be changed. an oscillating planar molecule would only show half the expected amplitude the way it is implemented now -> multiplication by 2 or comparison against mean without absolute values. But that may not make sense if the molecule loses its planarity.

import os, sys, shutil, re, datetime, readline
sys.path.insert(0, os.environ['SHARC']+'/../lib')
try:
    import numpy
except ImportError:
    print 'numpy package not installed'
    sys.exit()
try:
    import file_handler, vib_molden, traj_manip, struc_linalg
except ImportError:
    print 'file_handler, vib_molden, traj_manip or struc_linalg not found. They should be part of this package. Check the installation and if $SHARC/../lib is part of the PYTHONPATH environment variable.'
    sys.exit()
try:
    import plotting
    plot_possible = True
except:
    print 'Plotting not possible (probably because pylab/matplotlib is not installed or because there is no X connection)'
    plot_possible = False

version='1.0'
versiondate=datetime.date(2016,05,12)

def centerstring(string,n,pad=' '):
  l=len(string)
  if l>=n:
    return string
  else:
    return  pad*((n-l+1)/2)+string+pad*((n-l)/2)

def displaywelcome():
  print 'Script for performing essential dynamics analysis started ...\n'
  string='\n'
  string+='  '+'='*80+'\n'
  string+='||'+centerstring('',80)+'||\n'
  string+='||'+centerstring('Reading structures from SHARC dynamics',80)+'||\n'
  string+='||'+centerstring('',80)+'||\n'
  string+='||'+centerstring('Author: Felix Plasser, Andrew Atkins',80)+'||\n'
  string+='||'+centerstring('',80)+'||\n'
  string+='||'+centerstring('Version:'+version,80)+'||\n'
  string+='||'+centerstring(versiondate.strftime("%d.%m.%y"),80)+'||\n'
  string+='||'+centerstring('',80)+'||\n'
  string+='  '+'='*80+'\n\n'
  string+='''
This script reads output.xyz files and calculates the essential dynamics 
(i.e., Shows you which are the most important motions).
  '''
  print string


def open_keystrokes():
  global KEYSTROKES
  KEYSTROKES=open('KEYSTROKES.tmp','w')

def close_keystrokes():
  KEYSTROKES.close()
  shutil.move('KEYSTROKES.tmp','KEYSTROKES.nma')

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
      KEYSTROKES.write(line+' '*(40-len(line))+' #'+s+'\n')
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



### input
# ref_struc_file # file with a reference structure
# ref_struc_type # type of that file
# vibration_file # molden format file with vibrations
# first_traj = 1
# last_traj = 36 #36
# dt = .5 # length of time step
# num_steps = 601 # maximum number of timesteps
# abs_list = [7,8,9,11,12,13,15,17,19,20,22,26,28,29,30,32,33,36,37] # absolute value is taken because of symmetry; 19 is the correct number from irrep analysis
# neg_list = abs_list # values are negativ in the summary files
# ana_ints = [[0,201],[0,601]] # time intervals to be analysed
# plot = True # shall plots of average against time be drawn?
# descr '1' # descr added to the filenames
###

 ##  read in variable assignments from nma.inp, this is always done when the program is called
 # first default definitions
dt = .5
abs_list = []
neg_list = []
plot = False

 
 # read from file
#execfile('nma.inp')

def get_general():

  INFOS={}

  print centerstring('Paths to trajectories',60,'-')
  print '\nPlease enter the paths to all directories containing the "TRAJ_0XXXX" directories.\nE.g. Sing_2/ and Sing_3/. \nPlease enter one path at a time, and type "end" to finish the list.'
  count=0
  paths=[]
  while True:
    path=question('Path: ',str,'end')
    if path=='end':
      if len(paths)==0:
        print 'No path yet!'
        continue
      print ''
      break
    path=os.path.expanduser(os.path.expandvars(path))
    if not os.path.isdir(path):
      print 'Does not exist or is not a directory: %s' % (path)
      continue
    if path in paths:
      print 'Already included.'
      continue
    ls=os.listdir(path)
    print ls
    for i in ls:
      if 'TRAJ' in i:
        count+=1
    print 'Found %i subdirectories in total.\n' % count
    paths.append(path)
  INFOS['paths']=paths
  print 'Total number of subdirectories: %i\n' % (count)

  print centerstring('Paths to reference structure',60,'-')
  print '\nPlease enter the path to the equilibrium structure of your system (in the same atomic order as that given in the dynamics output)'
  print ''
  refpath=question('Path: ',str,'ref.xyz')
  refpath=os.path.expanduser(os.path.expandvars(refpath))
  print ''
  reftype=question('Please give the type of coordinate file',str,'xyz')
  INFOS['refstruc']=refpath
  INFOS['reftype']=reftype
  print ''


  print centerstring('Path to vibration file',60,'-')
  print '\nPlease enter the path to the Molden frequencies file for your molecule (in the same atomic order as that given in the dynamics output)'
  print ''
  refvib=question('Path: ',str,'ref.molden')
  refvib=os.path.expanduser(os.path.expandvars(refvib))
  INFOS['refvib']=refvib
  print ''
  
  massweightques=question('Do you wish to use mass weighted normal modes?',bool,True)
  INFOS['massweight']=massweightques
  print ''

  print centerstring('Number of total steps in your trajectories',60,'-')
  print '\n total simulation time *2 and +1 if timestep is 0.5 fs'
  print ''
  while True:
    numsteps=question('Simulation time steps: ',int,[2000])[0]
    if numsteps <0.:
      print 'Number of steps must be positive!'
      continue
    break
  INFOS['numsteps']=numsteps
  print ''


  print centerstring('The time step of your calculation',60,'-')
  print ''
  while True:
    timestep=question('Simulation time step: ',float,[0.5])[0]
    if numsteps <0.:
      print 'time step must be positive!'
      continue
    break
  INFOS['timestep']=timestep
  print ''

  print centerstring('Automatic plot creation',60,'-')
  print ''
  autoplot=question('Do you want to automatically create plots of your data?',bool,False)
  INFOS['plot']=autoplot
  print ''


  symmodes_list=[]
  print centerstring('Any symmetric modes that should have their absolute value taken?',60,'-')
  print 'List of normal modes for which the absolute value is taken because of symmetry. The numbering is according to the Molden input file. Without this setting all non-totally symmetric modes should average out to 0. If left blank all modes will be included as they are. A value of -1 ends this input section.'
  print ''
  while True:
      symmodes_val=question('Please give the number of a normal mode you wish to include as an absolute value',str,'-1')
      if '-1' in symmodes_val:
         INFOS['symmmodes']=symmodes_list
         break
      else:
         symmodes_list1=symmodes_val.split()
         for i in range(len(symmodes_list1)):
            symmodes_list.append(int(symmodes_list1[i]))
         continue
      print ''
  print ''

  negmodes_list=[]
  print centerstring('Any modes you want to mark with a negative value?',60,'-')
  print 'List of normal modes where the negative value is taken in total_std.txt and cross_av_std.txt. This is only for convenience when viewing the results. A value of -1 ends this input section.'
  print ''
  while True:
      negmodes_val=question('Please give the number of a normal mode you wish to mark as negative',int,[-1])[0]
      if negmodes_val==-1:
         INFOS['negmodes']=negmodes_list
         break
      else:
         negmodes_list.append(negmodes_val)
         continue
      print ''

  print ''
  intervallist=[]
  print centerstring('Simulation time to be analysed',60,'-')
  print '\n Time intervals in which the trajectory is to be analyzed (0 to 2000 is 1000fs with 0.5fs timesteps)'
  print ''
  while True:
    starttime=question('Start time of interval: ',int,[0])[0]
    endtime=question('End time of interval: ',int,[2000])[0]
    print ''
    interval=[starttime,endtime]
    intervallist.append(interval)
    moreinterval=question('Do you want to add another time interval for analysis?',bool,False)
    if moreinterval==True:
      continue
    else:
      INFOS['interval']=intervallist
      break
    print ''
  print ''

  print centerstring('Please give the name of the subdirectory to be used for the results',60,'-')
  destin=question('Name for subdirectory?',str,'nma')
  INFOS['descr']=destin
  print ''

  return INFOS

 
def nm_analysis(INFOS):
    """
    Normal mode analysis. Typically this script is carried out.
    """
    print 'Preparing NMA ...'

    num_steps=INFOS['numsteps']
    descr=INFOS['descr']
    out_dir = os.path.join('NMA',descr)
    ref_struc_file=INFOS['refstruc']
    ref_struc_type=INFOS['reftype']
    ana_ints=INFOS['interval']
    vibration_file=INFOS['refvib']
    dt=INFOS['timestep']
    abs_list=INFOS['symmmodes']
    neg_list=INFOS['negmodes']
    plot=INFOS['plot']
    mawe=INFOS['massweight']    

    try:
        os.makedirs(out_dir)
    except OSError:
        print 'Output directory could not be created. It either already exists or you do not have writing access.'
    
#    shutil.copy('nma.inp',out_dir)
    
    ref_struc = struc_linalg.structure('ref_struc') # define the structure that all the time step structures are superimposed onto
    ref_struc.read_file(ref_struc_file, ref_struc_type)
    num_at = ref_struc.ret_num_at()
    mol_calc = struc_linalg.mol_calc(def_file_path=ref_struc_file, file_type=ref_struc_type)
    
    # read in data from the vibration file
    vmol = vib_molden.vib_molden()
    vmol.read_molden_file(vibration_file)
    nma_mat = numpy.linalg.pinv(vmol.ret_vib_matrix()) # this way it is a coordinate transformation
        # +++ the alternative would be an orthogonal projection, e.g. if only a few modes are chosen
    if mawe ==True:    
       mass_mat = mol_calc.ret_mass_matrix(power=0.5) # the mass matrix is for the mass weighted skalar product, it is the unit matrix if mass_wt_pw=0
       nma_mat_mawe=numpy.dot(mass_mat,nma_mat)
       nma_mat=nma_mat_mawe
    header = vmol.ret_nma_header()
#     eff_mass_array = numpy.array(vmol.ret_eff_masses(mol_calc=mol_calc, mass_wt_pw = mass_wt_pw))
    num_vib=len(nma_mat[0])
    
    mult_array = numpy.zeros(num_vib, float) # for final summary files, output is multiplied with this list
    for i in xrange(num_vib):
        if i+1 in neg_list:
            mult_array[i] = -1
        else:
            mult_array[i] = 1

    # used for computing the variance for each mode over all trajectories and timesteps
    num_points = numpy.zeros(len(ana_ints)) # number of all timesteps in all the trajectories for each interval analysed
    sum_array = numpy.zeros([len(ana_ints), num_vib], float) # a number for every time interval analysed and normal mode
    sum_sq_array = numpy.zeros([len(ana_ints), num_vib], float)
    
    # used for computing time resolved mean and variance
    cross_num_array = numpy.zeros(num_steps)
    cross_sum_array = numpy.zeros([num_steps,num_vib], float) # a number for every time step and normal mode; sum, has to be divided by cross_num_array
    cross_mean_array = numpy.zeros([num_steps,num_vib], float) # cross_sum_array / cross_num_array
    cross_sum_sq_array = numpy.zeros([num_steps,num_vib], float)
    
    #not_list = []

    forbidden=['crashed','running','dead','dont_analyze']
    width=30
    files=[]
    ntraj=0
    print 'Checking the directories...'
    for idir in INFOS['paths']:
      ls=os.listdir(idir)
      for itraj in ls:
        if not 'TRAJ_' in itraj:
          continue
        path=idir+'/'+itraj
        s=path+' '*(width-len(path))
        pathfile=path+'/output.xyz'
        if not os.path.isfile(pathfile):
          s+='%s NOT FOUND' % (pathfile)
          print s
          continue
        lstraj=os.listdir(path)
        valid=True
        for i in lstraj:
          if i.lower() in forbidden:
            s+='DETECTED FILE %s' % (i.lower())
            print s
            valid=False
            break
        if not valid:
          continue
        s+='OK'
        print s
        ntraj+=1
        files.append(pathfile)
    print 'Number of trajectories: %i' % (ntraj)
    if ntraj==0:
      print 'No valid trajectories found, exiting...'
      sys.exit(0)


    #Numtraj=(last_traj+1)-first_traj
    for i in xrange(ntraj):
#        ls=os.listdir(os.getcwd())
#        numfile=len(ls)
#        k=i+first_traj
#        string=str(k).rjust(5, '0')
#        trajfolder=None
#        j=0
#        for j in range(numfile):
#            trajfolder=re.search(string,str(ls[j]))
#            if trajfolder!=None:
#               break
#        trajcheck=None
#        if trajfolder !=None:
#           trajcheck=re.search('TRAJ',str(ls[j]))
#        if trajcheck !=None:
           print 'Reading trajectory ' + str(files[i]) + ' ...'
           folder_name = str(files[i])[:-10]
           trajectory = traj_manip.trajectory(folder_name, ref_struc, dt=dt)
    
           # actual normal mode analysis
           try:
               nma_list = trajectory.normal_mode_analysis(nma_mat, ref_struc, header=header, out_file=folder_name+'/nma_'+descr+'.txt', abs_list = abs_list)[:num_steps] # +++ abs_list should actually only be considered when averaging
           except:
               print ' *** Error: Coordinate transformation failed for trajectory ' + str(i) + '. Is there a proper calculation?'
               print ' Trajectory skipped ...'
               #not_list += [i]
           else:
           # addition for total std and for trajectory specific average and std
               tm_traj_av = file_handler.table_maker([35]+num_vib*[20])
               tm_traj_av.write_line(['Nr']+header[0])
               tm_traj_av.write_line(['Wavenumber (1/cm)']+header[1])
               tm_traj_av.write_line(['Period (fs)']+header[2])
               
               tm_traj_std = file_handler.table_maker([35]+num_vib*[20])
               tm_traj_std.write_line(['Nr']+header[0])
               tm_traj_std.write_line(['Wavenumber (1/cm)']+header[1])
               tm_traj_std.write_line(['Period (fs)']+header[2])
               for ii,interv in enumerate(ana_ints):
                   st = interv[0]
                   en = interv[1]
                   np = nma_list[st:en].shape[0]
                   num_points[ii] += np
                   sa = numpy.add.reduce(nma_list[st:en]) # add the values in one column vector
                   sum_array[ii] += sa
                   sqa = numpy.add.reduce(nma_list[st:en]**2)
                   sum_sq_array[ii] += sqa
                   
                   # determine average and std for this trajectory and interval
                   if not np == 0:
                       exp = sa / np
                       exp2 = sqa / np
                       std_array = (np/(np-1)*(exp2 - exp**2))**.5 # empirical standard deviation
                       
                       tm_traj_av.write_line([str(st)+'-'+str(en)] + exp.tolist())
                       tm_traj_std.write_line([str(st)+'-'+str(en)] + std_array.tolist())
                   
               # output of trajectory specific information
               tm_traj_av.write_to_file(str(folder_name)+'/nma_'+descr+'_av.txt')
               tm_traj_std.write_to_file(str(folder_name)+'/nma_'+descr+'_std.txt')
           
               # addition for time resolved trajectory averages
               for nr, tstep in enumerate(numpy.array(nma_list)):
                   cross_num_array[nr] += 1
                   cross_sum_array[nr] += tstep
                   cross_sum_sq_array[nr] += tstep**2
            
    #for i in not_list:
     #   print 'TRAJ' + str(i),
         
    print 'Processing data ...'
    for inum,num in enumerate(cross_num_array):
        if num == 0:
            print '*** WARNING: No trajectory found for step %i. Will perform analysis only until step %i.' % (inum,inum-1)
            num_steps=inum-1
            break
            #sys.exit()
            
    # determine the total standard deviation
    tm_tot_std = file_handler.table_maker([35]+num_vib*[20])
    tm_tot_std.write_line(['Nr']+header[0])
    tm_tot_std.write_line(['Wavenumber (1/cm)']+header[1])
    tm_tot_std.write_line(['Period (fs)']+header[2])
    
    for i,interv in enumerate(ana_ints):
        exp_x = sum_array[i] / num_points[i] # expected values of x and x**2
        exp_x2 = sum_sq_array[i] / num_points[i]
        std = (num_points[i]/(num_points[i]-1)*exp_x2 - exp_x**2)**.5 # empirical standard deviation
        
        std = std * mult_array
        
        tm_tot_std.write_line([str(interv[0])+'-'+str(interv[1])] + std.tolist())
        
    tm_tot_std.write_to_file(out_dir + '/total_std.txt')
    
    # time resolved average curves
    tm_mean = file_handler.table_maker(num_vib*[20])
    tm_std = file_handler.table_maker(num_vib*[20])
    
    tm_mean.write_line(header[0])
    tm_std.write_line(header[0])
    tm_mean.write_line(header[1])
    tm_std.write_line(header[1])
    tm_mean.write_line(header[2])
    tm_std.write_line(header[2])
    

    std_list = [0 for i in xrange(num_vib)]
    
    for i in xrange(num_steps):
            tm_mean.write_line([coor/cross_num_array[i] for coor in cross_sum_array[i]])
            cross_mean_array[i] = cross_sum_array[i] / cross_num_array[i] 
            for j in xrange(num_vib):
                exp_x = cross_sum_array[i,j]/cross_num_array[i]
                exp_x2 = cross_sum_sq_array[i,j]/cross_num_array[i]
                std_list[j] = (cross_num_array[i]/(cross_num_array[i]-1)*exp_x2 - exp_x**2)**.5
                
            tm_std.write_line(std_list)
    
    tm_mean.write_to_file(out_dir + '/mean_against_time.txt')
    tm_std.write_to_file(out_dir + '/std_against_time.txt')
    
    # get the variance of the time averaged structures
    tm_av_var = file_handler.table_maker([35]+num_vib*[20])
    tm_av_var.write_line(['Nr']+header[0])
    tm_av_var.write_line(['Wavenumber (1/cm)']+header[1])
    tm_av_var.write_line(['Period (fs)']+header[2])
    
    for st,en in ana_ints:
        av_exp = numpy.add.reduce(cross_mean_array[st:en]) / (en-st) # the way python defines this the field really has en-st entries, not en-st+1
        av_exp2 = numpy.add.reduce(cross_mean_array[st:en]**2) / (en-st)
        av_std_array = ((en-st)/(en-st-1)*(av_exp2 - av_exp**2))**.5 # empirical standard deviation
    
        av_std_array = av_std_array * mult_array
    
        tm_av_var.write_line([str(st)+'-'+str(en)] + av_std_array.tolist())
        
    tm_av_var.write_to_file(out_dir + '/cross_av_std.txt')
    
    print 'Data processing finished.'
    
    if plot: plot_summary(INFOS)
    
def plot_summary(INFOS):
    # plotting

    descr=INFOS['descr']
    out_dir = os.path.join('NMA',descr)
    num_steps=INFOS['numsteps']

    if plot_possible:
        print 'Drawing plots ...'
        
        # Plots for time dependent cross averages
        plotting.mean_std_from_files(mean_file=out_dir+'/mean_against_time.txt',out_dir=out_dir+'/time_plots',xlist=[dt * i for i in xrange(num_steps)],std_file=out_dir+'/std_against_time.txt')
        
        # Bar graphs with the standard deviation of time dependent cross averages
        plotting.bars_from_file(in_file=out_dir+'/total_std.txt', out_dir=out_dir+'/bar_graphs/total_std')
        plotting.bars_from_file(in_file=out_dir+'/cross_av_std.txt', out_dir=out_dir+'/bar_graphs/cross_av_std')
    else:
        print 'Plotting not possible'
        
def plot_1traj(ind, modes_list=None):
    """
    Script for plotting the analysis for one trajectory with index ind.
    """
    if plot_possible:
        print 'Drawing plots for trajectory ' + str(ind) + ' ...'
        plotting.mean_std_from_files(mean_file='TRAJ'+str(ind)+'/RESULTS/nma_'+descr+'.txt',out_dir='TRAJ'+str(ind)+'/RESULTS/nma_plots/'+descr,xlist=[dt * i for i in xrange(num_steps)],col_list=modes_list)
    else:
        print 'Plotting not possible'
    
def main():
    displaywelcome()
    open_keystrokes()

    INFOS=get_general()
    if len(sys.argv) == 1:
        nm_analysis(INFOS)
    else:
       if sys.argv[1] == 'plot':
            if len(sys.argv) == 2:
               plot_summary(INFOS)
            else:
                if 'modes' in sys.argv:
                    modes_ind = sys.argv.index('modes')
                    modes_list = [eval(mode) for mode in sys.argv[modes_ind + 1:]]
                else:
                    modes_ind = None
                    modes_list = None
                if sys.argv[2] == 'all':
                    plot_list = range(first_traj, last_traj + 1)
                else:
                    plot_list = sys.argv[2:modes_ind]
                for ind in plot_list:
                    plot_1traj(ind, modes_list=modes_list)
                    
    close_keystrokes()
                
if __name__=='__main__':
  try:
    main()
  except KeyboardInterrupt:
    print '\nCtrl+C makes me a sad SHARC ;-(\n'
    quit(0)

#print modes_ind, plot_list, modes_list
