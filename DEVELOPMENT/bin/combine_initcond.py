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
      sys.exit(14)
    f.close()
  except IOError:
    print 'Could not write to file %s!' % (filename)
    sys.exit(15)


new=readfile("initconds")    		   ## name of initconds file from amber_to_initconds.py
original=readfile("initconds_original")    ## name of initconds file with wigner QM velocities
qmatoms=input("Number of QM atoms:")

velocities=[]
for i in range(len(original)):
   if "Atoms" in original[i]:
     geom_veloc=[]
     for j in range(qmatoms):
	line=original[i+1+j].split()
    #    print line
        geom_veloc.append(line[6:9])
   #  print geom_veloc
     velocities.append(geom_veloc)
#print velocities[0]
number=0
outputstring=""
for i in range(len(new)):
   if "Atoms" in new[i]:
      for j in range(qmatoms):
	line=new[i+1+j].split()
        line[6:9]=velocities[number][j]
        #new[i+1+j]="  ".join(line)
        new[i+1+j]="%2s%6.1f % 12.8f % 12.8f % 12.8f % 12.8f % 12.8f % 12.8f % 12.8f\n" %(line[0],float(line[1]),float(line[2]),float(line[3]),float(line[4]),float(line[5]),float(line[6]),float(line[7]),float(line[8]))
      number+=1
   outputstring+=new[i]

writefile("initconds_new", outputstring)
