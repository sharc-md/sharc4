#!/usr/bin/python

import os

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

# =========================================================
def read_QMin():
  # reads the geometry, unit keyword, nstates keyword
  # does not read the request keywords, since it calculates by default all quantities
  QMin={}
  f=open('QM.in')
  qmin=f.readlines()
  f.close()

  QMin['natom']=int(qmin[0])
  QMin['comment']=qmin[1]

  # get geometry
  line=qmin[2].split()[1:4]
  geom=[ [ float(line[i]) for i in range(3) ] ]

  geom=[]
  for i in range(2,QMin['natom']+2):
    line=qmin[i].split()
    for j in range(3):
      line[j+1]=float(line[j+1])
    geom.append(line)

  # find states keyword
  for line in qmin:
    s=line.split()
    if len(s)==0:
      continue
    if 'states' in s[0].lower():
      states=[]
      for iatom in range(len(s)-1):
        states.append(int(s[iatom+1]))
      break
  else:
    print 'No state keyword given!'
    sys.exit(15)
  nstates=0
  nmstates=0
  for mult,i in enumerate(states):
    nstates+=i
    nmstates+=(mult+1)*i
  QMin['states']=states
  QMin['nstates']=nstates
  QMin['nmstates']=nmstates
  QMin['nmult'] = 0
  statemap={}
  i=1
  for imult,nstates in enumerate(states):
    if nstates==0:
      continue
    QMin['nmult'] += 1
    for ims in range(imult+1):
      ms=ims-imult/2.
      for istate in range(nstates):
        statemap[i]=[imult+1,istate+1,ms]
        i+=1
  QMin['statemap']=statemap

  # find unit keyword
  factor=1.
  for line in qmin:
    s=line.split()
    if len(s)==0:
      continue
    if 'unit' in s[0].lower():
      if not 'bohr' in s[1].lower():
        factor=BOHR_TO_ANG
  for i in range(QMin['natom']):
    for j in range(3):
      geom[i][j+1]/=factor
  QMin['geom']=geom

  # find forbidden keywords and optional keywords
  QMin['init'] = False
  for line in qmin:
    s=line.lower().split()
    if len(s)==0:
      continue
    if 'nacdr' in s[0]:
      QMin['nacdr'] = True
    if 'nacdt' in s[0]:
      print 'NACDT is not supported!'
      sys.exit(16)
    if 'dmdr' in s[0]:
      QMin['dmdr']=[]
    if s[0] == 'init':
      QMin['init'] = True

  # add request keywords
  QMin['soc']=[]
  QMin['dm']=[]
  QMin['grad']=[]
  QMin['overlap']=[]
  QMin['pwd']=os.getcwd()
  return QMin

# ======================================================================= #

def read_QMout(path,nstates,natom,request):
  targets={'h':         {'flag': 1,
                         'type': complex,
                         'dim':  (nstates,nstates)},
           'dm':        {'flag': 2,
                         'type': complex,
                         'dim':  (3,nstates,nstates)},
           'grad':      {'flag': 3,
                         'type': float,
                         'dim':  (nstates,natom,3)},
           'nacdr':      {'flag': 5,
                         'type': float,
                         'dim':  (nstates,nstates,natom,3)}
          }

  # read QM.out
  lines=readfile(path)

  # obtain all targets
  QMout={}
  for t in targets:
    if t in request:
      iline=-1
      while True:
        iline+=1
        if iline>=len(lines):
          print 'Could not find target %s with flag %i in file %s!' % (t,targets[t]['flag'],path)
          sys.exit(11)
        line=lines[iline]
        if '! %i' % (targets[t]['flag']) in line:
          break
      values=[]
      # =========== single matrix
      if len(targets[t]['dim'])==2:
        iline+=1
        for irow in range(targets[t]['dim'][0]):
          iline+=1
          line=lines[iline].split()
          if targets[t]['type']==complex:
            row=[ complex(float(line[2*i]),float(line[2*i+1])) for i in range(targets[t]['dim'][1]) ]
          elif targets[t]['type']==float:
            row=[ float(line[i]) for i in range(targets[t]['dim'][1]) ]
          values.append(row)
      # =========== list of matrices
      elif len(targets[t]['dim'])==3:
        for iblocks in range(targets[t]['dim'][0]):
          iline+=1
          block=[]
          for irow in range(targets[t]['dim'][1]):
            iline+=1
            line=lines[iline].split()
            if targets[t]['type']==complex:
              row=[ complex(float(line[2*i]),float(line[2*i+1])) for i in range(targets[t]['dim'][2]) ]
            elif targets[t]['type']==float:
              row=[ float(line[i]) for i in range(targets[t]['dim'][2]) ]
            block.append(row)
          values.append(block)
      # =========== matrix of matrices
      elif len(targets[t]['dim'])==4:
        for iblocks in range(targets[t]['dim'][0]):
          sblock=[]
          for jblocks in range(targets[t]['dim'][1]):
            iline+=1
            block=[]
            for irow in range(targets[t]['dim'][2]):
              iline+=1
              line=lines[iline].split()
              if targets[t]['type']==complex:
                row=[ complex(float(line[2*i]),float(line[2*i+1])) for i in range(targets[t]['dim'][3]) ]
              elif targets[t]['type']==float:
                row=[ float(line[i]) for i in range(targets[t]['dim'][3]) ]
              block.append(row)
            sblock.append(block)
          values.append(sblock)
      QMout[t]=values

  #pprint.pprint(QMout)
  return QMout

if __name__ == "__main__":
    import sys

    print "QMout2LVC.py <V.txt>"
    NMfile = sys.argv[1]

    qmi = read_QMin()
    targets = ['h', 'dm', 'grad', 'nacdr']
    qmo = read_QMout('QM.out', qmi['nmstates'], qmi['natom'], targets)

    print 'epsilon'
    print qmi['nstates']
    ival = 0
    eref = qmo['h'][0][0]
    for imult, nmult in enumerate(qmi['states']):
        for istate in range(nmult):
            print "%3i %3i % .10f"%(imult+1, istate+1, (qmo['h'][ival][ival]-eref).real)
            ival += 1

    V = [[float(v) for v in line.split()] for line in open(NMfile, 'r').readlines()]
