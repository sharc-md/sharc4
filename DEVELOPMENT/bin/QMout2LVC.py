#!/usr/bin/python

import os
import sys

sys.path = [os.environ['SHARC']] + sys.path
import SHARC_LVC

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

# ======================================================================= #

def LVC_complex_mat(header, mat, deldiag=False, oformat=' % .7e'):
    rnonzero = False
    inonzero = False

    rstr = header + ' R\n'
    istr = header + ' I\n'
    for i in range(len(mat)):
        for j in range(len(mat)):
            val = mat[i][j].real
            if deldiag and i==j:
                val = 0.
            rstr += oformat%val
            if val*val > pthresh: rnonzero = True

            val = mat[i][j].imag
            if deldiag and i==j:
                val = 0.
            istr += oformat%val
            if val*val > pthresh: inonzero = True

        rstr += '\n'
        istr += '\n'

    retstr = ''
    if rnonzero: retstr += rstr
    if inonzero: retstr += istr

    return retstr

# ======================================================================= #

def main():
    QMin = SHARC_LVC.read_QMin()
    targets = ['h', 'dm', 'grad', 'nacdr']
    QMout = read_QMout('QM.out', QMin['nmstates'], QMin['natom'], targets)

    for sti in SHARC_LVC.itnmstates(QMin['states']):
        imult, istate, ims = sti
        if imult != 1 and imult != 3:
            print "ERROR: only singlets and triplets supported (for now)"
            sys.exit()

    wf = open('LVC.template', 'w')
    wf.write('V0.txt\n')
    for state in QMin['states']:
        wf.write('%i '%state)
    wf.write('\n')

    wf.write('epsilon\n')
    wf.write('%i\n'%QMin['nstates'])
    inm = 0
    eref = QMout['h'][0][0]
    for imult, nmult in enumerate(QMin['states']):
        for istate in range(nmult):
            wf.write( "%3i %3i % .10f\n"%(imult+1, istate+1, (QMout['h'][inm][inm]-eref).real) )
            inm += 1

# ------------------------------------------------------------------------- #

    SH2LVC = {}
    SHARC_LVC.read_V0(QMin, SH2LVC)
    r3N = range(3*QMin['natom'])

    # OVM is the full transformation matrix from Cartesian to dimensionless
    #   mass-weighted coordinates
    OVM = [[0. for i in r3N] for j in r3N]
    for ixyz in r3N:
        for imode in r3N:
            if SH2LVC['Om'][imode] > 1.e-6:
                OVM[ixyz][imode] = SH2LVC['Om'][imode]**(-.5) * SH2LVC['V'][ixyz][imode] / SH2LVC['Ms'][ixyz]

# ------------------------------------------------------------------------- #

    wf.write('kappa\n')
    nkappa = 0
    kstr = ''
    for i, sti in enumerate(SHARC_LVC.itnmstates(QMin['states'])):
        imult, istate, ims = sti
        if imult == 3 and ims >= 0.: break
        gradi = []
        for comp in QMout['grad'][i]:
            gradi += comp
        for imode in r3N:
            kappa = sum(OVM[ixyz][imode]*gradi[ixyz] for ixyz in r3N)
            if kappa*kappa > pthresh:
                kstr += "%3i %3i %5i % .5e\n"%(imult, istate, imode+1, kappa)
                nkappa += 1
    wf.write('%i\n'%nkappa)
    wf.write(kstr)

# ------------------------------------------------------------------------- #

    wf.write('lambda\n')
    nlam = 0
    lstr = ''
    for i, sti in enumerate(SHARC_LVC.itnmstates(QMin['states'])):
        imult, istate, ims = sti
        if imult == 3 and ims >= 0.: break
        for j, stj in enumerate(SHARC_LVC.itnmstates(QMin['states'])):
            jmult, jstate, jms = stj
            if jmult == 3 and jms >= 0.: break
            if j <= i: continue

            nacij = []
            for comp in QMout['nacdr'][i][j]:
                nacij += comp
            for imode in r3N:
                dE = (QMout['h'][j][j]-QMout['h'][i][i]).real
                lam = sum(OVM[ixyz][imode]*nacij[ixyz] for ixyz in r3N) * dE
                if lam*lam > pthresh:
                    lstr += "%3i %3i %3i %5i % .5e\n"%(imult, istate, jstate, imode+1, lam)
                    nlam += 1
    wf.write('%i\n'%nlam)
    wf.write(lstr)

# ------------------------------------------------------------------------- #

    wf.write( LVC_complex_mat('SOC', QMout['h'], deldiag=True) )
    wf.write( LVC_complex_mat('DMX', QMout['dm'][0]) )
    wf.write( LVC_complex_mat('DMY', QMout['dm'][1]) )
    wf.write( LVC_complex_mat('DMZ', QMout['dm'][2]) )

# ------------------------------------------------------------------------- #

    wf.close()
    print 'File %s written.'%wf.name

# ======================================================================= #


if __name__ == "__main__":
    pthresh = 1.e-7**2
    main()
