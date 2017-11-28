#!/usr/bin/python

import numpy
import sys, os

sys.path = [os.path.join(os.environ['SHARC'],'..','lib')]
import vib_molden

au2rcm = 219474.631

################################################################################

print "Molden2LVC.py <molden_file>"
mfilen = sys.argv[1]

numpy.set_printoptions(precision=5, suppress=True, threshold=50)#

vmol = vib_molden.vib_molden()
vmol.read_molden_file(mfilen)

Vc = vmol.ret_vib_matrix().T

nat = 0
massl = []
for line in open('geom', 'r'):
    massl += 3 * [line.split()[-1]]
    nat += 1

print 'Cartesian: Vc'
print Vc

VcTVc = numpy.dot(Vc.T, Vc)
print '\n Cartesian: Vc^T Vc'
print VcTVc

# Normalize the Cartesian modes
#   -> then the reduced masses are correct
inorms = VcTVc.diagonal()**(-.5)
Vcn = Vc * inorms

massv = numpy.array(massl, float)**.5
VmwR = massv[...,None] * Vcn
Mred = numpy.dot(VmwR.T, VmwR)
# The diagonal elements are the reduced masses.
#   The off-diagonal elements are numerical noise.
print '\nReduced masses (amu)'
print Mred
offnorm = numpy.sum((Mred - numpy.diag(Mred.diagonal()) )**2)
print 'Squared norm of off-diagonal elements:', offnorm
assert offnorm < 0.1

# Obtain normalized mass-weighted coordinates
#   Vmw = sqrt(M) * Vc * sqrt(Mred^-1)
#Vmw = numpy.dot(VmwR, numpy.diag(Mred.diagonal()**(-.5)) )

# Do this by a Lowdin orthogonalization using the SVD?
#   This leads to very similar results as above and assures that the matrix is orthogonal
(U, sqrlam, Vt) = numpy.linalg.svd(VmwR)
#print sqrlam**2
# These are again the reduced masses (but in a different order)
Vmw = numpy.dot(U, Vt)

print '\nMass-weighted: Vmw'
print Vmw

VTV = numpy.dot(Vmw.T, Vmw)
offnorm = numpy.sum( (VTV - numpy.identity(len(massl)))**2 )
print 'Mass-weighted V, deviation from orthonormality:', offnorm
assert offnorm < 1.e-6

numpy.savetxt('V.txt', Vmw)

################################################################################
# Create the input file for SHARC_LVC.py

print "\nCreating template for SH2LVC.inp ... "
wf = open('SH2LVC.inp', 'w')
wf.write('%i\n'%nat)
wf.write('<states>\n')
for line in open('geom', 'r'):
    wf.write(line)
wf.write('Mass-weighted normal modes\n')
wf.write('<path>/V.txt\n')
wf.write('Frequencies\n')
for o in vmol.ret_freqs():
    wf.write('% .10f'%(o/au2rcm))
wf.write('\n')
print 'SH2LVC.inp written.'
