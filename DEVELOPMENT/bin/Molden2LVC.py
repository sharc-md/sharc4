#!/usr/bin/python

import numpy
import sys, os

sys.path = [os.path.join(os.environ['SHARC'],'..','lib')]
import vib_molden

print "Molden2LVC.py <molden_file>"
mfilen = sys.argv[1]

numpy.set_printoptions(precision=5, suppress=True, threshold=50)#

vmol = vib_molden.vib_molden()
vmol.read_molden_file(mfilen)

Vc = vmol.ret_vib_matrix().T

massl = []
for line in open('geom', 'r'):
    massl += 3 * [line.split()[-1]]

#print 'Vc'
#print Vc
#print '\n Vc^T Vc'
#print numpy.dot(Vc.T, Vc)

massv = numpy.array(massl, float)**.5
VmwR = massv[...,None] * Vc
Mred = numpy.dot(VmwR.T, VmwR)
# The diagonal elements are the reduced masses.
#   The off-diagonal elements are numerical noise.
print '\nReduced masses'
print Mred
offnorm = numpy.sum((Mred - numpy.diag(Mred.diagonal()) )**2)
print 'Squared norm of off-diagonal elements:', offnorm
assert offnorm < 1.e-6

# Obtain normalized mass-weighted coordinates
#   Vmw = sqrt(M) * Vc * sqrt(Mred^-1)
#Vmw = numpy.dot(VmwR, numpy.diag(Mred.diagonal()**(-.5)) )

# Do this by a Lowdin orthogonalization using the SVD?
#   This leads to very similar results as above and assures that the matrix is orthogonal
(U, sqrlam, Vt) = numpy.linalg.svd(VmwR)
#print sqrlam**2
# These are again the reduced masses (but in a different order)
Vmw = numpy.dot(U, Vt)

VTV = numpy.dot(Vmw.T, Vmw)
offnorm = numpy.sum( (VTV - numpy.identity(len(massl)))**2 )
print 'Mass-weighted V, deviation from orthonormality:', offnorm
assert offnorm < 1.e-6

numpy.savetxt('V.txt', Vmw)
