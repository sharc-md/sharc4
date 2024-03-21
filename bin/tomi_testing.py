import time
import numpy as np

N = 300
M = 3*N
K = 5000
F = np.ones((K,N))
A = np.ones((K,N,N))
L = np.ones((M,N,N))
G = np.zeros((N,N,N,N))
B = np.ones((K,N,N))
path1 = np.einsum_path('nij,ijkl,nkl->n',A,G,B,optimize='optimal')
print('ECI_J = ', path1[0])
path2 = np.einsum_path('nij,iljk,nkl->n',A,G,B,optimize='optimal')
print('ECI_K = ', path2[0])
path3 = np.einsum_path('nij,mij,mkl,nkl->n',A,L,L,B,optimize='optimal')
print('RI-ECI_J = ', path3[0])
path4 = np.einsum_path('nij,mil,mjk,nkl->n',A,L,L,B,optimize='optimal')
print('RI-ECI_K = ', path4[0])
path5 = np.einsum_path('ni,nj,ijkl,nkl->n',F,F,G,B,optimize='optimal')
print('SCT_J = ', path5[0])
path6 = np.einsum_path('ni,nj,iljk,nkl->n',F,F,G,B,optimize='optimal')
print('SCT_K = ', path6[0])
path7 = np.einsum_path('ni,nj,mij,mkl,nkl->n',F,F,L,L,B,optimize='optimal')
print('RI-SCT_J = ', path7[0])
print('RI-SCT_J = ', path7[1])
path8 = np.einsum_path('ni,nj,mil,mjk,nkl->n',F,F,L,L,B,optimize='optimal')
print('RI-SCT_K = ', path8[0])
print('RI-SCT_K = ', path8[1])
A = np.ones((N,N))
path9 = np.einsum_path('ij,kl,il,jk->',A,A,A,A,optimize='optimal')
print('O = ', path9[0])
print('O = ', path9[1])

def get density():
    knowns, from_program, constructable = [], [], []
    while any_new:
        any_new = False
        for pair in dens_requests:
            if not pair in knowns:
                if is_provided_by_program( pair ):
                    from_program.append( pair )
                    any_new = True
                elif is_constructable( pair ):
                    constructable.append( pair )
                    any_new = True
                knowns = from_program + constructable

def is_constructable( pair ):
    m1, ms1, n1, m2, ms2, n2, spin = pair
    for s in ['aa', 'bb', 'ab', 'ba']:
        if any( [ p])s != spin:


     


