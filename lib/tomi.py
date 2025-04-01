#from sympy.physics.wigner import wigner_3j

#S1 = 1/2
#M1 = 1/2
#S2 = 1
#M2 = 0
#S3 = 3/2
#M3 = -1/2
#
#print(S1, M1, S2, M2, S3, M3)
#print(wigner_3j(S1,S2,S3,M1,M2,M3).evalf())


import numpy as np
N = 200
mos = np.zeros((N,N))
rho = np.zeros((2,5,5,N,N))

path = np.einsum_path('ab,sijbc,cd->sijad', mos,rho,mos, optimize=True)
print(path[0])
print(path[1])

