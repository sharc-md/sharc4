from numba import jit, prange, set_num_threads,objmode
#from numba import typeof
import numba
import numpy as np
import time

@jit(nopython=True,cache=True,fastmath=True) 
def deltaS0( tCI, nst, dets, CI, mos ): # all dets in list dets have equal number of alpha and equal number of beta electrons
    ndets = len(dets[:,0])
    nmos = len(dets[0,:])
    keep = np.array([ i for i in range(ndets) if np.any( np.abs(CI[i,:]) >= tCI ) ])
    with objmode(CInew='f8[:,:]',detsnew='i8[:,:]'):
        CInew = np.take(CI, keep, axis=0)
        detsnew = np.take(dets, keep, axis=0) 
    CI = CInew
    dets = detsnew
    ndets = len(dets) 
    mos_alpha = [ np.array([ mo for mo in range(nmos) if dets[i,mo] == 7 or dets[i,mo] == 5 ]) for i in range(ndets) ]
    mos_beta = [ np.array([ mo for mo in range(nmos) if dets[i,mo] == 7 or dets[i,mo] == 1 ]) for i in range(ndets) ]

    pairs_a = []
    pairs_b = []
    with objmode(t1='f8'):
        t1 = time.perf_counter()
    for i in range(ndets):
        for j in range(i+1,ndets):
            diffs = dets[i,:] - dets[j,:]
            orbs = np.nonzero(diffs)[0]
            if len(orbs) == 2:
                mo1, mo2 = orbs
                if diffs[mo1] == 6 and diffs[mo2] == -6:
                    phase = np.nonzero( mos_alpha[i] == mo1 )[0][0] + np.nonzero( mos_alpha[j] == mo2 )[0][0] #+ 1
                    pairs_a.append([ i, j, mo1, mo2, phase ])
                    #  pairs_a.append([ j, i, mo2, mo1, phase ])
                elif diffs[mo2] == 6 and diffs[mo1] == -6:
                    phase = np.nonzero( mos_alpha[j] == mo1 )[0][0] + np.nonzero( mos_alpha[i] == mo2 )[0][0] #+ 1 
                    pairs_a.append([ i, j, mo2, mo1, phase ])
                    #  pairs_a.append([ j, i, mo1, mo2, phase ])
                elif diffs[mo1] == 2 and diffs[mo2] == -2:
                    phase = np.nonzero( mos_beta[i] == mo1 )[0][0] + np.nonzero( mos_beta[j] == mo2 )[0][0] #+ 1 
                    pairs_b.append([ i, j, mo1, mo2, phase ])
                    #  pairs_b.append([ j, i, mo2, mo1, phase ])
                elif diffs[mo2] == 2 and diffs[mo1] == -2:
                    phase = np.nonzero( mos_beta[j] == mo1 )[0][0] + np.nonzero( mos_beta[i] == mo2 )[0][0] #+ 1
                    pairs_b.append([ i, j, mo2, mo1, phase ])
                    #  pairs_b.append([ j, i, mo1, mo2, phase ])

    with objmode():
        t2 = time.perf_counter()
        print('Time in SD-TDM generation = ', t2-t1,flush=True)

    with objmode(t1='f8'):
        t1 = time.perf_counter()
    for i in range(ndets):
        amos = mos_alpha[i]
        bmos = mos_beta[i]
        for mo in amos:
            pairs_a.append( [ i, i, mo, mo, 0] )
        for mo in bmos:
            pairs_b.append( [ i, i, mo, mo, 0])
    with objmode():
        t2 = time.perf_counter()
        print('Time in diagonal generation = ', t2-t1,flush=True)

    with objmode(t1='f8'):
        t1 = time.perf_counter()
    rho = np.zeros((2,nst,nst,nmos,nmos))
    for p in pairs_a:
        outer =np.outer(CI[p[0],:], CI[p[1],:]) 
        rho[0,:,:,p[2],p[3]] += (-1.)**p[4]*np.ascontiguousarray(outer)
        if p[0] != p[1]:
            #  #  pass
            rho[0,:,:,p[3],p[2]] += (-1.)**p[4]*np.ascontiguousarray(outer.T)
    for p in pairs_b:
        outer =np.outer(CI[p[0],:], CI[p[1],:]) 
        rho[1,:,:,p[2],p[3]] += (-1.)**p[4]*np.ascontiguousarray(outer)#outer 
        if p[0] != p[1]:
            #  #  pass
            rho[1,:,:,p[3],p[2]] += (-1.)**p[4]*np.ascontiguousarray(outer.T)
    with objmode(rho='f8[:,:,:,:,:]'):
        t2 = time.perf_counter()
        rho = np.ascontiguousarray(rho)
        print('Time in rho generation = ', t2-t1,flush=True)

    with objmode(rho='f8[:,:,:,:,:]'):
        t1 = time.perf_counter()
        rho = np.einsum('ia,smnab,bj->smnij',mos,rho,mos.T,optimize=['einsum_path',(0,1),(0,1)],casting='no')
        t2 = time.perf_counter()
        print('Time in rho rotation = ', t2-t1,flush=True)
    return rho 

@jit(nopython=True,cache=True,fastmath=True) 
def deltaS1( tCI, nst1, nst2, dets1, dets2, CI1, CI2, mos1, mos2 ): # all dets in dets1 have two alpha electrons less then those in dets2
    ndets1 = len(dets1[:,0])
    ndets2 = len(dets2[:,0])
    nmos1 = len(dets1[0,:])
    nmos2 = len(dets2[0,:])
    print(nmos1,nmos2)
    keep1 = np.array([ i for i in range(ndets1) if np.any( np.abs(CI1[i,:]) >= tCI ) ])
    keep2 = np.array([ i for i in range(ndets2) if np.any( np.abs(CI2[i,:]) >= tCI ) ])
    with objmode(CInew1='f8[:,:]',detsnew1='i8[:,:]',CInew2='f8[:,:]',detsnew2='i8[:,:]'):
        CInew1 = np.take(CI1, keep1, axis=0)
        detsnew1 = np.take(dets1, keep1, axis=0) 
        CInew2 = np.take(CI2, keep2, axis=0)
        detsnew2 = np.take(dets2, keep2, axis=0) 
    CI1 = CInew1
    CI2 = CInew2
    dets1 = detsnew1
    dets2 = detsnew2
    ndets1 = len(dets1) 
    ndets2 = len(dets2) 
    mos_alpha_1 = [ np.array([ mo for mo in range(nmos1) if dets1[i,mo] == 7 or dets1[i,mo] == 5 ]) for i in range(ndets1) ]
    mos_alpha_2 = [ np.array([ mo for mo in range(nmos2) if dets2[i,mo] == 7 or dets2[i,mo] == 5 ]) for i in range(ndets2) ]
    mos_beta_1 = [ np.array([ mo for mo in range(nmos1) if dets1[i,mo] == 7 or dets1[i,mo] == 1 ]) for i in range(ndets1) ]
    mos_beta_2 = [ np.array([ mo for mo in range(nmos2) if dets2[i,mo] == 7 or dets2[i,mo] == 1 ]) for i in range(ndets2) ]

    pairs = []
    with objmode(t1='f8'):
        t1 = time.perf_counter()
    for i in range(ndets1):
        for j in range(ndets2):
            diffs = dets1[i,:] - dets2[j,:]
            orbs = np.nonzero(diffs)[0]
            if len(orbs) == 2:
                mo1, mo2 = orbs
                if diffs[mo1] == 2 and diffs[mo2] == -6:
                    phase = np.nonzero( mos_beta_1[i] == mo1 )[0][0] + np.nonzero( mos_alpha_2[j] == mo2 )[0][0] 
                    pairs.append([ i, j, mo1, mo2, phase ])
                elif diffs[mo1] == -6 and diffs[mo2] == 2:
                    phase = np.nonzero( mos_beta_1[i] == mo2 )[0][0] + np.nonzero( mos_alpha_2[j] == mo1 )[0][0] 
                    pairs.append([ i, j, mo2, mo1, phase ])
    with objmode():
        t2 = time.perf_counter()
        print('Time in SD-TDM generation = ', t2-t1,flush=True)

    with objmode(t1='f8'):
        t1 = time.perf_counter()
    rho = np.zeros((nst1,nst2,nmos1,nmos2))
    for p in pairs:
        outer =np.outer(CI1[p[0],:], CI2[p[1],:]) 
        rho[:,:,p[2],p[3]] += (-1.)**p[4]*outer 
    with objmode(rho='f8[:,:,:,:]'):
        t2 = time.perf_counter()
        rho = np.ascontiguousarray(rho)
        print('Time in rho generation = ', t2-t1,flush=True)

    with objmode(rho='f8[:,:,:,:]'):
        t1 = time.perf_counter()
        print(np.shape(mos1),np.shape(rho), np.shape(mos2))
        rho = np.einsum('ia,mnab,bj->mnij',mos1,rho,mos2.T,optimize=['einsum_path',(0,1),(0,1)],casting='no')
        t2 = time.perf_counter()
        print('Time in rho rotation = ', t2-t1,flush=True)
    return rho 

