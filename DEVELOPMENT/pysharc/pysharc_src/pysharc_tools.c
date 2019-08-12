/*
 * @author: Maximilian F.S.J. Menger
 * @date: 18.04.2018
 * @version: 0.1.1 
 *
 * Python Wrapper for the SHARC LIBRARY
 *
 */

#include <complex.h>
#include "pysharc_tools.h"

/* CLEAR MEMORY */
void clear_double(int N, double * vec)
{
    for (int i = 0; i<N; i++){
        *(vec + i) = 0.0;
    }
}

void clear_complex_double(int N, complex double * vec)
{
    for (int i = 0; i<N; i++){
        *(vec + i) = 0.0;
    }
}

/* SET VECTOR ELEMENTS */

/* GRADIENT */
void set_gradient(double * gradient, int NAtoms, int IState, double * state_gradient, double scale)
{
    int ishift = IState*(3*NAtoms);
    if (scale == 1.0) {
        for (int i=0; i<3*NAtoms; i++){
            *(gradient + ishift + i) = *(state_gradient + i); 
        }
    } else {
        for (int i=0; i<3*NAtoms; i++){
            *(gradient + ishift + i) = *(state_gradient + i) * scale; 
        }
    }
}
void set_gradient_in_sharc_order(double * gradient, int NAtoms, int NStates, int IState, double * state_gradient, double scale)
{
    if (scale == 1.0) {
        for (int i=0; i<NAtoms; i++){
            for (int j=0; j<3; j++){
            *(gradient + j*NStates*NAtoms + i*NStates+ IState) = *(state_gradient + i*3 + j); 
            }
        }
    } else {
        for (int i=0; i<NAtoms; i++){
            for (int j=0; j<3; j++){
            *(gradient + j*NStates*NAtoms + i*NStates+ IState) = *(state_gradient + i*3 + j) * scale; 
            }
        }
    }
}
/* NACdr */
void set_nacdr(double * nac, int NAtoms,  int NStates,
        int IState, int JState, double * nac_i_j)
{
    int ishift = IState*(NStates*3*NAtoms)+JState*(3*NAtoms);
    for (int i=0; i<3*NAtoms; i++){
        *(nac + ishift + i) = *(nac_i_j + i); 
    }
}
/* to set state */
void set_nacdr_in_sharc_order(double * nac, 
        int NAtoms, int NStates, int IState, int JState, double * nac_i_j)
{
    for (int i=0; i<NAtoms; i++){
        for (int j=0; j<3; j++){
        *(nac + j*NStates*NStates*NAtoms + i*NStates*NStates  + 
                JState*NStates + IState) = *(nac_i_j + i*3 + j); 
        }
    }
}

