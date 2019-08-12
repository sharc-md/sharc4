/*
 * @author: Maximilian F.S.J. Menger
 * @date: 18.04.2018
 * @version: 0.1.1 
 *
 * Python Wrapper for the SHARC LIBRARY
 *
 */

#ifndef __TOOLS_H_
#define __TOOLS_H_
#ifdef __cplusplus
extern "C" {
#endif
void clear_double(int N, double * vec);
void clear_complex_double(int N, complex double * vec);
void set_gradient(double * gradient, int NAtoms, int IState, double * state_gradient, double scale);
void set_gradient_in_sharc_order(double * gradient, int NAtoms, int NStates, int IState, double * state_gradient, double scale);
void set_nacdr(double * nac, int NAtoms, int NStates,
        int IState, int JState, double * nac_i_j);
void set_nacdr_in_sharc_order(double * nac, 
        int NAtoms, int NStates, int IState, int JState, double * nac_i_j);
#ifdef __cplusplus
}
#endif
#endif
