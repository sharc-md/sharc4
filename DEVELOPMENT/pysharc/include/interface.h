/*
 * @author: Maximilian F.S.J. Menger
 * @date: 18.04.2018
 * @version: 0.1.1 
 *
 * Python Wrapper for the SHARC LIBRARY
 *
 * header for interface.f90
 *
 */
#ifndef __INTERFACE_H_
#define __INTERFACE_H_


#ifdef __cplusplus
extern "C" {
#endif
// GET INFO for QMIN etc.
void get_states_(char * string);
void get_dt_(char * string);
void get_savedir_(char * string);
void get_tasks_(char * string, int * icall);
void get_grad_(char * string, int * icall);
void get_nacdr_(char * string, int * icall);
void get_scalingfactor_(double * scale);
void get_constants_(double * consts);
// Molecule info
void get_natoms_(int * natoms);
void get_nsteps_(int * nsteps);
void get_trajstep_(int * nsteps);
// GET COORDINATES
void get_current_coordinates_(int * NAtoms,double * Crd, int * Ang);
void get_element_name_(int * NAtoms, char * value); 
void get_ian_(int * NAtoms, int * IAn);
// set pointer
void setPointers(void ** H, void ** dm, 
                 void ** overlap, 
                 void ** grad,
                 void ** nacs 
                 );
void setQMinPointers(void ** Crd);
void postprocess_qmout_data_(int * IH, int * IDM,
                             int * IGrad, 
                             int * IOverlap,
                             int * INAC);
// SET VALUES
void set_phases_(void);
void set_hamiltonian_(int * N, double complex * H);
void set_dipolemoments_(int * N, double complex * DM);
void set_overlap_(int * N, double complex * overlap);
void set_gradients_(int * N, int * NAtoms, double * grad);
void set_nacs_(int * NStates, int * NAtoms, double * nacs);
void post_process_data_(int * isecond);
// initial qm
void initial_qm_pre_(void);
void initial_qm_post_(void);
// SHARC MAIN ROUTINE
void setup_sharc_(char * input, int * IRestart);
void initial_step_(int * IRestart);
void do_initial_step_2(void);
void verlet_xstep_(int * i_step);
void verlet_vstep_(int * iredo);
void verlet_finalize_(int * IExit, int * iskip);
void finalize_sharc_(void);
void write_restart_(void);
void error_finalize_sharc_(void);
#ifdef __cplusplus
}
#endif
#endif
