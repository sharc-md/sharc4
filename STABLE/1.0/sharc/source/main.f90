!> # Main program
!> \author Sebastian Mai
!> \date 10.07.2014
!> 
!> This is the main code of SHARC
!> It contains the following tasks:
!> - Reading of the input files (in read_input)
!> - Performing all steps of the initialization:
!>   - allocation
!>   - calling the interface for the initial quantum chemistry calculation
!>   - digesting the output of the interface (mixing gradients)
!>   - initial write-out
!> - Main loop of the dynamics:
!>   - Velocity Verlet (nuclear coordinates)
!>   - calling the interface
!>   - digesting the interface output (mixing gradients, phase adjustment)
!>   - electronic wavefunction propagation
!>   - surface hopping
!>   - Calculation of energy (kinetic, potential, total)
!>   - writing of output
!> - Timing
program sharc
use definitions
use electronic
use electronic_laser
use input
use matrix
use misc
use nuclear
use qm
use restart
use output
implicit none

!> \param traj Contains all data which would be private to each trajectory in an ensemble
type(trajectory_type) :: traj
!> \param ctrl Contains all data which would be shared in an ensemble
type(ctrl_type) :: ctrl
!> \param i_step Loop variable for the dynamics loop
integer :: i_step
!> \param time Define the integer function time()
integer :: time



! open(0,file='output.err',status='replace',action='write')

traj%time_start=time()
traj%time_last=traj%time_start

call read_input(traj,ctrl)
call allocate_lapack(ctrl%nstates)

if (.not.ctrl%restart) then
  call write_list_header(u_lis)
  call do_initial_qm(traj,ctrl)
  call Mix_gradients(traj,ctrl)
  call Update_old(traj)
  call Calculate_etot(traj,ctrl)
  call set_time(traj)
  call write_dat(u_dat, traj, ctrl)
  call write_list_line(u_lis,traj,ctrl)
  call write_geom(u_geo, traj, ctrl)
  call write_restart_ctrl(u_resc,ctrl)
  call write_restart_traj(u_rest,ctrl,traj)
  call mkdir_restart(ctrl)
endif


! everything is set up for the loop
do i_step=traj%step+1,ctrl%nsteps
  traj%step=i_step
  call write_logtimestep(u_log,i_step)

  ! Velocity Verlet x
  call VelocityVerlet_xstep(traj,ctrl)
  ! QM Calculation
  call do_qm_calculations(traj,ctrl)
  ! Adjust Phases
  call Adjust_phases(traj,ctrl)
  ! Mix Gradients
  call Mix_gradients(traj,ctrl)
  ! Velocity Verlet v    (before SH)
  call VelocityVerlet_vstep(traj,ctrl)
  if (ctrl%dampeddyn/=1.d0) call Damp_Velocities(traj,ctrl)
  traj%Ekin=Calculate_ekin(ctrl%natom, traj%veloc_ad, traj%mass_a)
!   call Calculate_etot(traj,ctrl)
  ! Propagation
  if (ctrl%laser==0) then
    call propagate(traj,ctrl)
  else
    call propagate_laser(traj,ctrl)
  endif
  ! SH
  call surface_hopping(traj,ctrl)
  ! Decoherence
  call Decoherence(traj,ctrl)
  ! Rescale v
  call Rescale_Velocities(traj,ctrl)
  call Calculate_etot(traj,ctrl)
  ! obtain the correct gradient
  if (ctrl%calc_grad>=1) call redo_qm_gradients(traj,ctrl)
  if (traj%kind_of_jump/=0) call Mix_gradients(traj,ctrl)
  ! Finalization: Variable update, Output, Restart File, Consistency Checks
  call Calculate_cMCH(traj,ctrl)
  call Update_old(traj)
  call set_time(traj)
  call write_list_line(u_lis,traj,ctrl)
  call write_dat(u_dat, traj, ctrl)
  call write_geom(u_geo, traj, ctrl)
  ! write_restart_traj must be the last command
  call write_restart_traj(u_rest,ctrl,traj)
  call myflush()
  ! kill trajectory 
  call kill_after_relaxation(traj,ctrl)
  if ((ctrl%killafter>0).and.(traj%steps_in_gs>ctrl%killafter)) exit
  ctrl%restart=.false.
enddo



call write_final(traj)































endprogram
