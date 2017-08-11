!> # Module DECOHERENCE A-FSSH
!> 
!> \author Felix Plasser
!> \date 10.08.2017
!>
!> This module contains subroutines that allow to perform a decoherence
!> correction according to the approximated augmented fewest-switches
!> surface hopping method as described in
!> Jain, Alguire, Subotnik JCTC 2016, 12, 5256.

module decoherence_afssh
  implicit none

  private
   
  public :: afssh_step, allocate_afssh

contains

! ===========================================================

subroutine allocate_afssh(traj,ctrl)
    use definitions
    implicit none
    type(trajectory_type) :: traj
    type(ctrl_type) :: ctrl
    integer :: allocst
    integer :: natom, nstates, istate

    natom=ctrl%natom
    nstates = ctrl%nstates
    
    if (printlevel>1) then
        write(u_log,*) 'Initializing A-FSSH algorithm'
    endif
    
    !allocate(traj%aux_trajs(nstates), stat=allocst)
    !if (allocst/=0) stop 'Could not allocate aux_trajs'
    
    print *, 'starting allocation'
    do istate = 1, nstates
      print *, 'allocation', istate, nstates
      
      !allocate(traj%aux_trajs(istate)%mass_a(natom), traj%aux_trajs(istate)%geom_ad(natom,3),&
      !&traj%aux_trajs(istate)%veloc_ad(natom,3),traj%aux_trajs(istate)%accel_ad(natom,3),&
      !&traj%aux_trajs(istate)%grad_ad(natom,3), stat=allocst)
      !if (allocst/=0) stop 'Could not allocate'

      allocate(traj%aux_trajs(istate)%mass_a(natom),stat=allocst)
      if (allocst/=0) stop 'Could not allocate mass_a'
      traj%aux_trajs(istate)%mass_a = traj%mass_a
      
      allocate(traj%aux_trajs(istate)%geom_ad(natom,3),stat=allocst)
      if (allocst/=0) stop 'Could not allocate geom_ad'
      
      allocate(traj%aux_trajs(istate)%veloc_ad(natom,3),stat=allocst)
      if (allocst/=0) stop 'Could not allocate veloc_ad'
      
      allocate(traj%aux_trajs(istate)%accel_ad(natom,3),stat=allocst)
      if (allocst/=0) stop 'Could not allocate accel_ad'
      
      allocate(traj%aux_trajs(istate)%grad_ad(natom,3),stat=allocst)
      if (allocst/=0) stop 'Could not allocate grad_ad'
      !
      traj%aux_trajs(istate)%istate = istate
    enddo
    
    call reset_moments(traj,ctrl)
    print *, 'alloc afssh finished'
endsubroutine

! ===========================================================

subroutine reset_moments(traj,ctrl)
    use definitions
    implicit none
    type(trajectory_type) :: traj
    type(ctrl_type) :: ctrl
    integer :: istate

    if (printlevel>2) then
      write(u_log, *)'Resetting all A-FSSH moments'
    endif
    do istate = 1, ctrl%nstates
      traj%aux_trajs(istate)%geom_ad = 0.d0
      traj%aux_trajs(istate)%veloc_ad = 0.d0
      traj%aux_trajs(istate)%accel_ad = 0.d0
      traj%aux_trajs(istate)%grad_ad = 0.d0
    enddo
    
endsubroutine

! ===========================================================

subroutine afssh_step(traj,ctrl)
  use definitions
  implicit none
  type(trajectory_type) :: traj
  type(ctrl_type) :: ctrl
  
  integer :: nstates, istate
  real*8 :: refnorm

  nstates = ctrl%nstates

  if (printlevel>2) then
    write(u_log,*) 'Computing A-FSSH decoherence'
  endif

  if ((traj%kind_of_jump==1).or.(traj%kind_of_jump==3))then
    call reset_moments(traj,ctrl)
    return ! do not propagate moments in case of hop, correct?
  endif
  
  ! Propagate the auxiliary trajectories
  do istate = 1, nstates
    call afssh_prop(traj, traj%aux_trajs(istate), ctrl)
  enddo

  ! Check that the position of the reference trajectory is 0
  refnorm = sum(traj%aux_trajs(traj%state_diag)%geom_ad * traj%aux_trajs(traj%state_diag)%geom_ad)
  if (refnorm > 1.d-10) then
    write(0,*) 'A-FSSH: refnorm = ', refnorm
    stop 1
  endif

  ! Compute the rates and perform the required steps
  do istate = 1, nstates
    call afssh_rates(traj, traj%aux_trajs(istate), ctrl)
  enddo
    
endsubroutine

! ===========================================================

subroutine afssh_prop(traj, atraj, ctrl)
  use definitions
  implicit none
  type(trajectory_type) :: traj
  type(aux_trajectory_type) :: atraj
  type(ctrl_type) :: ctrl

  integer :: istate
    
  istate = atraj%istate
  
  ! Propagate the xstep with the density-matrix-weighted gradient from the last time step
  call VelocityVerlet_xstep_afssh(atraj, ctrl)
        
  ! Compute the new density-matrix-weighted gradient and propagate
  ! TODO: transform!!
  atraj%grad_ad  = (real(traj%Gmatrix_ssad(istate,istate,:,:)) - traj%grad_ad) &
    & * traj%coeff_diag_old_s(istate)**2
  call VelocityVerlet_vstep_afssh(atraj, ctrl)

  ! store the density-matrix-weighted gradient to be used in the next time step
  atraj%grad_ad  = (real(traj%Gmatrix_ssad(istate,istate,:,:)) - traj%grad_ad) &
    & * traj%coeff_diag_s(istate)**2

  ! transform to the diabatic basis
endsubroutine

! ===========================================================

!> compute the two parts of the decoherence rate
subroutine afssh_rates(traj, atraj, ctrl)
  use definitions
  implicit none
  type(trajectory_type) :: traj
  type(aux_trajectory_type) :: atraj
  type(ctrl_type) :: ctrl
  
  !type(aux_trajectory_type) :: reftraj
  !real*8 :: refnorm
  complex*16 :: clam, cn

  !reftraj = traj%aux_trajs(traj%state_diag)
  !refnorm = sum(reftraj%geom_ad * reftraj%geom_ad)
  !write(u_log,'(A,1X,F14.10)') 'refnorm', refnorm
  
  ! F_nn . (R_nn - R_ll) / 2 hbar
  atraj%rate1 = 0.5 * sum(atraj%grad_ad * atraj%geom_ad)

  ! rate2
  ! if coupling vectors are available
  atraj%rate2 = 2 * abs(sum(real(traj%Gmatrix_ssad(traj%state_diag,atraj%istate,:,:)) * atraj%geom_ad))

  if (printlevel>2) then
    write(u_log, '(A,1X,I3,1X,A,F12.9,A,F12.9)') 'State', atraj%istate, 'rate1: ', atraj%rate1, ' rate2: ', atraj%rate2
  endif
  
  ! decoherence
  if (traj%randnum < ctrl%dtstep * (atraj%rate1 - atraj%rate2)) then
    if (printlevel>2) then
      write(u_log, '(A,1X,I3)') 'Collapsing amplitude of state', atraj%istate
    endif
    cn = traj%coeff_diag_s(atraj%istate)
    clam = traj%coeff_diag_s(traj%state_diag)
    
    traj%coeff_diag_s(atraj%istate) = 0.d0
    traj%coeff_diag_s(traj%state_diag) = clam / abs(clam) * sqrt(abs(clam)**2 + abs(cn)**2)
  endif
    
  ! reset
  if (traj%randnum < ctrl%dtstep * atraj%rate1) then
    if (printlevel>2) then
      write(u_log, '(A,1X,I3)') 'Resetting moments for state', atraj%istate
    endif
    atraj%geom_ad = 0.d0
    atraj%veloc_ad = 0.d0    
  endif

endsubroutine

! ===========================================================

!> performs the geometry update of the Velocity Verlet algorithm
!> a(t)=g(t)/M
!> x(t+dt)=x(t)+v(t)*dt+0.5*a(t)*dt^2
subroutine VelocityVerlet_xstep_afssh(atraj,ctrl)
  use definitions
  use matrix
  implicit none
  type(aux_trajectory_type) :: atraj
  type(ctrl_type) :: ctrl
  integer :: iatom, idir

  if (printlevel>3) then
    write(u_log,*) '============================================================='
    write(u_log,'(A,1X,I3,1X,A)') 'Velocity Verlet (A-FSSH), state', atraj%istate, '-- X-step'
    write(u_log,*) '============================================================='
    call vec3write(ctrl%natom,atraj%accel_ad,u_log,'Old accel','F12.9')
    call vec3write(ctrl%natom,atraj%geom_ad,u_log,'Old geom','F12.7')
  endif

  do iatom=1,ctrl%natom
    do idir=1,3
      atraj%accel_ad(iatom,idir)=&
      &-atraj%grad_ad(iatom,idir)/atraj%mass_a(iatom)

      atraj%geom_ad(iatom,idir)=&
      & atraj%geom_ad(iatom,idir)&
      &+atraj%veloc_ad(iatom,idir)*ctrl%dtstep&
      &+0.5d0*atraj%accel_ad(iatom,idir)*ctrl%dtstep**2
    enddo
  enddo

  if (printlevel>3) then
    call vec3write(ctrl%natom,atraj%accel_ad,u_log,'accel','F12.9')
    call vec3write(ctrl%natom,atraj%geom_ad,u_log,'geom','F12.7')
  endif

endsubroutine

! ===========================================================

!> performs the velocity update of the Velocity Verlet algorithm
!> a(t+dt)=g(t+dt)/M
!> v(t+dt)=v(t)+a(t+dt)*dt
subroutine VelocityVerlet_vstep_afssh(atraj,ctrl)
  use definitions
  use matrix
  implicit none
  type(aux_trajectory_type) :: atraj
  type(ctrl_type) :: ctrl
  integer :: iatom, idir

  if (printlevel>3) then
    write(u_log,*) '============================================================='
    write(u_log,'(A,1X,I3,1X,A)') 'Velocity Verlet (A-FSSH), state', atraj%istate, '-- V-step'
    write(u_log,*) '============================================================='
    call vec3write(ctrl%natom,atraj%accel_ad,u_log,'Old accel','F12.9')
    call vec3write(ctrl%natom,atraj%veloc_ad,u_log,'Old veloc','F12.9')
  endif

  do iatom=1,ctrl%natom
    do idir=1,3
      atraj%accel_ad(iatom,idir)=0.5d0*(atraj%accel_ad(iatom,idir)&
      &-atraj%grad_ad(iatom,idir)/atraj%mass_a(iatom) )

      atraj%veloc_ad(iatom,idir)=&
      & atraj%veloc_ad(iatom,idir)&
      &+atraj%accel_ad(iatom,idir)*ctrl%dtstep
    enddo
  enddo

  if (printlevel>3) then
    call vec3write(ctrl%natom,atraj%accel_ad,u_log,'accel','F12.9')
    call vec3write(ctrl%natom,atraj%veloc_ad,u_log,'veloc','F12.9')
  endif

endsubroutine

! ===========================================================
 
endmodule