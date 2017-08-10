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

subroutine afssh_step(traj,ctrl)
  use definitions
  implicit none
  type(trajectory_type) :: traj
  type(ctrl_type) :: ctrl
  integer :: nstates, istate

  nstates = ctrl%nstates

  if (printlevel>2) then
    write(u_log,*) 'Computing A-FSSH decoherence'
  endif

  ! Propagate the auxiliary trajectories
  do istate = 1, nstates
    print *, 'prop', istate
    !traj%aux_trajs(istate)%weight = traj%coeff_diag_old_s(istate)^2
    ! Propagate the xstep with the density-matrix-weighted gradient from the last time step
    call VelocityVerlet_xstep_afssh(traj%aux_trajs(istate), ctrl)
        
    ! Compute the new density-matrix-weighted gradient and propagate
    ! TODO: transform!!
    traj%aux_trajs(istate)%grad_ad  = (real(traj%Gmatrix_ssad(istate,istate,:,:)) - traj%grad_ad) &
      & * traj%coeff_diag_old_s(istate)**2
    call VelocityVerlet_vstep_afssh(traj%aux_trajs(istate), ctrl)

    ! store the density-matrix-weighted gradient to be used in the next time step
    traj%aux_trajs(istate)%grad_ad  = (real(traj%Gmatrix_ssad(istate,istate,:,:)) - traj%grad_ad) &
      & * traj%coeff_diag_s(istate)**2


    ! transform to the diabatic basis
    enddo
    
endsubroutine

! ===========================================================

subroutine reset_moments(traj,ctrl)
    use definitions
    implicit none
    type(trajectory_type) :: traj
    type(ctrl_type) :: ctrl
    integer :: istate

    print *, 'reset 1'
    do istate = 1, ctrl%nstates
      traj%aux_trajs(istate)%geom_ad = 0.d0
      traj%aux_trajs(istate)%veloc_ad = 0.d0
      traj%aux_trajs(istate)%accel_ad = 0.d0
      traj%aux_trajs(istate)%grad_ad = 0.d0
    enddo
    print *, 'reset 2'
endsubroutine

! ===========================================================

!> performs the geometry update of the Velocity Verlet algorithm
!> a(t)=g(t)/M
!> x(t+dt)=x(t)+v(t)*dt+0.5*a(t)*dt^2
subroutine VelocityVerlet_xstep_afssh(traj,ctrl)
  use definitions
  use matrix
  implicit none
  type(aux_trajectory_type) :: traj
  type(ctrl_type) :: ctrl
  integer :: iatom, idir

  if (printlevel>2) then
    write(u_log,*) '============================================================='
    write(u_log,'(A,1X,I3,1X,A)') 'Velocity Verlet (A-FSSH), state', traj%istate, '-- X-step'
    write(u_log,*) '============================================================='
    call vec3write(ctrl%natom,traj%accel_ad,u_log,'Old accel','F12.9')
    call vec3write(ctrl%natom,traj%geom_ad,u_log,'Old geom','F12.7')
  endif

  do iatom=1,ctrl%natom
    do idir=1,3
      traj%accel_ad(iatom,idir)=&
      &-traj%grad_ad(iatom,idir)/traj%mass_a(iatom)

      traj%geom_ad(iatom,idir)=&
      & traj%geom_ad(iatom,idir)&
      &+traj%veloc_ad(iatom,idir)*ctrl%dtstep&
      &+0.5d0*traj%accel_ad(iatom,idir)*ctrl%dtstep**2
    enddo
  enddo

  if (printlevel>2) then
    call vec3write(ctrl%natom,traj%accel_ad,u_log,'accel','F12.9')
    call vec3write(ctrl%natom,traj%geom_ad,u_log,'geom','F12.7')
  endif

endsubroutine

! ===========================================================

!> performs the velocity update of the Velocity Verlet algorithm
!> a(t+dt)=g(t+dt)/M
!> v(t+dt)=v(t)+a(t+dt)*dt
subroutine VelocityVerlet_vstep_afssh(traj,ctrl)
  use definitions
  use matrix
  implicit none
  type(aux_trajectory_type) :: traj
  type(ctrl_type) :: ctrl
  integer :: iatom, idir

  if (printlevel>2) then
    write(u_log,*) '============================================================='
    write(u_log,'(A,1X,I3,1X,A)') 'Velocity Verlet (A-FSSH), state', traj%istate, '-- V-step'
    write(u_log,*) '============================================================='
    call vec3write(ctrl%natom,traj%accel_ad,u_log,'Old accel','F12.9')
    call vec3write(ctrl%natom,traj%veloc_ad,u_log,'Old veloc','F12.9')
  endif

  do iatom=1,ctrl%natom
    do idir=1,3
      traj%accel_ad(iatom,idir)=0.5d0*(traj%accel_ad(iatom,idir)&
      &-traj%grad_ad(iatom,idir)/traj%mass_a(iatom) )

      traj%veloc_ad(iatom,idir)=&
      & traj%veloc_ad(iatom,idir)&
      &+traj%accel_ad(iatom,idir)*ctrl%dtstep
    enddo
  enddo

  if (printlevel>2) then
    call vec3write(ctrl%natom,traj%accel_ad,u_log,'accel','F12.9')
    call vec3write(ctrl%natom,traj%veloc_ad,u_log,'veloc','F12.9')
  endif

endsubroutine

! ===========================================================
 
endmodule