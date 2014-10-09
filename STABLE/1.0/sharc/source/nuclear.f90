module nuclear
 contains

! ===========================================================

subroutine VelocityVerlet_xstep(traj,ctrl)
  use definitions
  use matrix
  implicit none
  type(trajectory_type) :: traj
  type(ctrl_type) :: ctrl
  integer :: iatom, idir

  if (printlevel>2) then
    write(u_log,*) '============================================================='
    write(u_log,*) '              Velocity Verlet -- X-step'
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

subroutine VelocityVerlet_vstep(traj,ctrl)
  use definitions
  use matrix
  implicit none
  type(trajectory_type) :: traj
  type(ctrl_type) :: ctrl
  integer :: iatom, idir

  if (printlevel>2) then
    write(u_log,*) '============================================================='
    write(u_log,*) '              Velocity Verlet -- V-step'
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

real*8 function Calculate_ekin(n, veloc, mass) result(Ekin)
  implicit none
  integer, intent(in) :: n
  real*8,intent(in) :: veloc(n,3), mass(n)
  integer :: i

  Ekin=0.d0
  do i=1,n
    ! sum of square can be written as sum(a**2)
    Ekin=Ekin + 0.5d0*mass(i)*sum(veloc(i,:)**2)
  enddo


endfunction

! ===========================================================

subroutine Calculate_etot(traj,ctrl)
  use definitions
  implicit none
  type(trajectory_type) :: traj
  type(ctrl_type) :: ctrl

  traj%Ekin=Calculate_ekin(ctrl%natom, traj%veloc_ad, traj%mass_a)
  traj%Epot=real(traj%H_diag_ss(traj%state_diag,traj%state_diag))
  traj%Etot=traj%Ekin+traj%Epot

  if (printlevel>2) then
    write(u_log,*) '============================================================='
    write(u_log,*) '                        Energies'
    write(u_log,*) '============================================================='
    write(u_log,'(A,1X,F14.9,1X,A)') 'Ekin:',traj%Ekin*au2ev,'eV'
    write(u_log,'(A,1X,F14.9,1X,A)') 'Epot:',traj%Epot*au2ev,'eV'
    write(u_log,'(A,1X,F14.9,1X,A)') 'Etot:',traj%Etot*au2ev,'eV'
    write(u_log,*) ''
  endif


endsubroutine

! ===========================================================

subroutine Rescale_velocities(traj,ctrl)
  use definitions
  use matrix
  implicit none
  type(trajectory_type) :: traj
  type(ctrl_type) :: ctrl
  real*8 :: factor, sum_kk, sum_vk, deltaE
  integer :: i

  if (printlevel>2) then
    write(u_log,*) '============================================================='
    write(u_log,*) '                     Velocity Rescaling'
    write(u_log,*) '============================================================='
  endif
  select case (traj%kind_of_jump)
    case (0)
      if (printlevel>2) write(u_log,'(A)') 'No jump occured.'
    case (1)
      select case (ctrl%ekincorrect)
        case (0)
          if (printlevel>2) write(u_log,*) 'Velocity is not rescaled after surface hop.'
        case (1)
          factor=sqrt( (traj%Etot-real(traj%H_diag_ss(traj%state_diag,traj%state_diag)))/traj%Ekin )
          traj%veloc_ad=traj%veloc_ad*factor
          if (printlevel>2) then
            write(u_log,'(A)') 'Velocity is rescaled along velocity vector.'
            write(u_log,'(A,1X,F12.9)') 'Scaling factor is ',factor
          endif
        case (2)
          call available_ekin(ctrl%natom,&
          &traj%veloc_ad,real(traj%gmatrix_ssad(traj%state_diag_old, traj%state_diag,:,:)),&
          &traj%mass_a, sum_kk, sum_vk)
          deltaE=4.d0*sum_kk*(traj%Etot-traj%Ekin-&
          &real(traj%H_diag_ss(traj%state_diag,traj%state_diag)))+sum_vk**2
          if (sum_vk<0.d0) then
            factor=(sum_vk+sqrt(deltaE))/2.d0/sum_kk
          else
            factor=(sum_vk-sqrt(deltaE))/2.d0/sum_kk
          endif
          do i=1,3
            traj%veloc_ad(:,i)=traj%veloc_ad(:,i)-factor*&
            &real(traj%gmatrix_ssad(traj%state_diag_old, traj%state_diag,:,i))/traj%mass_a(:)
          enddo
! ! ! ! !           ! TODO: The Gmatrix contains the SCALED NACs, but this is ok, since only the unit vector of the NAC vector is used in the following
! ! ! ! !           ! TODO: The rescaling of the velocities is most probably wrong      25.4: Fix attempt: still wrong
! ! ! ! !           do i=1,ctrl%natom
! ! ! ! !             scaled_nac(i,:)=sqrt(traj%mass_a(i))*real(traj%gmatrix_ssad(traj%state_diag_old, traj%state_diag,i,:))
! ! ! ! !             scaled_vel(i,:)=sqrt(traj%mass_a(i))*traj%veloc_ad(i,:)
! ! ! ! !           enddo
! ! ! ! !           call project_a_on_b(ctrl%natom, scaled_vel, scaled_nac, veloc_par)
! ! ! ! !           ekin_par=0.5d0*sum(veloc_par**2)
! ! ! ! !           veloc_ort=scaled_vel-veloc_par
! ! ! ! !           ekin_ort=0.5d0*sum(veloc_ort**2)
! ! ! ! ! !           write(0,*) ekin_par, ekin_ort, ekin_par+ekin_ort, traj%Ekin
! ! ! ! !           factor=sqrt( (traj%H_diag_ss(traj%state_diag_old,traj%state_diag_old)-traj%Epot+ekin_par)/ekin_par )
! ! ! ! !           do i=1,ctrl%natom
! ! ! ! !             traj%veloc_ad(i,:)=(veloc_ort(i,:)+factor*veloc_par(i,:))/sqrt(traj%mass_a(i))
! ! ! ! !           enddo
          if (printlevel>2) then
            write(u_log,'(A)') 'Velocity is rescaled along non-adiabatic coupling vector.'
            write(u_log,'(A,1X,F12.6)') 'Scaling factor is ',factor
          endif
        case (3)
        if (printlevel>2) write(u_log,*) 'Velocity is not rescaled after resonant surface hop.'
      endselect
    case (2)
      if (printlevel>2) write(u_log,'(A)') 'Frustrated jump.'
  endselect

endsubroutine

! ===========================================================

subroutine available_ekin(natom,veloc_ad,nac_ad,mass_a, sum_kk, sum_vk)
  ! calculates the following sums:
  ! sum_kk=1/2*sum_atom nac_atom*nac_atom/mass_atom
  ! sum_vk=sum_atom vel_atom*nac_atom
  implicit none
  integer, intent(in) :: natom
  real*8, intent(in) :: veloc_ad(natom,3), nac_ad(natom,3), mass_a(natom)
  real*8, intent(out) :: sum_kk, sum_vk

  integer :: idir

  sum_kk=0.d0
  sum_vk=0.d0
  do idir=1,3
    sum_kk=sum_kk+0.5d0*sum( nac_ad(:,idir)*nac_ad(:,idir)/mass_a(:) )
    sum_vk=sum_vk+      sum( nac_ad(:,idir)*veloc_ad(:,idir) )
  enddo

endsubroutine

! ===========================================================

subroutine Damp_velocities(traj,ctrl)
  use definitions
  implicit none
  type(trajectory_type) :: traj
  type(ctrl_type) :: ctrl

  if (printlevel>2) then
    write(u_log,*) '============================================================='
    write(u_log,*) '                     Velocity Damping'
    write(u_log,*) '============================================================='
    write(u_log,*)
    write(u_log,*) 'Factor for the velocities is',sqrt(ctrl%dampeddyn)
  endif
  traj%veloc_ad=traj%veloc_ad*sqrt(ctrl%dampeddyn)

endsubroutine

! ===========================================================

















endmodule