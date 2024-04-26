!******************************************
!
!    SHARC Program Suite
!
!    Copyright (c) 2023 University of Vienna
!
!    This file is part of SHARC.
!
!    SHARC is free software: you can redistribute it and/or modify
!    it under the terms of the GNU General Public License as published by
!    the Free Software Foundation, either version 3 of the License, or
!    (at your option) any later version.
!
!    SHARC is distributed in the hope that it will be useful,
!    but WITHOUT ANY WARRANTY; without even the implied warranty of
!    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!    GNU General Public License for more details.
!
!    You should have received a copy of the GNU General Public License
!    inside the SHARC manual.  If not, see <http://www.gnu.org/licenses/>.
!
!******************************************

program create_laser

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!
!! Simulate arbitrary laser pulses
!! written by Philipp Marquetand
!! www.marquetand.net
!! this version is part of the SHARC suite of programs
!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
  use LASER_definitions
  use LASER_input
  use LASER_calc
  use vector_operations
  implicit none
  
  integer :: ilasers
  integer :: ip_xyz ! index running over the polarization directions x,y,z
 ! integer :: id_xyz ! index running over the gradient directions x,y,z
  integer :: it
!   integer :: NE
!   integer :: max_polarizaton_index(1)
  
  real(kind=8) :: t
!   real(kind=8) :: bandwidth_factor
  real(kind=8), allocatable :: envelope(:)
  real(kind=8), allocatable :: env(:,:)
  real(kind=8), allocatable :: momentary_frequency(:)
  real(kind=8), allocatable :: mom_freq(:,:)
  real(kind=8), allocatable :: wavevector(:)
  real(kind=8),allocatable :: x(:)

  complex(kind=8),allocatable :: laser_efield(:,:)
  complex(kind=8),allocatable :: laser_bfield(:,:)
  complex(kind=8),allocatable :: laser_t(:)
  call read_params
  
  allocate (laser_efield(Nt,3), STAT=allocatestatus)
  if (allocatestatus /= 0) stop "*** Not enough memory 1 ***"
  allocate (laser_bfield(Nt,3), STAT=allocatestatus)
  if (allocatestatus /= 0) stop "*** Not enough memory 2 ***"
  allocate (wavevector(3), STAT=allocatestatus)
  if (allocatestatus /= 0) stop "*** Not enough memory 4 ***"
  allocate (x(3), STAT=allocatestatus)
  if (allocatestatus /= 0) stop "*** Not enough memory 5 ***"
  allocate (laser_t(Nt), STAT=allocatestatus)
  if (allocatestatus /= 0) stop "*** Not enough memory 6 ***"
  allocate (envelope(Nt), STAT=allocatestatus)
  if (allocatestatus /= 0) stop "*** Not enough memory 7 ***"
  allocate (env(Nt,Nlasers), STAT=allocatestatus)
  if (allocatestatus /= 0) stop "*** Not enough memory 8 ***"
  allocate (momentary_frequency(Nt), STAT=allocatestatus)
  if (allocatestatus /= 0) stop "*** Not enough memory 9 ***"
  allocate (mom_freq(Nt,Nlasers), STAT=allocatestatus)
  if (allocatestatus /= 0) stop "*** Not enough memory 10 ***"
  laser_efield = 0.
  laser_bfield = 0.
  do ilasers = 1, Nlasers
    call field_transform(laser_t,type_envelope(ilasers),field_strength(ilasers),fwhm(ilasers), &
                         pulse_begin(ilasers),pulse_center(ilasers),pulse_center2(ilasers), &
                         pulse_end(ilasers),omega_0(ilasers),phase(ilasers),b_1(ilasers), &
                         b_2(ilasers),b_3(ilasers),b_4(ilasers),dt,t0,Nt,envelope,momentary_frequency) 
    env(:,ilasers) = sqrt(dble(laser_t(:))**2+aimag(laser_t(:))**2)
    mom_freq(:,ilasers) = momentary_frequency(:)
    wavevector = cross_product( polarization_e(:, ilasers), polarization_b(:, ilasers))
    do ip_xyz = 1,3
      do it = 1,Nt
        if (abs(laser_t(it)) < threshold(ilasers)) then
          laser_t(it) = 0.
          endif
          laser_efield(it, ip_xyz) = laser_efield(it,ip_xyz) + polarization_e(ip_xyz,ilasers)*laser_t(it)
          x = laser_efield(it, :)
          laser_bfield(it,:) = cross_product( wavevector, x)/speed_of_light_au
      enddo
    enddo
  enddo
  write(6,*) 'Writing out laser field'
  open (10,file='laser')
  ! Write the header information
  write(10, '(a)') ' ! laser file '
  write(10, '(a)') ' ! SHARC 4.0'
  write(10, '(a)') ' ! file_version 2.0'
  write(10, '(a, i0)') ' ! nsteps = ', Nt
  write(10, '(a, 107(es16.8,x))') ' ! dt = ', dt
  write(10, '(a)') ' ! E-field = true'
  write(10, '(a)') ' ! B-field = true'
  write(10, '(a)') ' ! E-field_gradients = false'
  write(10, '(a)') ' ! laser_freq_path = laser_freq'
  write(10, '(A2, A14, A17, A17, A17, A17, A17, A17, A17, A17, A17, A17, A17, A17)') & 
                                        ' # ', 'Time |', 'Re(Ex) |', 'Im(Ex) |', 'Re(Ey) |', 'Im(Ey) |', 'Re(Ez) |', 'Im(Ez) |', & 
                                                         'Re(Bx) |', 'Im(Bx) |', 'Re(By) |', 'Im(By) |', 'Re(Bz) |', 'Im(Bz) |'!,
 !                                                      'Re(Ex_grad_x) |', 'Im(Ex_grad_x) |', 'Re(Ex_grad_y) |', 'Im(Ex_grad_y) |', 
 !                                                      'Re(Ex_grad_z) |', 'Im(Ex_grad_z) |',
 !                                                      'Re(Ey_grad_x) |', 'Im(Ey_grad_x) |', 'Re(Ey_grad_y) |', 'Im(Ey_grad_y) |',
 !                                                      'Re(Ey_grad_z) |', 'Im(Ey_grad_z) |',
 !                                                      'Re(Ez_grad_x) |', 'Im(Ez_grad_x) |', 'Re(Ez_grad_y) |', 'Im(Ez_grad_y) |',
 !                                                      'Re(Ez_grad_z) |', 'Im(Ez_grad_z) |',

 write(10, '(A2, A14, A17, A17, A17, A17, A17, A17, A17, A17, A17, A17, A17, A17)') &
                                       ' # ', '[fs] |', '[a.u.] |', '[a.u.] |', '[a.u.] |', '[a.u.] |', '[a.u.] |', '[a.u.] |', &
                                                        '[a.u.] |', '[a.u.] |', '[a.u.] |', '[a.u.] |', '[a.u.] |', '[a.u.] |'
  write(10, '(a)') ''
  do it = 1,Nt
    t = t0 + (it-1) * dt
    if (realvalued) then
      write(10,'(13(es16.8,x))') t*au2fs, &
                             dble(laser_efield(it,1)), 0.d0, &
                             dble(laser_efield(it,2)), 0.d0, &
                             dble(laser_efield(it,3)), 0.d0, &
                             dble(laser_bfield(it,1)), 0.d0, &
                             dble(laser_bfield(it,2)), 0.d0, &
                             dble(laser_bfield(it,3)), 0.d0
    else
      write(10,'(13(es16.8,x))') t*au2fs, &
                             dble(laser_efield(it,1)), aimag(laser_efield(it,1)), &
                             dble(laser_efield(it,2)), aimag(laser_efield(it,2)), &
                             dble(laser_efield(it,3)), aimag(laser_efield(it,3)), &
                             dble(laser_bfield(it,1)), aimag(laser_bfield(it,1)), &
                             dble(laser_bfield(it,2)), aimag(laser_bfield(it,2)), &
                             dble(laser_bfield(it,3)), aimag(laser_bfield(it,3))
    endif 
  enddo
  close (10)
  
  write(6,*) 'Writing out laser frequency'
  
  open (10,file='laser_freq')
  ! Write the header information
  write(10, '(a)') ' ! Laser freq file'
  write(10, '(a)') ' ! SHARC 4.0'
  write(10, '(a)') ' ! file_version 2.0' 
  write(10, '(a)') ''
  do it = 1,Nt
    t = t0 + (it-1) * dt
    write(10,'(107(e16.8,x))') t*au2fs, &
                            (mom_freq(it,ilasers),ilasers=1,Nlasers)
  enddo
  close (10)

  deallocate (polarization_e)
  deallocate (polarization_b)
  deallocate (wavevector)
  deallocate (type_envelope)
  deallocate (field_strength)
  deallocate (fwhm)
  deallocate (pulse_begin)
  deallocate (pulse_center)
  deallocate (pulse_center2)
  deallocate (pulse_end)
  deallocate (omega_0)
  deallocate (phase)
  deallocate (b_2)
  deallocate (b_3)
  deallocate (b_4)
  deallocate (laser_efield)
  deallocate (laser_bfield)
  deallocate (laser_t)
  
end


