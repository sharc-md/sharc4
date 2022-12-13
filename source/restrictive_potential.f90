!******************************************
!
!    SHARC Program Suite
!
!    Copyright (c) 2019 University of Vienna
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

!> # Module RESTRICTIVE_POTENTIAL
!>
!> \author Brigitta Bachmair
!> \date 28.02.2022
!> 
!> This module provides additional restrictive potentials to be applied 
!> on top of the qm gradient.
!> These include the droplet potential (s.t. solvent remains as a
!> droplet and doesn't diffuse) and a tethering potential to keep (part
!> of) atom in center of droplet.
!>

module restrictive_potential
  contains
! ==================================================================================================
! ==================================================================================================
! ==================================================================================================

! modifies gradient to include restrictive droplet potential
subroutine restrict_droplet(traj,ctrl)
  use definitions
  implicit none
  type(trajectory_type) :: traj
  type(ctrl_type) :: ctrl
  real*8 :: radius, dist, force
  integer :: iatom, idir
  do iatom=1,ctrl%natom
    if (ctrl%sel_restricted_droplet(iatom) .eqv. .true.) then
      radius = sqrt(traj%geom_ad(iatom,1)**2 + traj%geom_ad(iatom,2)**2 + traj%geom_ad(iatom,3)**2)
      ! droplet_radius - radius -> inverse distance for inverse force
      dist = radius - ctrl%restricted_droplet_radius
      if (dist > 0.) then
        force = ctrl%restricted_droplet_force * dist
        do idir=1,3
          ! add force * direction (normalized distance)
          traj%grad_ad(iatom,idir) = traj%grad_ad(iatom,idir) + force * (traj%geom_ad(iatom,idir)/radius)
        enddo
      endif
    endif
  enddo
endsubroutine

! modifies gradient to tether atom
subroutine tether_atom(traj,ctrl)
  use definitions
  implicit none
  type(trajectory_type) :: traj
  type(ctrl_type) :: ctrl
  integer :: i, iatom, idir
  real*8 :: radius, dist, force
  
  do i=1,size(ctrl%tether_at)
    iatom = ctrl%tether_at(i)
    radius = sqrt((traj%geom_ad(iatom,1) - traj%tethering_pos(1))**2 + (traj%geom_ad(iatom,2) - traj%tethering_pos(2))**2 +&
            & (traj%geom_ad(iatom,3) - traj%tethering_pos(3))**2)
    dist = radius - ctrl%tethering_radius
    if (dist > 0.) then
      force = ctrl%tethering_force * dist / radius
      do idir=1,3
        traj%grad_ad(iatom,idir) = traj%grad_ad(iatom,idir) + force * (traj%geom_ad(iatom,idir)-traj%tethering_pos(idir))
      enddo
    endif
  enddo

endsubroutine

! calculates the center of mass for given atoms to tether (in ctrl%sel_teter_at)
function calc_centerofmass(traj,ctrl) result(centerofmass)
  use definitions
  implicit none
  type(trajectory_type), intent(in) :: traj
  type(ctrl_type), intent(in) :: ctrl
  real*8 :: centerofmass(3), mass
  integer :: iatom, idir

  !allocate(centerofmass(3))
  centerofmass = 0.0
  mass=0
  do iatom = 1,size(ctrl%tether_at)
    mass = mass + traj%mass_a(ctrl%tether_at(iatom))
    do idir = 1,3
      centerofmass(idir) = centerofmass(idir) + traj%geom_ad(ctrl%tether_at(iatom),idir) * traj%mass_a(ctrl%tether_at(iatom))
    enddo
  enddo
  do idir = 1,3
    centerofmass(idir) = centerofmass(idir)/mass
  enddo
endfunction

endmodule

