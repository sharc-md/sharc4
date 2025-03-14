!******************************************
!
!    SHARC Program Suite
!
!    Copyright (c) 2025 University of Vienna
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

!> # Module MISC
!>
!> \author Sebastian mai
!> \date 27.02.2015
!> 
!> This module defines some miscellaneous subroutines for various purposes,
!> which are:
!> - generation of the SHARC fun facts
!> - hashing the input files to generate a unique identifier for this input
!> - updating the runtime in traj
!> - initializing the random number generator
!> 
module misc

integer,parameter :: n_sharcfacts=9             !< number of SHARC facts, needs to be changed when adding new facts
character*1024    :: sharcfacts(n_sharcfacts)   !< array containing the fun facts.

 contains

! ===================================================

!> initializes the SHARC fun facts array. New facts should be added here, while adjusting n_sharcfacts above.
  subroutine init_sharcfacts
    implicit none

    sharcfacts(1) ='SHARC fun fact #1: If you print the source code of SHARC and fold boats out of the paper, SHARC will &
      &actually swim.'
    sharcfacts(2) ='SHARC fun fact #2: If you try to run SHARC backwards, you will get a CRAHS.'
    sharcfacts(3) ='SHARC fun fact #3: Seemingly, some anagrams of SHARC are frowned upon in german-speaking coutries.'
    sharcfacts(4) ='SHARC fun fact #4: SHARC is a common misspelling of a flightless, cartilaginous fish belonging to the &
      &superorder selachimorpha, usually referred to as "sea dogs" until the 16th century.'
    sharcfacts(5) ='SHARC fun fact #42: SHARC is not the ultimate answer to life and everything :('
    sharcfacts(6) ='SHARC fun fact #5: SHARC can detect electromagnetic radiation.'
    sharcfacts(7) ='SHARC fun fact #6: SHARC is not a living fossil.'
    sharcfacts(8) ='SHARC fun fact #7: SHARC is a rather sophisticated random number generator.'
    sharcfacts(9) ='SHARC fun fact #8: In 2014, more people were killed by lightning than by SHARC.'

  endsubroutine

! ===================================================

!> Writes a randomly chosen SHARC fun fact to unit u
!> \param u unit to which the fun fact is written
!> this routine also takes care of linebreaking the fun fact
  subroutine write_sharcfact(u)
    implicit none
    integer, intent(in) :: u
    character*1024 :: str, str2
    real*8 :: r
    integer :: r2, length, breaks=90

    call init_sharcfacts()
    call random_number(r)
    r2=int(n_sharcfacts*r)+1
    str=sharcfacts(r2)
    write(u,*) '------------------------------------------------------------------------------------------'
    do
      length=len(trim(str))
      if (length>breaks) then
        str2=str(1:breaks)
        str=str(breaks+1:1024)
        write(u,*) trim(str2)
      else
        write(u,*) trim(str)
        exit
      endif
    enddo
    write(u,*) '------------------------------------------------------------------------------------------'
  endsubroutine

! ===================================================

!> takes a string of variable length and calculates an integer number which is
!> can be used as a check sum for the input.
!> the calling routine has to take care to concatenate all input files to str.
!> \param str variable length string
  function djb_hash(str) result(res)
    character(len=*),intent(in) :: str
    integer :: hash = 16661     ! some prime
    integer :: i = 0
    integer*8 :: res
 
    do i=1,len(str)
        hash = (ishft(hash,5) + hash) + ichar(str(i:i))
    enddo

    res = abs(hash)

  endfunction DJB_hash

! ===================================================

!> updates in traj the wallclock time of the last timestep 
!> and the time where the last step was completed
  subroutine set_time(traj)
    use definitions
    implicit none
    type(trajectory_type) :: traj
    integer :: time

    traj%time_step=time()-traj%time_last
    traj%time_last=time()

  endsubroutine

! ===================================================

!> Initializes the random number generator 
!> In order to obtain a set of uncorrelated seeds from the single input seed 
!> a two-step procedure is used, which is described in the SHARC manual in detail.
!> \param rngseed a single input seed which is used to generate the set of actual seeds
  subroutine init_random_seed(rngseed)
    implicit none
    integer,intent(in) :: rngseed
    integer :: n,i
    integer,allocatable :: seed(:)
    real*8 :: r

    ! find the number of seeds required
    call random_seed(size=n)
    allocate(seed(n))
    ! calculate a sequence of seeds from rngseed
    do i=1,n
      seed(i)=rngseed+37*i+17*i**2
    enddo
    ! initialize with the first sequence (low quality)
    call random_seed(put=seed)

    ! the elements of seed should be uncorrelated, thus we
    ! calculate some with the random number generator and reseed
    do i=1,n
      call random_number(r)
      seed(i)=int(65536*(r-0.5d0))
    enddo
    call random_seed(put=seed)
!     deallocate(seed)

  endsubroutine

! ===================================================

! the following routine does not work and also not make any sense
!> Initializes the random number generator for themostat
!  subroutine init_random_seed_thermostat(rngseed)
!    use ziggurat
!    implicit none
!    integer,intent(in) :: rngseed
!    integer :: n,i
!    integer :: seed
!    real*8 :: r
!
!    seed=rngseed+37+17**2
!    call zigset(seed)
!
!    r = shr3()
!    seed=int(65536*(r-0.5d0))
!    call zigset(seed)
!
!  endsubroutine

! ===================================================

!> Checks whether the file "STOP" exists in the CWD
  logical function check_stop(cwd)
    use definitions
    implicit none
    character*1023, intent(in) :: cwd
    character*1023 :: filename
    logical :: exists

    ! filename
    filename=trim(cwd)//'/STOP'

    if (printlevel>2) then
      write(u_log,*) 'Inquiring STOP file: "'//trim(filename)//'"'
    endif

    ! inquire the file
    inquire(file=filename, exist=exists)
    check_stop=exists

    if ((exists).and.(printlevel>0)) then
      write(u_log,*) '============================================================='
      write(u_log,*) '                      File STOP detected!'
      write(u_log,*) '============================================================='
    endif

  endfunction

! ===================================================

!> calculates 3 vectors (as one big matrix 3*Natoms x 3) in directions of the total rotations of the system
!> orthonormalize translat. and rot. vectors of system ->
!> substract translat. and other-direction rot. components from angular momentum derivatives for this
!> and normalize (total translation vectors just in x-dir (1 0 0 1 0 0 ...), y/z analogous)
  subroutine get_rotation_tot(ctrl,traj)
    use definitions
    implicit none
    type(trajectory_type) :: traj
    type(ctrl_type) :: ctrl
    integer :: iatom
    real*8 :: d0, d1, d2, d3, d4, norm

    ! set up angular momentum derivative vector nL_x nL_y nL_x
    ! indices: ctrl%rotation_tot(3*(iatom-1)+idir,jdir) with atom iatom, atomic coord. direction idir, global direction jdir 
    do iatom=1,ctrl%natom
      ctrl%rotation_tot(3*(iatom-1)+1,1) = 0. 
      ctrl%rotation_tot(3*(iatom-1)+2,1) = traj%mass_a(iatom)*traj%geom_ad(iatom,3)
      ctrl%rotation_tot(3*(iatom-1)+3,1) = -traj%mass_a(iatom)*traj%geom_ad(iatom,2)
      ctrl%rotation_tot(3*(iatom-1)+1,2) = -traj%mass_a(iatom)*traj%geom_ad(iatom,3)
      ctrl%rotation_tot(3*(iatom-1)+2,2) = 0.
      ctrl%rotation_tot(3*(iatom-1)+3,2) = traj%mass_a(iatom)*traj%geom_ad(iatom,1)
      ctrl%rotation_tot(3*(iatom-1)+1,3) = traj%mass_a(iatom)*traj%geom_ad(iatom,2)
      ctrl%rotation_tot(3*(iatom-1)+2,3) = -traj%mass_a(iatom)*traj%geom_ad(iatom,1)
      ctrl%rotation_tot(3*(iatom-1)+3,3) = 0.
    enddo

    write(*,*) 'rot_tot nL_'
    write(*,*) ctrl%rotation_tot

    ! create corresp. dot products to subtract projected translational components
    !1/(sum m_i) * sum m_i^2 x_i -> entries for scalar products of nL_ and translat.
    d0 = dot_product(traj%mass_a,traj%mass_a)
    d1 = dot_product(traj%mass_a**2 ,traj%geom_ad(:,1))/d0
    d2 = dot_product(traj%mass_a**2 ,traj%geom_ad(:,2))/d0
    d3 = dot_product(traj%mass_a**2 ,traj%geom_ad(:,3))/d0

    !rot_x
    ! subtract projected translational components
    do iatom=1,ctrl%natom
      ctrl%rotation_tot(3*(iatom-1)+2,1) = ctrl%rotation_tot(3*(iatom-1)+2,1) - d3 * traj%mass_a(iatom)
      ctrl%rotation_tot(3*(iatom-1)+3,1) = ctrl%rotation_tot(3*(iatom-1)+3,1) + d2 * traj%mass_a(iatom)
    enddo

    !normalize (check before if norm == 0)
    norm = sqrt(dot_product(ctrl%rotation_tot(:,1),ctrl%rotation_tot(:,1)))
    if (norm==0) then
      write(u_log, *) 'Warning: Rotational x-component is zero!'
      if (ctrl%natom >= 3) then
        stop 'Rotational x-component cannot be zero for molecules with 3 or more atoms!'
      endif
    else
      ctrl%rotation_tot(:,1) = ctrl%rotation_tot(:,1) / norm
    endif 

    ! nL_y*rot_x and nL_z*rot_x
    d4 = dot_product(ctrl%rotation_tot(:,2), ctrl%rotation_tot(:,1))

    !rot_y
    ! subtract projected translational components
    do iatom=1,ctrl%natom
      ctrl%rotation_tot(3*(iatom-1)+1,2) = ctrl%rotation_tot(3*(iatom-1)+1,2) + d3 * traj%mass_a(iatom)
      ctrl%rotation_tot(3*(iatom-1)+3,2) = ctrl%rotation_tot(3*(iatom-1)+3,2) - d1* traj%mass_a(iatom)
    enddo
    ! subtract projected other rotational components
    ctrl%rotation_tot(:,2) = ctrl%rotation_tot(:,2) - d4 * ctrl%rotation_tot(:,1)
    !normalize
    norm = sqrt(dot_product(ctrl%rotation_tot(:,2),ctrl%rotation_tot(:,2)))
    if (norm==0) then
      write(u_log, *) 'Warning: Rotational y-component is zero!'
      if (ctrl%natom >= 3) then
        stop 'Rotational y-component cannot be zero for molecules with 3 or more atoms!'
      endif
    else
      ctrl%rotation_tot(:,2) = ctrl%rotation_tot(:,2) / norm
    endif 
    
    ! nL_z*rot_y
    d3 = dot_product(ctrl%rotation_tot(:,3), ctrl%rotation_tot(:,1))
    d4 = dot_product(ctrl%rotation_tot(:,3),  ctrl%rotation_tot(:,2))

    !rot_z
    ! subtract projected translational components
    do iatom=1,ctrl%natom
      ctrl%rotation_tot(3*(iatom-1)+1,3) = ctrl%rotation_tot(3*(iatom-1)+1,3) - d2 * traj%mass_a(iatom)
      ctrl%rotation_tot(3*(iatom-1)+2,3) = ctrl%rotation_tot(3*(iatom-1)+2,3) + d1 * traj%mass_a(iatom)
    enddo
    ! subtract projected other rotational components
    ctrl%rotation_tot(:,3) = ctrl%rotation_tot(:,3) - d3 * ctrl%rotation_tot(:,1) - d4 * ctrl%rotation_tot(:,2)
    !normalize
    norm = sqrt(dot_product(ctrl%rotation_tot(:,3),ctrl%rotation_tot(:,3)))
    if (norm==0) then
      write(u_log, *) 'Warning: Rotational z-component is zero!'
      if (ctrl%natom >= 3) then
        stop 'Rotational z-component cannot be zero for molecules with 3 or more atoms!'
      endif
    else
      ctrl%rotation_tot(:,3) = ctrl%rotation_tot(:,3) / norm
    endif 

  endsubroutine

! =================================================================== !

endmodule misc
