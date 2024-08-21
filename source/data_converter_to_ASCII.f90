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


!> # Program DATA_EXTRACTOR.X
!> \authors Sebastian Mai, Philipp Marquetand
!> \date 09.03.2017
!>
!> This program reads the output.dat file of a trajectory and calculates various
!> properties per timestep, which are printed in plottable tables.
!>
!> Usage: '$SHARC/data_extractor.x <output.dat>'
!> No further files are necessary, the output.dat file contains all relevant data.
!>
!> If a file "Reference/QM.out" exists, the overlap matrix from this file is read 
!> and used as reference overlap for along-trajectory-diabatization.
!>
!> Output files:
!> - energy.out 
!> - fosc.out
!> - coeff_diab.out
!> - coeff_MCH.out
!> - coeff_diag.out
!> - spin.out
!> - prop.out
!> - expec.out
!> 
!> Additionally, the output file <input.file>.ext contains build infos of the data_extractor program
program data_extractor
  use definitions_NetCDF
  use data_extractor_NetCDFmodule, only: Tgeneral_infos, Tshdata, Twrite_options, Tprop_info
  use data_extractor_NetCDFmodule, only: print_usage, get_commandline_input  
  use data_extractor_NetCDFmodule, only: write_build_info
  use data_extractor_NetCDFmodule, only: read_data_file
  use data_extractor_NetCDFmodule, only: mk_output_folder, write_output_file_headers, open_output_files
  use data_extractor_NetCDFmodule, only: initialize_data, read_properties_from_output
  use data_extractor_NetCDFmodule, only: write_data_to_file
  use data_extractor_NetCDFmodule, only: process_data
  use matrix
  implicit none
  !> # Parameters: unit numbers for all files
  include 'u_parameter_data_extractor.inc'
  !> # Information which is constant throughout all timesteps
  type(Tprop_info)     :: prop_info
  type(Tgeneral_infos) :: general_infos
  type(Twrite_options) :: write_options
  type(Tsharc_ncoutput) :: ncdat
  integer              :: nstates
  double precision     :: Energy(3)
  !> # Information which is updated per time step
  !> Most of these are equivalent to their definition in definitions.f90
  type(Tshdata) :: shdata
  integer :: istep, nsteps
  ! helper
  character*8000 :: filename, string
  integer :: io, u_new
  ! Command line argument processing
  call get_commandline_input(filename, write_options)
  ! Open dat file and write build info
  call write_build_info(filename, u_info, u_dat)
  ! Read dat file header
  call read_data_file(prop_info, write_options, shdata, general_infos, nstates, u_dat)
  ! create output directory "output_data"
  ! call mk_output_folder()
  ! open output files
  ! call open_output_files(write_options)
  ! write output file headers
  ! call write_output_file_headers(nstates, write_options)
  ! Initialize data
  call initialize_data(nstates, general_infos, shdata, write_options)
  ! =============================================================================================
  ! Main loop
  ! =============================================================================================
  write(6,*) 
  write(6,*) 'Running...'
  istep = 0 
  
  ! open copy of output.dat
  u_new=121
  open(unit=u_new, file='output.dat.cp', status='replace', action='write')

  ! copy header
  rewind(u_dat)
  do
    read(u_dat,'(A)') string
    write(u_new,'(A)') trim(string)
    if (index(string,trim('End of header array data')).ne.0) then
     exit
    endif
  enddo

  do 
    call read_sharc_ncoutputdat_istep(nsteps, istep, general_infos%natom, nstates, &
       &  shdata%H_MCH_ss, shdata%U_ss, shdata%DM_ssd, shdata%overlaps_ss,&
       &  shdata%coeff_diag_s, Energy, shdata%hopprob_s, &
       &  shdata%geom_ad, shdata%veloc_ad,&
       &  shdata%randnum, shdata%state_diag, shdata%state_MCH, shdata%time_step,&
       &  ncdat)

   shdata%Etot = Energy(1)
   shdata%Epot = Energy(2)
   shdata%Ekin = Energy(3)

!    call read_properties_from_output(nstates, step, u_dat, general_infos, prop_info, shdata, io)
!    if (io/=0) exit
    ! ========== Reading is done for this time step =============
    call process_data(nstates, istep, general_infos, write_options, shdata)
    ! ========== Calculating is done for this time step =============
    ! call write_data_to_file(nstates, istep, general_infos, write_options, shdata)
    
    write(u_new,'(A)') '! 0 Step'
    write(u_new,'(I12)') istep
    call matwrite(nstates, shdata%H_MCH_ss, u_new, '! 1 Hamiltonian (MCH) in a.u.', 'E21.13e3')
    call matwrite(nstates, shdata%U_ss, u_new, '! 2 U matrix', 'E21.13e3')
    call matwrite(nstates, shdata%DM_ssd(:,:,1), u_new, '! 3 Dipole moments X (MCH) in a.u.', 'E21.13e3')
    call matwrite(nstates, shdata%DM_ssd(:,:,2), u_new, '! 3 Dipole moments Y (MCH) in a.u.', 'E21.13e3')
    call matwrite(nstates, shdata%DM_ssd(:,:,3), u_new, '! 3 Dipole moments Z (MCH) in a.u.', 'E21.13e3')
    if (prop_info%have_overlap==1) then
      call matwrite(nstates, shdata%overlaps_ss, u_new, '! 4 Overlap matrix (MCH)', 'E21.13e3')
    endif

    call vecwrite(nstates, shdata%coeff_diag_s, u_new, '! 5 Coefficients (diag)','E21.13e3')
    call vecwrite(nstates, shdata%hopprob_s, u_new, '! 6 Hopping Probabilities (diag)','E21.13e3')

    write(u_new,'(A)') '! 7 Ekin (a.u.)'
    write(u_new,'(E21.13e3)') shdata%Ekin
    write(u_new,'(A)') '! 8 states (diag, MCH)'
    write(u_new,'(I12,1X,I12)') shdata%state_diag, shdata%state_MCH
    write(u_new,'(A)') '! 9 Random number'
    write(u_new,'(E21.13e3)') shdata%randnum
    write(u_new,'(A)') '! 10 Runtime (sec)'
    write(u_new,'(I12)') shdata%time_step

    call vec3write(general_infos%natom, shdata%geom_ad, u_new, '! 11 Geometry in a.u.','E21.13e3')
    call vec3write(general_infos%natom, shdata%veloc_ad, u_new, '! 12 Velocities in a.u.','E21.13e3')




    ! write progress to screen
    write(*,'(A,A,F9.2,A)',advance='no') achar(13), 't=',shdata%time_step*general_infos%dtstep,' fs'

    istep = istep + 1
    if (istep == nsteps) then
        exit
    endif
  enddo
  call close_ncfile(ncdat%id)
  write(*,*) "closed ncfile"
endprogram


