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
  character*8000 :: filename
  integer :: io 
  ! Command line argument processing
  call get_commandline_input(filename, write_options)
  ! Open dat file and write build info
  call write_build_info(filename, u_info, u_dat)
  ! Read dat file header
  call read_data_file(prop_info, write_options, shdata, general_infos, nstates, u_dat)
  ! create output directory "output_data"
  call mk_output_folder()
  ! open output files
  call open_output_files(write_options)
  ! write output file headers
  call write_output_file_headers(nstates, write_options)
  ! Initialize data
  call initialize_data(nstates, general_infos, shdata, write_options)
  ! =============================================================================================
  ! Main loop
  ! =============================================================================================
  write(6,*) 
  write(6,*) 'Running...'
  istep = 0 
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
    call write_data_to_file(nstates, istep, general_infos, write_options, shdata)
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


