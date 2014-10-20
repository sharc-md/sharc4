#******************************************
#
#SHARC Program Suite
#
#Copyright (c) 2014 University of Vienna
#
#******************************************
# Makefile for Sharc 
# Version 1.0
# Oct-16-2014
#
# uncomment the following line and comment the 2nd next line to 
# use the intel fortran compiler.
# F90 = ifort 
F90 = gfortran


# uncomment the following line and comment the 2nd next line to use intel's MKL
# Note: you probably need to adjust the search path if you haven't set
# $(MKLROOT) 
# LD =  -L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm
LD = -lblas -llapack -lfftw3


# works for GFORTRAN and IFORT
F90FLAGS     = -O2
DEBUGFLAGS   = 
#DEBUGFLAGS   = -Wall -Wextra -Winline -Wno-unused-parameter -Wimplicit-interface -fbounds-check -fimplicit-none -fbacktrace -g 


# will be included in sharc source code so that compiler, host, etc. can be printed at runtime
DATE  = $(shell date)
HOST  = $(shell hostname)
COMPILEDIR = $(shell pwd)
COMPILER   = $(shell which ${F90})

# where the binaries should be put
EXEDIR     = ../bin

# sources for sharc.x, data_extractor.x and diagonalizer.x
TOOLS = matrix.o \
 	definitions.o \
        string.o \
	input_list.o

SHARC  =  main.o \
	  misc.o \
	  output.o \
	  restart.o \
   	  qm_out.o \
       	  nuclear.o \
          electronic.o \
          qm.o \
          electronic_laser.o \
	  input.o 



DATA_EXTRACTOR =      definitions.o \
		      matrix.o \
  	              data_extractor.o \
  	              qm_out.o

LASER=  LASER_definitions.o \
	LASER_calc_fftw.o \
        LASER_input.o \
        LASER_main.o 


DIAGONALIZER=string.o \
             matrix.o \
             diagonalizer.o


all: sharc data_extractor diagonalizer laser

sharc: build_info $(TOOLS) $(SHARC)  
	$(F90)  $(LD) $(SHARC) $(TOOLS) -o  $@.x

data_extractor: build_info $(DATA_EXTRACTOR) 
	$(F90)  $(LD) $(DATA_EXTRACTOR) -o $@.x

diagonalizer: $(DIAGONALIZER) 
	$(F90)  $(LD) $(DIAGONALIZER) -o $@.x


laser:  $(LASER) 
	$(F90)  $(LD)  $(LASER) -o $@.x

clean:
	rm -rvf *.o
	rm -rvf *.mod
	rm -rvf build_info.inc


install: 
	cp -f *.x $(EXEDIR)	

# the following lines define the dependencies of the Fortran90 module files
main.o: output.o qm.o restart.o misc.o electronic_laser.o input.o
misc.o:	definitions.o
output.o: $(TOOLS) build_info
nuclear.o: $(TOOLS)
electronic.o: nuclear.o $(TOOLS)
electronic_laser.o: electronic.o $(TOOLS)
input.o: $(TOOLS) output.o restart.o misc.o
qm.o: $(TOOLS) electronic.o qm_out.o restart.o
qm_out.o: $(TOOLS)
restart.o: $(TOOLS)

LASER_input.o: LASER_definitions.o 
LASER_calc_fftw.o: LASER_definitions.o



# do not call this target build_info.inc, because the file should always be updated when compiling
build_info: 
	@echo "Updating \"build_info.inc\""
	@echo "character*8000, parameter :: build_date=\"$(DATE)\"" > build_info.inc
	@echo "character*8000, parameter :: build_host=\"$(HOST)\"" >> build_info.inc
	@echo "character*8000, parameter :: build_dir=\"$(COMPILEDIR)\"" >> build_info.inc
	@echo "character*8000, parameter :: build_compiler=\"$(COMPILER)\"" >> build_info.inc

%.o: %.f90 
	$(F90) $(DEBUGFLAGS) $(F90FLAGS) -c $<

