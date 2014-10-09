# Makefile for Sharc 
# Version 1.0
# Oct-09-2014
# (c) AG Leti
#
F90 = ifort
LD = -lblas -llapack -lfftw3

# this is a comment for testing purposes.

# works for GFORTRAN and IFORT
F90FLAGS     = -O2
DEBUGFLAGS   = 


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


main.o: output.o qm.o restart.o misc.o electronic_laser.o input.o
misc.o:	definitions.o
output.o: $(TOOLS) build_info.inc
nuclear.o: $(TOOLS)
electronic.o: nuclear.o $(TOOLS)
electronic_laser.o: electronic.o $(TOOLS)
input.o: $(TOOLS) output.o restart.o misc.o
qm.o: $(TOOLS) electronic.o qm_out.o restart.o
qm_out.o: $(TOOLS)
restart.o: $(TOOLS)




DATA_EXTRACTOR =      definitions.o \
		      matrix.o \
  	              data_extractor.o

LASER=  LASER_definitions.o \
	LASER_calc_fftw.o \
        LASER_input.o \
        LASER_main.o 

LASER_input.o: LASER_definitions.o 
LASER_calc_fftw.o: LASER_definitions.o

DIAGONALIZER=string.o \
             matrix.o \
             diagonalizer.o


all: sharc data_extractor diagonalizer laser

sharc: build_info.inc $(TOOLS) $(SHARC)  
	$(F90)  $(LD) $(SHARC) $(TOOLS) -o  $@.x

data_extractor: $(DATA_EXTRACTOR) 
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



# build_info.mod is never created, so the following is always executed
build_info.inc: 
	@echo "Updating \"build_info.inc\""
	@echo "character*8000, parameter :: build_date=\"$(DATE)\"" > build_info.inc
	@echo "character*8000, parameter :: build_host=\"$(HOST)\"" >> build_info.inc
	@echo "character*8000, parameter :: build_dir=\"$(COMPILEDIR)\"" >> build_info.inc
	@echo "character*8000, parameter :: build_compiler=\"$(COMPILER)\"" >> build_info.inc

%.o: %.f90 
	$(F90) $(DEBUGFLAGS) $(F90FLAGS) -c $<

