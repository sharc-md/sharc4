# kate: replace-tabs FALSE; indent-width 8;
F90 = ifort
LD = -lblas -llapack -lfftw3

# works for GFORTRAN and IFORT
OPTFLAGS     = -O2

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


electronic.o: nuclear.o $(TOOLS)
electronic_laser.o: electronic.o $(TOOLS)
input.o: $(TOOLS) output.o restart.o misc.o
misc.o:	definitions.o
nuclear.o: $(TOOLS)
output.o: $(TOOLS)
qm.o: $(TOOLS) electronic.o
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

sharc: build_info.mod $(TOOLS) $(SHARC)  
	$(F90) $(OPTFLAGS) $(LD) $(SHARC) $(TOOLS) -o  $@.x

data_extractor: $(DATA_EXTRACTOR) 
	$(F90) $(LD) $(OPTFLAGS) $(DATA_EXTRACTOR) -o $@.x

diagonalizer: $(DIAGONALIZER) 
	$(F90) $(LD) $(OPTFLAGS) $(DIAGONALIZER) -o $@.x


laser:  $(LASER) 
	$(F90) $(LD) $(OPTFLAGS) $(LASER) -o $@.x

clean:
	rm -rvf *.o
	rm -rvf *.mod
	rm -rvf build_info.inc


install: 
	cp -f *.x $(EXEDIR)	



# build_info.mod is never created, so the following is always executed
build_info.mod: 
	@echo "Updating \"build_info.inc\""
	@echo "character*8000, parameter :: build_date=\"$(DATE)\"" > build_info.inc
	@echo "character*8000, parameter :: build_host=\"$(HOST)\"" >> build_info.inc
	@echo "character*8000, parameter :: build_dir=\"$(COMPILEDIR)\"" >> build_info.inc
	@echo "character*8000, parameter :: build_compiler=\"$(COMPILER)\"" >> build_info.inc

%.o: %.f90 
	$(F90) $(F90FLAGS) -c $<

