#******************************************
#
#SHARC Program Suite
#
#    Copyright (c) 2025 University of Vienna
#
#    This file is part of SHARC.
#
#    SHARC is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    SHARC is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    inside the SHARC manual.  If not, see <http://www.gnu.org/licenses/>.
#
#******************************************
# Makefile for Sharc 
# Version 3.0
# 1.9.2023


# -------------------------------------- configure ---------------------------------------

# =======================================
#true, false     # also implies NETCDF
USE_PYSHARC := true

#intel, gnu
USE_COMPILER := intel

#mkl,gnu
USE_LIBS := mkl

#Static libraries
COMP_STATIC := false

#needed for PYSHARC
ANACONDA := ${CONDA_PREFIX}
# =======================================




# -------------------------------------- settings ---------------------------------------

# path
SHARC_SOURCE := $(shell pwd)
SHARCDIR := $(shell cd ..; pwd)
#MKLROOT := ${ANACONDA}/lib


# -------------------------------------- conditionals ---------------------------------------

# 
ifeq ($(USE_PYSHARC),true)
  FPREPROCESSOR := -D__PYSHARC__
  NETCDF_LIB = -L$(ANACONDA)/lib \
           -lhdf5 \
           -lhdf5_hl \
           -lmfhdf \
           -ldf \
           -ljpeg \
           -L../pysharc/lib \
           -lsharcnc \
           -lnetcdf 
else
  FPREPROCESSOR := 
  NETCDF_LIB = 
endif


ifeq ($(USE_COMPILER), )
  USE_COMPILER:=gnu
endif
ifeq ($(USE_COMPILER), gnu)
  CC :=gcc
  F90 :=gfortran 
  EXTRAFLAGS := -ffree-line-length-none 
else ifeq ($(USE_COMPILER), intel)
  CC:=icc
  F90:=ifx
  EXTRAFLAGS := -DIFORT
endif


# Note: you probably need to adjust the search path if you haven't set $(MKLROOT) 
ifeq ($(COMP_STATIC),true)
ifeq ($(USE_LIBS), mkl)
ifeq ($(USE_COMPILER), gnu)
  LD=  -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_gf_lp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl
else ifeq ($(USE_COMPILER), intel)
#  LD= -L$(MKLROOT)/lib/intel64 -lmkl_rt -lpthread -lm -lgfortran -static $(NETCDF_LIB)
  LD=  -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl
endif
else ifeq ($(USE_LIBS), gnu)
  LD= -llapack -lfftw3 -lblas -lgfortran  -static $(NETCDF_LIB)
endif
else 
ifeq ($(USE_LIBS), mkl)
  LD= -L$(MKLROOT)/lib/intel64 -lmkl_rt -lpthread -lm -lgfortran $(NETCDF_LIB)
#   LD= -L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -lgfortran $(NETCDF_LIB)
else ifeq ($(USE_LIBS), gnu)
  LD= -llapack -lfftw3 -lblas -lgfortran   $(NETCDF_LIB)
endif
endif


# -------------------------------------- settings ---------------------------------------



# include files
# FINCLUDE= -I$(SHARC_SOURCE)/
FINCLUDE= 
CINCLUDE= -I$(ANACONDA)/include

# flags
F90FLAGS = -O3 -fPIC -g  $(EXTRAFLAGS)

# debug
DEBUGFLAGS  = #-Wall -Wextra -Winline -Wno-unused-parameter -Wimplicit-interface -fbounds-check -fimplicit-none -fbacktrace -g 

# C flags
CLIB= -lgfortran
CPREPROCESSOR=
CSHARE= -fPIC
CDEBUG=
CFLAGS= -Wall -std=c99 -g -pedantic



# will be included in sharc source code so that compiler, host, etc. can be printed at runtime
DATE  = $(shell date)
HOST  = $(shell hostname)
COMPILEDIR = $(shell pwd)
COMPILER   = $(shell which ${F90})








# -------------------------------------- lists ---------------------------------------

# where the binaries should be put
EXEDIR     = ../bin

# sources for sharc.x, data_extractor.x and diagonalizer.x
TOOLS = definitions.o \
	matrix.o \
	string.o \
	input_list.o

SHARC  =  integrators.o \
          misc.o \
	  output.o \
	  restart.o \
	  qm_out.o \
	  nuclear.o \
          bsh.o \
	  electronic.o \
	  qm.o \
	  electronic_laser.o \
	  input.o \
	  decoherence_afssh.o \
          decoherence_dom.o \
          zpe.o \
          tsh_tu.o \
          army_ants.o \
          pointer_basis.o \
	  ziggurat.o \
          restrictive_potential.o

ifeq ($(USE_PYSHARC),true)
SHARC += interface.o
endif

ifeq ($(USE_PYSHARC),true)
SHARCTARGET := sharc2.x
else
SHARCTARGET := sharc1.x
endif


DATA_EXTRACTOR =  qm_out.o \
		  data_extractor.o

LASER =	definitions.o \
		LASER_calc_fftw.o \
		LASER_input.o \
		LASER_main.o 


DIAGONALIZER = string.o \
	      matrix.o \
	      diagonalizer.o


DATA_EXTRACTOR_NETCDF = data_extractor_NetCDF.o data_extractor_NetCDFmodule.o definitions_NetCDF.o qm_out.o 


DATA_CONVERTER = data_converter.o data_extractor_NetCDFmodule.o definitions_NetCDF.o qm_out.o 

DATA_CONVERTER_TO_ASCII = data_converter_to_ASCII.o data_extractor_NetCDFmodule.o definitions_NetCDF.o qm_out.o 



# -------------------------------------- rules ---------------------------------------

ifeq ($(USE_PYSHARC),true)
all: sharc data_extractor laser sharcvars libsharc data_extractor_NetCDF data_converter data_converter_to_ASCII
else
all: sharc data_extractor laser sharcvars
endif


sharc: build_info $(TOOLS) $(SHARC) main.o
	$(F90)    $(TOOLS) $(SHARC) main.o  -o  $(SHARCTARGET) $(LD)
	ln -sf $(SHARCTARGET) sharc.x


libsharc: build_info   $(TOOLS) $(SHARC)
	$(F90) -shared $(TOOLS) $(SHARC)  -o ../pysharc/lib/libsharc.so $(LD)


data_extractor: build_info $(DATA_EXTRACTOR)  $(TOOLS)
	$(F90)  $(DATA_EXTRACTOR) $(TOOLS) -o $@.x $(LD)


data_extractor_NetCDF: $(DATA_EXTRACTOR_NETCDF) $(TOOLS)
	$(F90)  $(DATA_EXTRACTOR_NETCDF) $(TOOLS) -o $@.x $(LD)


data_converter: $(DATA_CONVERTER) $(TOOLS)
	$(F90)  $(DATA_CONVERTER) $(TOOLS) -o $@.x $(LD)

data_converter_to_ASCII: $(DATA_CONVERTER_TO_ASCII) $(TOOLS)
	$(F90)  $(DATA_CONVERTER_TO_ASCII) $(TOOLS) -o $@.x $(LD)


diagonalizer: $(DIAGONALIZER) 
	$(F90)  $(DIAGONALIZER) -o $@.x $(LD)


laser:  $(LASER) 
	$(F90)   $(LASER) -o $@.x $(LD)


install: all
	cp -f *.x $(EXEDIR)
	cp -f sharcvars.sh $(EXEDIR)
	cp -f sharcvars.csh $(EXEDIR)


# write sharcvars.sh file
ifeq ($(USE_PYSHARC),true)
sharcvars:
	@echo 'export SHARC=$(SHARCDIR)/bin' > sharcvars.sh
	@echo 'export PYSHARC=$(SHARCDIR)/pysharc' >> sharcvars.sh
	@echo 'export SHARCLIB=$(SHARCDIR)/lib' >> sharcvars.sh
	@echo 'export ANACONDA=$(ANACONDA)' >> sharcvars.sh
	@echo 'export PYTHONPATH=$$SHARCLIB:$$PYSHARC:$$PYTHONPATH' >> sharcvars.sh
	@echo 'export LD_LIBRARY_PATH=$$SHARCLIB:$$ANACONDA/lib:$$LD_LIBRARY_PATH' >> sharcvars.sh
	@echo 'setenv SHARC=$(SHARCDIR)/bin' > sharcvars.csh
	@echo 'setenv PYSHARC=$(SHARCDIR)/pysharc' >> sharcvars.csh
	@echo 'setenv SHARCLIB=$(SHARCDIR)/lib' >> sharcvars.csh
	@echo 'setenv ANACONDA=$(ANACONDA)' >> sharcvars.csh
	@echo 'setenv PYTHONPATH=$$SHARCLIB:$$PYSHARC:$$PYTHONPATH' >> sharcvars.csh
	@echo 'setenv LD_LIBRARY_PATH=$$SHARCLIB:$$ANACONDA/lib:$$LD_LIBRARY_PATH' >> sharcvars.csh
else
sharcvars:
	@echo 'export SHARC=$(SHARCDIR)/bin' > sharcvars.sh
	@echo 'export SHARCLIB=$(SHARCDIR)/lib' >> sharcvars.sh
	@echo 'export PYTHONPATH=$$SHARCLIB:$$PYTHONPATH' >> sharcvars.sh
	@echo 'export LD_LIBRARY_PATH=$$SHARCLIB:$$LD_LIBRARY_PATH' >> sharcvars.sh
	@echo 'setenv SHARC=$(SHARCDIR)/bin' > sharcvars.csh
	@echo 'setenv SHARCLIB=$(SHARCDIR)/lib' >> sharcvars.csh
	@echo 'setenv PYTHONPATH=$$SHARCLIB:$$PYTHONPATH' >> sharcvars.csh
	@echo 'setenv LD_LIBRARY_PATH=$$SHARCLIB:$$LD_LIBRARY_PATH' >> sharcvars.csh
endif


# do not call this target build_info.inc
build_info: 
	@echo "Updating \"build_info.inc\""
	@echo "character(len=500), parameter :: build_date=\"$(DATE)\"" > build_info.inc
	@echo "character(len=500), parameter :: build_host=\"$(HOST)\"" >> build_info.inc
	@echo "character(len=500), parameter :: build_dir=\"$(COMPILEDIR)\"" >> build_info.inc
	@echo "character(len=500), parameter :: build_compiler=\"$(COMPILER)\"" >> build_info.inc
	@echo "character(len=500), parameter :: use_pysharc=\"$(USE_PYSHARC)\"" >> build_info.inc


# clean function
clean:
	rm -rvf *.o
	rm -rvf *.x
	rm -rvf *.mod
	rm -rvf build_info.inc
	rm -rvf *.so
	rm -rvf ../pysharc/lib/libsharc.so
	rm -rvf sharcvars.*sh

test_path:
	echo $(MAKEFILEPATH)

%.o: %.f90 
	$(F90) $(FINCLUDE) $(DEBUGFLAGS) $(F90FLAGS) -c $<

%.o: %.F90 
	$(F90) $(FINCLUDE) $(FPREPROCESSOR) $(DEBUGFLAGS) $(F90FLAGS) -c $<

%.o: %.c
	$(CC) $(CINCLUDE) $(CPREPROCESSOR) $(CFLAGS) $(CSHARE) $(CDEBUG) -c $<



# -------------------------------------- dependencies ---------------------------------------


# the following lines define the dependencies of the Fortran90 module files
ifeq ($(USE_PYSHARC),true)
main.o: output.o qm.o restart.o misc.o electronic_laser.o input.o decoherence_afssh.o decoherence_dom.o zpe.o tsh_tu.o army_ants.o pointer_basis.o interface.o
else
main.o: integrators.o output.o qm.o bsh.o restart.o misc.o electronic_laser.o input.o decoherence_afssh.o decoherence_dom.o zpe.o tsh_tu.o army_ants.o pointer_basis.o
endif

integrators.o: output.o qm.o bsh.o restart.o misc.o electronic_laser.o input.o decoherence_afssh.o decoherence_dom.o zpe.o tsh_tu.o army_ants.o pointer_basis.o
interface.o: electronic.o definitions.o definitions_NetCDF.o output.o qm.o restart.o misc.o electronic_laser.o input.o decoherence_afssh.o decoherence_dom.o zpe.o tsh_tu.o army_ants.o pointer_basis.o
misc.o: definitions.o
output.o: $(TOOLS) build_info misc.o
nuclear.o: $(TOOLS) decoherence_afssh.o decoherence_dom.o ziggurat.o misc.o
electronic.o: nuclear.o $(TOOLS) decoherence_afssh.o decoherence_dom.o
electronic_laser.o: electronic.o $(TOOLS)
input.o: $(TOOLS) output.o restart.o misc.o restrictive_potential.o
qm.o: $(TOOLS) electronic.o nuclear.o qm_out.o restart.o restrictive_potential.o
bsh.o: $(TOOLS) electronic.o decoherence_dom.o qm.o
qm_out.o: $(TOOLS)
restart.o: $(TOOLS) misc.o decoherence_afssh.o ziggurat.o
decoherence_afssh.o: $(TOOLS) definitions.o
decoherence_dom.o: $(TOOLS) definitions.o
zpe.o: $(TOOLS) definitions.o nuclear.o
tsh_tu.o: $(TOOLS) definitions.o electronic.o nuclear.o army_ants.o
army_ants.o: $(TOOLS) definitions.o electronic.o 
pointer_basis.o: $(TOOLS) definitions.o
restrictive_potential.o: $(TOOLS) definitions.o

LASER_input.o: definitions.o 
LASER_calc_fftw.o: definitions.o

data_extractor_NetCDFmodule.o: $(TOOLS) qm_out.o
data_extractor_NetCDF.o: data_extractor_NetCDFmodule.o definitions.o matrix.o definitions_NetCDF.o
data_converter.o:        data_extractor_NetCDFmodule.o definitions.o matrix.o definitions_NetCDF.o
data_converter_to_ASCII.o: data_extractor_NetCDFmodule.o definitions.o matrix.o definitions_NetCDF.o 
