#******************************************
#
# SHARC Program Suite
#
# Copyright (c) 2014 University of Vienna
#
#******************************************
#
# Main Makefile for the sharc suite,
# handles everything that is done via the setuptools
#


all: build_ext




build_ext: lib/libsharc.so
	python sharc_setup build_ext --build-lib .

# install: lib/libsharc.so
# 	python sharc_setup install

clean:
	python sharc_setup clean --all

lib/libsharc.so: lib/libsharcnc.so
	cd ../source; make libsharc

lib/libsharcnc.so:
	cd netcdf; make lib

# PYSHARC_INCLUDE= pysharc_src/
# 
# PYSHARC_OBJECTS= $(PYSHARC_INCLUDE)/pysharc.o $(PYSHARC_INCLUDE)/pysharc_tools.o
# 
# SHARC_LIB = -Llib/ -lsharc
# 
# PY_LD = $(shell python2-config --ldflags)
# 
# pysharc: $(PYSHARC_OBJECTS)
# 	$(CC) $(CPREPROCESSOR) $(PYINCLUDE) $(CPYTHON) $(CFLAGS) $(CSHARE) $(CDEBUG) -shared -o sharc/sharc.so $(PYSHARC_OBJECTS) $(LD) $(PY_LD)  $(SHARC_LIB)
# 
# %.o: %.c
# 	$(CC) $(CPREPROCESSOR) $(PYINCLUDE) -Iinclude/ $(CPYTHON) $(CFLAGS) $(CSHARE) $(CDEBUG) -c $< -o $@


