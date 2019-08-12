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

install: lib/libsharc.so lib/libsharcnc.so build_ext
	cp lib/*.so ../lib
	cp bin/*.py ../bin

clean:
	python sharc_setup clean --all
	rm lib/libsharc.so lib/libsharcnc.so
	rm sharc/sharc.so

lib/libsharc.so: lib/libsharcnc.so
	cd ../source; make libsharc

lib/libsharcnc.so:
	cd netcdf; make lib



