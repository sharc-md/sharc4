

# -------------------------------------- configure ---------------------------------------

# =======================================
ANACONDA := /user/maximilian/anaconda2/
# =======================================


INC= -I.  -I$(ANACONDA)/include
OPT= -O3 -Wall
SHARED= -fPIC

CC=$(ANACONDA)/bin/gcc


OBJ_LIB=nc_basic.o nc_sharc.o
library=libsharcnc.so




all: lib

lib: $(OBJ_LIB)
	ar -cvq ../lib/$(library) $(OBJ_LIB)

clean:
	rm ../lib/$(library)

%.o: %.c
	$(CC) $(SHARED) $(INC) $(OPT) -c $<

