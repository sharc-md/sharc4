
#include <stdlib.h>
#include <stdio.h>
#include "nc_basic.h"

inline
void
prt_error(int istatus)
{
    if (istatus != NC_NOERR) {
             fprintf(stderr, "%s\n", nc_strerror(istatus)); 
             exit(-1);
    }

};

inline
int
create_ncfile(const char *str, int mode)
{
    int iretval;
    int ncid = 0;
    if ((iretval = nc_create(str, mode, &ncid)))
        prt_error(iretval);
    return ncid;
};

inline
int
open_ncfile(const char *str, int mode)
{
    int iretval;
    int ncid = 0;
    if ((iretval = nc_open(str, mode, &ncid)))
        prt_error(iretval);
    return ncid;
};

inline
void 
close_ncfile_(const int* ncid)
{
    int iretval;
    if ((iretval = nc_close(*ncid)))
        prt_error(iretval);
};


inline
int
define_dimension(const int ncid, const char *name,const size_t N)
{

    int iretval;
    int id = 0;
    if ((iretval = nc_def_dim(ncid, name, N, &id)))
        prt_error(iretval);
    return id;
};
