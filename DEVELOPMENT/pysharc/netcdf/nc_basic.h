#ifndef NETCDEF_ERROR_H_
#define NETCDEF_ERROR_H_

#include <netcdf.h>

// check_nccall macro
#define check_nccall(iret, ncall) \
    if (( iret = ncall)) \
        prt_error(iret);


#ifdef __cplusplus
extern"C" {
#endif

void prt_error(int istatus);

int open_ncfile(const char *str, int imode);
int create_ncfile(const char *str, int imode);
void close_ncfile_(const int* ncid);
int define_dimension(const int ncid, const char *name, const size_t N);


#ifdef __cplusplus
}
#endif

// end code
#endif
