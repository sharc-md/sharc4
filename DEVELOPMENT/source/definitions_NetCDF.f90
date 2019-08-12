module definitions_NetCDF
    implicit none
    save
    public

type Tsharc_ncoutput
    sequence
    integer :: id
    integer :: H_MCH_id
    integer :: U_id
    integer :: DM_id
    integer :: overlaps_id
    integer :: coeff_diag_id
    integer :: e_id
    integer :: hopprob_id
    integer :: crd_id
    integer :: veloc_id
    integer :: randnum_id
    integer :: state_diag_id
    integer :: state_MCH_id
    integer :: time_step_id
end type

contains

end module definitions_NetCDF

