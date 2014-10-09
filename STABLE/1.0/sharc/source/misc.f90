module misc
 contains

! ===================================================

  function djb_hash(str) result(res)
    character(len=*),intent(in) :: str
    integer :: hash = 16661     ! some prime
    integer :: i = 0
    integer*8 :: res
 
    do i=1,len(str)
        hash = (ishft(hash,5) + hash) + ichar(str(i:i))
    enddo

    res = abs(hash)

  endfunction DJB_hash

! ===================================================

  subroutine set_time(traj)
    use definitions
    implicit none
    type(trajectory_type) :: traj
    integer :: time

    traj%time_step=time()-traj%time_last
    traj%time_last=time()

  endsubroutine

! ===================================================

  subroutine init_random_seed(rngseed)
    implicit none
    integer,intent(in) :: rngseed
    integer :: n,i
    integer,allocatable :: seed(:)
    real*8 :: r

    call random_seed(size=n)
    allocate(seed(n))
    do i=1,n
      seed(i)=rngseed+37*i+17*i**2
    enddo
    call random_seed(put=seed)

    ! the elements of seed should be uncorrelated, thus we
    ! calculate some with the random number generator and reseed
    do i=1,n
    call random_number(r)
      seed(i)=int(65536*(r-0.5d0))
    enddo
    call random_seed(put=seed)
!     deallocate(seed)

  endsubroutine


endmodule