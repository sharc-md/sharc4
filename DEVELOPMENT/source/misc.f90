module misc

integer,parameter :: n_sharcfacts=9
character*1024    :: sharcfacts(n_sharcfacts)

 contains

! ===================================================

  subroutine init_sharcfacts
    implicit none

    sharcfacts(1)='SHARC fun fact #1: If you print the source code of SHARC and fold boats out of the paper, SHARC will &
      &actually swim.'
    sharcfacts(2)='SHARC fun fact #2: If you try to run SHARC backwards, you will get a CRAHS.'
    sharcfacts(3)='SHARC fun fact #3: Seemingly, some anagrams of SHARC are frowned upon in german-speaking coutries.'
    sharcfacts(4)='SHARC fun fact #4: SHARC is a common misspelling of a flightless, cartilaginous fish belonging to the &
      &superorder selachimorpha, usually referred to as "sea dogs" until the 16th century.'
    sharcfacts(5)='SHARC fun fact #42: SHARC is not the ultimate answer to life and everything :('
    sharcfacts(6)='SHARC fun fact #5: SHARC can detect electromagnetic radiation.'
    sharcfacts(7)='SHARC fun fact #6: SHARC is not a living fossil.'
    sharcfacts(8)='SHARC fun fact #7: SHARC is a rather sophisticated random number generator.'
    sharcfacts(9)='SHARC fun fact #8: In 2014, more people were killed by lightning than by SHARC.'

  endsubroutine

! ===================================================

  subroutine write_sharcfact(u)
    implicit none
    integer, intent(in) :: u
    character*1024 :: str, str2
    real*8 :: r
    integer :: r2, length, breaks=90

    call init_sharcfacts()
    call random_number(r)
    r2=int(n_sharcfacts*r)
    str=sharcfacts(r2)
    write(u,*) '------------------------------------------------------------------------------------------'
    do
      length=len(trim(str))
      if (length>breaks) then
        str2=str(1:breaks)
        str=str(breaks+1:1024)
        write(u,*) trim(str2)
      else
        write(u,*) trim(str)
        exit
      endif
    enddo
    write(u,*) '------------------------------------------------------------------------------------------'
  endsubroutine

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