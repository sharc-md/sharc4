program diagonalizer
! Program reading a symmetric or hermitian matric from stdin
! and writes the eigenvalue matrix and eigenvector matrix to stdout
!
! Input looks like:
! r 2 2
! title
! 5.5 4.1
! 4.1 6.6
!
! Output looks like:
! Eigenvalue matrix
!  0.1913274241625E+01  0.0000000000000E+00
!  0.0000000000000E+00  0.1018672575837E+02
! Eigenvector matrix
! -0.7526471262340E+00  0.6584241060074E+00
!  0.6584241060074E+00  0.7526471262340E+00
!
! Input comments:
! - the first line contains three elements
!   * first a string which is either "r" or "c"
!     for real symmetric or hermitian matrices, respectively
!   * two integers giving the matrix dimensions (matrix must be square)
! - the second line is a comment and has no effect
! - starting in the third line, the matrix elements are read
!
! Comments:
! - uses real*8 and complex*16 variables
!
use matrix
use string
implicit none

integer :: i,j,n, typeflag=-1
character*8000 :: str

complex*16,allocatable :: Hc(:,:), Uc(:,:)
real*8,allocatable :: Hr(:,:),Ur(:,:)

read(5,*) str,i,j
call lowercase(str)
if (i/=j) then
  write(0,*) 'Only square matrices allowed!'
  stop 1
endif
if (trim(str)=='r') then
  typeflag=0
elseif (trim(str)=='c') then
  typeflag=1
else
  write(0,*) 'Type must be "r" or "c"!'
  stop 1
endif
n=i
call allocate_lapack(n)

allocate( Hr(n,n),Ur(n,n) )
allocate( Hc(n,n),Uc(n,n) )

select case (typeflag) 
  case (0)
    call matread(n,Hr,5,str)
    if (.not.ishermitian(n,Hr)) then
      write(0,*) 'Only symmetric matrices allowed!'
      stop 1
    endif
    call diagonalize(n,Hr,Ur)
    call matwrite(n,Hr,6,'Eigenvalue matrix','E20.13')
    call matwrite(n,Ur,6,'Eigenvector matrix','E20.13')
  case (1)
    call matread(n,Hc,5,str)
    if (.not.ishermitian(n,Hc)) then
      write(0,*) 'Only hermitian matrices allowed!'
      stop 1
    endif
    call diagonalize(n,Hc,Uc)
    call matwrite(n,Hc,6,'Eigenvalue matrix','E20.13')
    call matwrite(n,Uc,6,'Eigenvector matrix','E20.13')
endselect

deallocate( Hr, Ur, Hc, Uc)

endprogram