program data_extractor
use matrix
use definitions, only: au2a, au2fs, au2u, au2rcm, au2eV, au2debye
use qm_out
implicit none

integer, parameter :: u_dat=11
integer, parameter :: u_ener=21
integer, parameter :: u_dm=22
integer, parameter :: u_spin=23
integer, parameter :: u_coefd=24
integer, parameter :: u_coefm=25
integer, parameter :: u_prob=26
integer, parameter :: u_expec=27
integer, parameter :: u_coefdiab=28
integer, parameter :: u_ref=31
integer, parameter :: u_info=42


! constant
integer :: nstates, narg, maxmult, natom
integer, allocatable :: nstates_s(:)
real*8 :: dtstep, ezero
integer :: calc_overlap, laser, nsteps, nsubsteps

! per step
integer :: step
complex*16, allocatable :: H_MCH_ss(:,:),U_ss(:,:),DM_ssd(:,:,:), Prop_ss(:,:)
complex*16, allocatable :: coeff_diag_s(:),overlaps_ss(:,:), ref_ovl_ss(:,:)
real*8, allocatable :: geom_ad(:,:), veloc_ad(:,:)
real*8 , allocatable :: hopprob_s(:)
real*8 :: Ekin, Epot, randnum
integer :: state_diag, state_MCH, runtime

complex*16, allocatable :: H_diag_ss(:,:), A_ss(:,:), coeff_MCH_s(:), laser_td(:,:), coeff_diab_s(:)
real*8,allocatable :: expec_s(:), expec_dm(:), spin0_s(:)
real*8 :: sumc

! helper
character*8000 :: filename, string
character*21 :: string2
integer :: i, io, idir,istate,jstate,imult,ims
logical :: exists

! build_info.inc is written by the Makefile and contains the 
! date and host, when/where SHARC was built
include 'build_info.inc'

! first determine the filename of the dat file and open it
narg=iargc()
if (narg==0) then
  stop 'Usage: ./data_extractor <data-file>'
endif
call getarg(1,filename)
open(unit=u_dat, file=filename, status='old', action='read', iostat=io)
if (io/=0) then
  write(*,*) 'File ',trim(filename),' not found'
  stop
endif

string=trim(filename)//'.ext'
open(u_info,file=string,status='replace',action='write')
write(u_info,*) 'BUILD INFORMATION:'
write(u_info,*) 'Build date: ',trim(build_date)
write(u_info,*) 'Build host: ',trim(build_host)
write(u_info,*) 'Build directory: ',trim(build_dir)
write(u_info,*) 'Compiler: ',trim(build_compiler)
close(u_info)

! determine number of states per mult and total number of states
read(u_dat,*) maxmult
allocate( nstates_s(maxmult) )
read(u_dat,*) (nstates_s(i),i=1,maxmult)
nstates=0
do i=1,maxmult
  nstates=nstates+i*nstates_s(i)
enddo
write(*,*) 'This makes nstates=',nstates
read(u_dat,*) natom

! allocate everything
allocate( H_MCH_ss(nstates,nstates), H_diag_ss(nstates,nstates) )
allocate( U_ss(nstates,nstates) )
allocate( Prop_ss(nstates,nstates) )
allocate( overlaps_ss(nstates,nstates), ref_ovl_ss(nstates,nstates) )
allocate( DM_ssd(nstates,nstates,3) )
allocate( coeff_diag_s(nstates), coeff_MCH_s(nstates), coeff_diab_s(nstates) )
allocate( hopprob_s(nstates) )
allocate( A_ss(nstates,nstates) )
allocate( expec_s(nstates),expec_dm(nstates) )
allocate( spin0_s(nstates) )
allocate( geom_ad(natom,3), veloc_ad(natom,3) )
call allocate_lapack(nstates)
overlaps_ss=dcmplx(0.d0,0.d0)

! obtain the timestep, ezero and calc_overlap 
read(u_dat,*) dtstep
dtstep=dtstep*au2fs
read(u_dat,*) ezero
read(u_dat,*) calc_overlap
read(u_dat,*) laser
read(u_dat,*) nsteps
read(u_dat,*) nsubsteps
if (laser==2) then
  allocate( laser_td(nsteps*nsubsteps+1,3) )
  call vec3read(nsteps*nsubsteps+1,laser_td,u_dat,string)
endif

! create output directory "output_data"
inquire(file="output_data", exist=exists)
if (.not.exists) then
  write(*,'(A)') 'Creating directory "output_data"...'
  call system('mkdir output_data')
else
  write(*,'(A)') 'Writing to directory "output_data"...'
endif

! open output files
open(unit=u_ener, file='output_data/energy.out', status='replace', action='write')
open(unit=u_dm, file='output_data/fosc.out', status='replace', action='write')
open(unit=u_spin, file='output_data/spin.out', status='replace', action='write')
open(unit=u_coefd, file='output_data/coeff_diag.out', status='replace', action='write')
open(unit=u_coefm, file='output_data/coeff_MCH.out', status='replace', action='write')
open(unit=u_prob, file='output_data/prob.out', status='replace', action='write')
open(unit=u_expec, file='output_data/expec.out', status='replace', action='write')
open(unit=u_coefdiab, file='output_data/coeff_diab.out', status='replace', action='write')




! write output file headers
write(u_ener,'(A1,1X,1000(I20,1X))') '#',(i,i=1,nstates+4)
write(u_ener,'(A1,1X,5(A20,1X))') '#','Time |','Ekin |','Epot |','Etot |','=== Energy ===>'
write(u_ener,'(A1,1X,5(A20,1X))') '#','[fs] |','[eV] |','[eV] |','[eV] |','[eV] |'

write(u_dm,'(A1,1X,1000(I20,1X))') '#',(i,i=1,nstates+2)
write(u_dm,'(A1,1X,3(A20,1X))') '#','Time |','f_osc (state) |','=== f_osc ===>'
write(u_dm,'(A1,1X,3(A20,1X))') '#','[fs] |','[] |','[] |'

write(u_spin,'(A1,1X,1000(I20,1X))') '#',(i,i=1,nstates+2)
write(u_spin,'(A1,1X,3(A20,1X))') '#','Time |','Spin_s |','=== Spins ===>'
write(u_spin,'(A1,1X,3(A20,1X))') '#','[fs] |','[] |','[] |'

write(u_coefd,'(A1,1X,1000(I20,1X))') '#',(i,i=1,2*nstates+2)
write(u_coefd,'(A1,1X,3(A20,1X))') '#','Time |','Sum c**2 |','=== coeff_diag ===>'
write(u_coefd,'(A1,1X,3(A20,1X))') '#','[fs] |','[] |','[] |'

write(u_coefm,'(A1,1X,1000(I20,1X))') '#',(i,i=1,2*nstates+2)
write(u_coefm,'(A1,1X,3(A20,1X))') '#','Time |','Sum c**2 |','=== coeff_MCH ===>'
write(u_coefm,'(A1,1X,3(A20,1X))') '#','[fs] |','[] |','[] |'

write(u_coefdiab,'(A1,1X,1000(I20,1X))') '#',(i,i=1,2*nstates+2)
write(u_coefdiab,'(A1,1X,3(A20,1X))') '#','Time |','Sum c**2 |','=== coeff_diab ===>'
write(u_coefdiab,'(A1,1X,3(A20,1X))') '#','[fs] |','[] |','[] |'

write(u_prob,'(A1,1X,1000(I20,1X))') '#',(i,i=1,nstates+2)
write(u_prob,'(A1,1X,3(A20,1X))') '#','Time |','Random Number |','=== cumu Prob ===>'
write(u_prob,'(A1,1X,3(A20,1X))') '#','[fs] |','[] |','[] |'

write(u_expec,'(A1,1X,1000(I20,1X))') '#',(i,i=1,3*nstates+4)
write(string, '(A1,1X,4(A20,1X))') '#','Time |','Ekin |','Epot |','Etot |'
do i=1,nstates
  write(string2,'(1X,A8,I10,A2)') 'Energy ',i,' |'
  string=trim(string)//string2
enddo
!write(string2,'(X,A20)') 'Spin (occ) |'
!string=trim(string)//string2
do i=1,nstates
  write(string2,'(1X,A5,I13,A2)') 'Spin ',i,' |'
  string=trim(string)//string2
enddo
!write(string2,'(X,A20)') 'f_osc (occ) |'
!string=trim(string)//string2
do i=1,nstates
  write(string2,'(1X,A6,I12,A2)') 'f_osc ',i,' |'
  string=trim(string)//string2
enddo
write(u_expec,'(A)') trim(string)
write(string, '(A1,1X,4(A20,1X))') '#','[fs] |','[eV] |','[eV] |','[eV] |'
do i=1,nstates
  write(string2,'(1X,A20)') '[eV] |'
  string=trim(string)//string2
enddo
do i=1,nstates
  write(string2,'(1X,A20)') '[] |'
  string=trim(string)//string2
enddo
do i=1,nstates
  write(string2,'(1X,A20)') '[] |'
  string=trim(string)//string2
enddo
write(u_expec,'(A)') trim(string)

! spin values in MCH basis

spin0_s=0.d0
i=0
do imult=1,maxmult
  do ims=1,imult
    do istate=1,nstates_s(imult)
      i=i+1
      spin0_s(i)=real(imult-1)
    enddo
  enddo
enddo

! reference overlap
ref_ovl_ss=dcmplx(0.d0,0.d0)
do i=1,nstates
  ref_ovl_ss(i,i)=dcmplx(1.d0,0.d0)
enddo

filename='Reference/QM.out'
inquire(file=filename,exist=exists)
if (exists) then
  call open_qmout(u_ref,filename)
  call get_overlap(nstates,ref_ovl_ss)
  call close_qmout
  call lowdin(nstates,ref_ovl_ss)
else
  write(6,*) 'Reference overlap not available!'
endif


! main loop
do
  ! read everything
  read(u_dat,*,iostat=io) string
  if (io/=0) exit
  read(u_dat,*) step
  call matread(nstates,H_MCH_ss,u_dat,string)
  call matread(nstates,U_ss,u_dat,string)
  do idir=1,3
    call matread(nstates,DM_ssd(:,:,idir),u_dat,string)
  enddo
  if (calc_overlap==1) then
    call matread(nstates,overlaps_ss,u_dat,string)
  endif
  call vecread(nstates,coeff_diag_s,u_dat,string)
  call vecread(nstates,hopprob_s,u_dat,string)
  read(u_dat,*)
  read(u_dat,*) Ekin
  read(u_dat,*)
  read(u_dat,*) state_diag,state_MCH
  read(u_dat,*)
  read(u_dat,*) randnum
  read(u_dat,*)
  read(u_dat,*) runtime
  call vec3read(natom,geom_ad,u_dat,string)
  call vec3read(natom,veloc_ad,u_dat,string)
  call matread(nstates,Prop_ss,u_dat,string)

  ! calculate basics
  H_diag_ss=H_MCH_ss
  if (laser==2) then
    do idir=1,3
      H_diag_ss=H_diag_ss - DM_ssd(:,:,idir)*real(laser_td(step*nsubsteps+1,idir))
    enddo
  endif
  call transform(nstates,H_diag_ss,U_ss,'utau')
!   call matwrite(nstates,H_diag_ss,6,'','F12.9')
  call matvecmultiply(nstates,U_ss,coeff_diag_s,coeff_MCH_s,'n')
  Epot=real(H_diag_ss(state_diag,state_diag))

  ! calculate diabatic coefficients
  if (step>0) then
    call matmultiply(nstates,ref_ovl_ss,overlaps_ss,A_ss,'nn')
    ref_ovl_ss=A_ss
  endif
  call matvecmultiply(nstates,ref_ovl_ss,coeff_MCH_s,coeff_diab_s,'n')

  ! energy.out
  write(u_ener,'(2X,1000(E20.13,1X))') &
  &step*dtstep, Ekin*au2eV, Epot*au2eV, (Epot+Ekin)*au2eV,&
  (real(H_diag_ss(istate,istate)*au2eV),istate=1,nstates)

  ! fosc.out
  expec_dm=0.d0
  do idir=1,3
    A_ss=DM_ssd(:,:,idir)
    call transform(nstates,A_ss,U_ss,'utau')
    expec_dm=expec_dm+real(A_ss(:,1)*A_ss(1,:))
  enddo
  expec_dm=expec_dm*2./3.
  do i=1,nstates
    expec_dm(i)=expec_dm(i)*real(H_diag_ss(i,i)-H_diag_ss(1,1))
  enddo
  write(u_dm,'(2X,1000(E20.13,1X))') &
  &step*dtstep, expec_dm(state_diag),&
  (expec_dm(istate),istate=1,nstates)

  ! spin.out
  expec_s=0.d0
  do istate=1,nstates
    do jstate=1,nstates
      expec_s(istate)=expec_s(istate) + spin0_s(jstate) * real(U_ss(jstate,istate) * conjg(U_ss(jstate,istate)))
    enddo
  enddo
  write(u_spin,'(2X,1000(E20.13,1X))') &
  &step*dtstep, expec_s(state_diag),&
  (expec_s(istate),istate=1,nstates)

  ! coeff_diag.out
  sumc=0.d0
  do istate=1,nstates
    sumc=sumc + real(conjg(coeff_diag_s(istate))*coeff_diag_s(istate))
  enddo
  write(u_coefd,'(2X,1000(E20.13,1X))') &
  &step*dtstep, sumc,&
  (coeff_diag_s(istate),istate=1,nstates)

  ! coeff_MCH.out
  sumc=0.d0
  do istate=1,nstates
    sumc=sumc + real(conjg(coeff_MCH_s(istate))*coeff_MCH_s(istate))
  enddo
  write(u_coefm,'(2X,1000(E20.13,1X))') &
  &step*dtstep, sumc,&
  (coeff_MCH_s(istate),istate=1,nstates)

  ! coeff_diab.out
  sumc=0.d0
  do istate=1,nstates
    sumc=sumc + dconjg(coeff_diab_s(istate))*coeff_diab_s(istate)
  enddo
  write(u_coefdiab,'(2X,1000(E20.13,X))') &
  &step*dtstep, sumc,&
  (coeff_diab_s(istate),istate=1,nstates)

  ! prob.out
  do istate=2,nstates
    hopprob_s(istate)=hopprob_s(istate)+hopprob_s(istate-1)
  enddo
  write(u_prob,'(2X,1000(E20.13,1X))') &
  &step*dtstep, randnum,&
  (hopprob_s(istate),istate=1,nstates)

  ! expec.out
  write(u_expec,'(2X,1000(E20.13,1X))') &
  &step*dtstep, Ekin*au2eV, Epot*au2eV, (Epot+Ekin)*au2eV,&
  &(real(H_diag_ss(istate,istate)*au2eV),istate=1,nstates),&
  &(expec_s(istate),istate=1,nstates),&
  &(expec_dm(istate),istate=1,nstates)






  ! write progress to screen
  write(*,'(A,A,F9.2,A)',advance='no') achar(13), 't=',step*dtstep,' fs'

enddo

write(*,*)










endprogram