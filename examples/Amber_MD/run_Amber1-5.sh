#!/bin/bash

#SBATCH -p compute
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p compute                  ## is the default partition option and can thus be omitted, alternative: debug
#SBATCH -w terbium                 ## for requesting a particular machine (optional)
#SBATCH --gres gpu:1

if [ -z "${SLURM_NTASKS_PER_NODE}" ]; then
    SLURM_NTASKS_PER_NODE=1
fi

if [ -z "${SLURM_SUBMIT_DIR}" ]; then
    SLURM_SUBMIT_DIR=$(pwd)
fi
echo ${SLURM_NTASKS_PER_NODE}
echo ${SLURM_SUBMIT_DIR}


# ----------------------------------------------------------------

cd $SLURM_SUBMIT_DIR
hostname
pwd
date

# ----------------------------------------------------------------

# funktioniert auf terbium und samarium, und weitere?
module load amber/2023


# ----------------------------------------------------------------

PMEMD_CUDA=$AMBERHOME/bin/pmemd.cuda_SPFP
PMEMD_SINGLE="$AMBERHOME/bin/pmemd"
PMEMD_MPI="mpirun -np $SLURM_NTASKS_PER_NODE $AMBERHOME/bin/pmemd.MPI"
DRIVER=$PMEMD_CUDA
RUN="true"

# ----------------------------------------------------------------

DT=0.0002                      # in ps
NSTEPS_OPTI=1000
NSTEPS_HEAT=25000
NSTEPS_EQUI=50000
NSTEPS_PROD=250000
SNAPSHOTS=100
NSTEPS_SNAPSHOTS=$(echo "$NSTEPS_PROD/$SNAPSHOTS" | bc)
NTF=2           # 1: all bonds, 2: omit X-H bonds (SHAKE)
JFASTW=0        # 4: flexible, 0: SHAKE
TEMP=300        # in K

SOLVENTS_TO_KEEP=100
SOLVENT_NAME="WAT"

# ----------------------------------------------------------------

TOP=system.prmtop

# ----------------------------------------------------------------

BASE=01_min

INP=$BASE.in
OUT=$BASE.out
RST=$BASE.rst
PREV=system.inpcrd

echo "minimize
 &cntrl
  imin=1, 
  maxcyc=$NSTEPS_OPTI, 
  ncyc=500,
  cut=8.0, 
  ntb=1, 
  ntp=0,
  ntc=$NTF,
  ntf=$NTF,
  jfastw=$JFASTW,
 /" > $INP

if [ "$RUN" = "true" ];
then
$DRIVER  -O -i $INP -o $OUT -p $TOP -c $PREV -r $RST
fi

# ----------------------------------------------------------------

BASE=02_heat

PREV=$RST
INP=$BASE.in
OUT=$BASE.out
RST=$BASE.rst
CRD=$BASE.crd


echo "heat NVT 50 ps
 &cntrl
  imin=0, 
  irest=0, 
  ntx=1,
  nstlim=$NSTEPS_HEAT, 
  dt=$DT,
  ntc=$NTF, 
  ntf=$NTF,
  jfastw=$JFASTW,
  cut=8.0,
  ntt=3, 
  gamma_ln=1.0,
  ntb=1, 
  ntp=0,
  tempi=0.0, 
  temp0=$TEMP,
  ntpr=1000, 
  ntwx=$NSTEPS_HEAT, 
  ntwr=$NSTEPS_HEAT,
  ioutfm=1,
 /" > $INP

if [ "$RUN" = "true" ];
then
$DRIVER  -O -i $INP -o $OUT -p $TOP -c $PREV -r $RST -x $CRD
fi

# ----------------------------------------------------------------

BASE=03_equi

PREV=$RST
INP=$BASE.in
OUT=$BASE.out
RST=$BASE.rst
CRD=$BASE.crd


echo "equil NPT 100 ps
 &cntrl
   imin=0, 
   irest=1, 
   ntx=5,
  nstlim=$NSTEPS_EQUI, 
  dt=$DT,
    ntc=$NTF, 
    ntf=$NTF, 
    jfastw=$JFASTW,
    cut=8.0,
  ntt=3, 
  gamma_ln=1.0,
  temp0=$TEMP,
    ntp=1,
    ntb=2, 
    barostat=1, 
    taup=1.0, 
    pres0=1.0,
  ntpr=1000, 
  ntwx=$NSTEPS_EQUI, 
  ntwr=$NSTEPS_EQUI,
  ntxo=2,
  ioutfm=1,
  iwrap=1,
 /" > $INP

if [ "$RUN" = "true" ];
then
$DRIVER  -O -i $INP -o $OUT -p $TOP -c $PREV -r $RST -x $CRD
fi

# ----------------------------------------------------------------

BASE=04_prod

PREV=$RST
INP=$BASE.in
OUT=$BASE.out
RST=$BASE.rst
CRD=$BASE.crd

echo "prod NPT 10 ns
 &cntrl
    imin=0, 
    irest=1, 
    ntx=5,
  nstlim=$NSTEPS_PROD, 
  dt=$DT,
    ntc=$NTF, 
    ntf=$NTF, 
    jfastw=$JFASTW,
    cut=8.0,
  ntt=3, 
  gamma_ln=1.0,
  temp0=$TEMP,
    ntp=1,
    ntb=2, 
    barostat=1, 
    taup=1.0, 
    pres0=1.0,
  ntpr=1000, 
  ntwx=$NSTEPS_SNAPSHOTS, 
  ntwr=-$NSTEPS_SNAPSHOTS,
  ntxo=2,
  ioutfm=1,
  iwrap=1,
 /" > $INP

if [ "$RUN" = "true" ];
then
$DRIVER  -O -i $INP -o $OUT -p $TOP -c $PREV -r $RST -x $CRD
fi

# ----------------------------------------------------------------

NEWBASE=05_cppt
INP=$NEWBASE.in
OUT=$NEWBASE.out

echo "parm $TOP
" > $INP

for i in $BASE.rst_*;
do
    ext="${i##*.}"
    echo "trajin $i
autoimage @1 origin
solvent :$SOLVENT_NAME
closest $SOLVENTS_TO_KEEP @1 noimage parmout ${TOP%.*}_$SOLVENTS_TO_KEEP.prmtop
trajout ${NEWBASE}.$ext restartnc
run
clear trajin
" >> $INP
done

echo "quit" >> $INP

cpptraj < $INP > $OUT


rm $BASE.rst_*


