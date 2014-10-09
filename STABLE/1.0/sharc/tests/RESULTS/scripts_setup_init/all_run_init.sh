#/bin/bash

CWD=/user/mai/Documents/NewSHARC/SHARC_1.5/DISTRIBUTION/tests/INPUT/scripts_setup_init

cd $CWD/ICOND_00000//
bash run.sh
cd $CWD
echo ICOND_00000/ >> DONE
cd $CWD/ICOND_00001//
bash run.sh
cd $CWD
echo ICOND_00001/ >> DONE
