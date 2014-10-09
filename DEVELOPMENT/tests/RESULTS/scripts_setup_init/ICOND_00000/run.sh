#/bin/bash

#$-N init_00000

PRIMARY_DIR=/user/mai/Documents/NewSHARC/SHARC_1.5/DISTRIBUTION/tests/INPUT/scripts_setup_init/ICOND_00000//

cd $PRIMARY_DIR

$SHARC/SHARC_Analytical.py QM.in >> QM.log 2>> QM.err
