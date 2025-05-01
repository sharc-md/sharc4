cd QM
$SHARC/SHARC_TURBOMOLE.py QM.in >> QM.log 2>> QM.err
err=$?
exit $err
