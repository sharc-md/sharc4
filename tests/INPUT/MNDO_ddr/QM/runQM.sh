cd QM
$SHARC/SHARC_MNDO.py QM.in >> QM.log 2>> QM.err
err=$?

exit $err
