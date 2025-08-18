cd QM
$SHARC/SHARC_LEGACY.py QM.in >> QM.log 2>> QM.err
err=$?

exit $err
