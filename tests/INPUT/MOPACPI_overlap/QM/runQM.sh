cd QM
$SHARC/SHARC_MOPACPI.py QM.in >> QM.log 2>> QM.err
err=$?

exit $err
