cd QM
$SHARC/SHARC_ANALYTICAL.py QM.in >> QM.log 2>> QM.err
err=$?
exit $err
