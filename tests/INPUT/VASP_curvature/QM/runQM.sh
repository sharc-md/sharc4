cd QM
$SHARC/SHARC_CPA.py QM.in >> QM.log 2>>QM.err
err=$?

exit $err