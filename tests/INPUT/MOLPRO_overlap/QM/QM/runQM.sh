cd QM
$SHARC/SHARC_MOLPRO.py QM.in >> QM.log 2>> QM.err
err=$?

exit $err
