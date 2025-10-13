cd QM
$SHARC/SHARC_TEQUILA.py QM.in >> QM.log 2>> QM.err
err=$?

exit $err
