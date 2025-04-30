cd QM
$SHARC/SHARC_NWCHEM.py QM.in >> QM.log 2>> QM.err
err=$?

exit $err
