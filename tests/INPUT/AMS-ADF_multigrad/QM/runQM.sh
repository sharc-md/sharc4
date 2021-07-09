cd QM
$SHARC/SHARC_AMS.py QM.in >> QM.log 2>> QM.err
err=$?

# rm *.xml
exit $err