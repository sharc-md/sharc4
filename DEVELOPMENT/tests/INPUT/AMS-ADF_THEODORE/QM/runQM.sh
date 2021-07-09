cd QM
$SHARC/SHARC_AMS.py QM.in >> QM.log 2>> QM.err
err=$?

rm *.xml || echo "Could not remove *.xml"
exit $err