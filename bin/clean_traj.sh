#!/bin/bash

# Check if the current directory starts with "TRAJ_"
if [[ $(basename "$PWD") != TRAJ_* ]]; then
    echo "Not in a TRAJ_* directory. Exiting without deleting anything."
    exit 1
fi

echo "In TRAJ_* directory. Proceeding with deletions..."

# Delete files with prefixes output.* and restart.*
rm -v output.* restart.*

# Delete empty indicator files
if [ -f STOP ]; 
then 
  rm -v STOP
fi
if [ -f CRASHED ]; 
then 
  rm -v CRASHED
fi
if [ -f DONT_ANALYZE ]; 
then 
  rm -v DONT_ANALYZE
fi
if [ -f RUNNING ]; 
then 
  rm -v RUNNING
fi
if [ -f DEAD ]; 
then 
  rm -v DEAD
fi

# Delete all files in restart/ that are named STEP
find restart/ -type f -name 'STEP' -exec rm -v {} +

# Delete all files in restart/ that end with ".<integer>"
find restart/ -type f -name '*.[0-9]*' -exec rm -v {} +

# Recursively delete all files in QM/ that start with QM.
find QM/ -type f -name 'QM.*' -exec rm -v {} +

echo "Deletion completed."

