#!/bin/bash

# ******************************************
#
#    SHARC Program Suite
#
#    Copyright (c) 2025 University of Vienna
#
#    This file is part of SHARC.
#
#    SHARC is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    SHARC is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    inside the SHARC manual.  If not, see <http://www.gnu.org/licenses/>.
#
# ******************************************

# Check if the current directory starts with "TRAJ_"
if [[ $(basename "$PWD") != TRAJ_* ]]; then
    echo "Not in a TRAJ_* directory. Exiting without deleting anything."
    exit 1
fi

echo "In TRAJ_* directory. Proceeding with deletions..."

# Delete files with prefixes output.* and restart.*
rm -v output.* restart.*

# Delete output_data folder
rm -vr output_data/

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

# Recursively delete all files in restart/ that are named STEP
find restart/ -type f -name 'STEP' -exec rm -v {} +

# Recursively delete all files in restart/ that end with ".<integer>"
find restart/ -type f -name '*.[0-9]*' -exec rm -v {} +

# Recursively delete all files in QM/ that start with QM.
find QM/ -type f -name 'QM.*' -exec rm -v {} +

echo "Deletion completed."

