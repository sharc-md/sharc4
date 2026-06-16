#!/usr/bin/env python3

# ******************************************
#
#    SHARC Program Suite
#
#    Copyright (c) 2026 University of Vienna
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


import sys
import os

def update_time(path):
    # File paths
    input_file_path = os.path.join(path,"output.lis")  # Input file
    output_file_path = os.path.join(path, "output_start_time.lis")  # Output file
    start_time_file = os.path.join(path, "start.time")  # File containing the time offset
    
    # Default offset in case start.time is not found or invalid
    default_offset_time = 0.0
    
    # Read the offset time from start.time
    try:
        with open(start_time_file, "r") as f:
            for il, line in enumerate(f):
                if il==0:
                    offset_time = float(line.strip())
                #offset_time = float(f.read().strip())
    except FileNotFoundError:
        print(f"{start_time_file} not found. Using default offset of {default_offset_time} fs.")
        offset_time = default_offset_time
    except ValueError:
        print(f"{start_time_file} contains invalid data. Using default offset of {default_offset_time} fs.")
        offset_time = default_offset_time
    
    # Process the output.lis file
    updated_lines = []
    with open(input_file_path, "r") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                # Keep header and special lines unchanged
                updated_lines.append(line)
            else:
                # Update the time column while keeping exact formatting
                # Extract sections of the line using fixed column widths
                step = line[0:10]  # Step column
                time = line[10:25]  # Time column
                rest = line[25:]  # Rest of the columns
    
                try:
                    # Update the time column value
                    updated_time = float(time.strip()) + offset_time
                    updated_time_str = f"{updated_time:15.5f}"  # Preserve 12-character width, 5 decimals
                    updated_lines.append(step + updated_time_str + rest)
                except ValueError:
                    # If the time cannot be parsed, keep the line as is
                    updated_lines.append(line)
    
    # Write the updated content to a new file
    with open(output_file_path, "w") as f:
        f.writelines(updated_lines)
    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python shift_output_lis.py <directory_path>")
        sys.exit(1)
    
    update_time(os.path.abspath(sys.argv[1]))
   
