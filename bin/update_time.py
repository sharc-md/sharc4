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
        print("Usage: python update_time_in_folders.py <directory_path>")
        sys.exit(1)
    
    # Get the base directory from the command-line argument
    base_dir = [sys.argv[i] for i in range(1,len(sys.argv[:]))]
    
    for state_dir in base_dir:
        # Check if the directory exists
        if not os.path.isdir(state_dir):
            print(f"Error: {state_dir} is not a valid directory.")
            sys.exit(1)
        
        # Iterate through each subfolder in the base directory
        for subfolder in os.listdir(state_dir):
            subfolder_path = os.path.join(state_dir, subfolder)
            if os.path.isdir(subfolder_path) and subfolder.startswith("TRAJ_"):
                update_time(subfolder_path)
    
    print("Processing completed.")
