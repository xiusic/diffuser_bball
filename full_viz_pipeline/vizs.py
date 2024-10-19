import os
import subprocess

### Code to get Gifs of Multiple Trajectories at once

# List of file names
file_names = []

# Base path (Where Trajectories Are)
base_path = ""

# Command template
command_template = "python full_visual_pipeline.py --path '{}'"

# Execute commands
for file_name in file_names:
    # Loop through different possessions
    for i in range(110,121):
        file_path = os.path.join(base_path, file_name, f"2016.NBA.Raw.SportVU.Game.Logs12.05.2015.POR.at.MIN_dir-{i}-guided-245K.npy")
        command = command_template.format(file_path)
        subprocess.run(command, shell=True)