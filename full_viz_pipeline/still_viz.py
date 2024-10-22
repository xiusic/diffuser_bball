import os
import subprocess

### File to get SnapShots of Multiple trajectories at once

# List of file names
file_names = [
    "guided_samplesact_(2_3)_50100_0.1"
    ]

# Base path (Where Trajectories Are)
base_path = ""

# Command template
command_template = "python NBA-Player-Movements/shooter_png_dir/visual_2d.py '{}' 0 522 600 4"

# Execute commands
for file_name in file_names:
    # Loop through different possessions
    for i in range(12,121):
        print(i)
        file_path = os.path.join(file_name, f"2016.NBA.Raw.SportVU.Game.Logs12.05.2015.POR.at.MIN_dir-{i}-guided-245K")
        command = command_template.format(file_path)
        subprocess.run(command, shell=True)
        break