import os
import subprocess

# List of file names
file_names = [
    "guided_samplesact25_25_0.1",
    "guided_samplesact25_50_0.1",
    "guided_samplesact25lre3_25_0.1",
    "guided_samplesact25lre3_50_0.1",
    "guided_samplesact25lre5_25_0.1",
    "guided_samplesact25lre5_50_0.1",
    "guided_samplesact30_25_0.1",
    "guided_samplesact30_50_0.1",
    "guided_samplesact30lre3_25_0.1",
    "guided_samplesact30lre3_50_0.1",
    "guided_samplesact30lre5_25_0.1",
    "guided_samplesact30lre5_50_0.1"
]

# Base path
base_path = "/local2/dmreynos/diffuser_bball/logs/"

# Command template
command_template = "python full_visual_pipeline.py --path {}"

# Execute commands
for file_name in file_names:
    file_path = os.path.join(base_path, file_name, "2016.NBA.Raw.SportVU.Game.Logs12.05.2015.POR.at.MIN_dir-15-guided-245K.npy")
    command = command_template.format(file_path)
    subprocess.run(command, shell=True)