import shutil
import os

parent_directory = '/local2/dmreynos/diffuser_bball/logs'

directories_to_delete = [
    'guided_sampleshue_loose_final100_0.1',
    'guided_sampleshue_original25_0.1',
    'guided_sampleshue_original50_0.1',
    'guided_sampleshue_original100_0.1',
    'guided_sampleshue_stagnent25_0.1',
    'guided_sampleshue_stagnent50_0.1',
    'guided_sampleshue_stagnent75_0.1',
    'guided_sampleshue_stagnent100_0.1',
    'guided_sampleshue0_0.1',
    'guided_sampleshue1_0.1',
    'guided_sampleshue25_0.1',
    'guided_sampleshue50_0.1',
    'guided_sampleshue75_0.1',
    'guided_sampleshue100_0.1',
    'guided_sampleshuecond_0.1',
    'guided_sampleshueloose25_0.1',
    'guided_sampleshueloose50_0.1',
    'guided_sampleshueloose75_0.1',
    'guided_sampleshueloose100_0.1',
    'guided_sampleshuelooser25_0.1',
    'guided_sampleshuelooser50_0.1',
    'guided_sampleshuelooser75_0.1',
    'guided_sampleshuelooser100_0.1',
    'guided_sampleshuelooserer25_0.1',
    'guided_sampleshuelooserer50_0.1',
    'guided_sampleshuelooserer75_0.1',
    'guided_sampleshuelooserer100_0.1',
    'guided_sampleshuestopped25_0.1',
    'guided_sampleshuestopped50_0.1'
]

for directory in directories_to_delete:
    directory_path = os.path.join(parent_directory, directory)
    try:
        shutil.rmtree(directory_path)
        print(f"Deleted directory: {directory_path}")
    except Exception as e:
        print(f"Failed to delete directory: {directory_path}")
        print(e)