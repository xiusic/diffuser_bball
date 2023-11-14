import os
import sys
import torch
import numpy as np
# from diffuser.utils import LimitsNormalizer
from diffuser.datasets import BBwdSequenceDataset

# Set the path to the directory containing your dataset
dataset_path = "/local2/yao/diffuser/data/clean_trajectories_2"
reward_path = "/local2/yao/diffuser/data/2_final_json_rewards"

# Create an instance of the BBwdSequenceDataset
dataset = BBwdSequenceDataset(filepath=dataset_path)

# Access a specific observation from the dataset
index = 0  # Replace with the desired index
observation_at_index = dataset.observations[index, 0]

# Print the shape of the accessed observation
print("Shape of the accessed observation:", observation_at_index.shape)
print(observation_at_index)