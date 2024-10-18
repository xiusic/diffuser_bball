import os
import sys
import random
import torch
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
import diffuser.sampling as sampling
import diffuser.utils as utils
from diffuser.sampling.policies import Trajectories


#-----------------------------------------------------------------------------#
#----------------------------- Utility Functions ----------------------------#
#-----------------------------------------------------------------------------#

def set_seed(seed):
    """
    Set the random seed for reproducibility.
    
    Args:
        seed (int): The seed value to ensure consistency.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def normalize(x, dataset, index):
    """
    Normalize the input data between [-1, 1].

    Args:
        x (numpy.ndarray): The data to normalize.
        dataset: The dataset object containing normalizer.
        index (int): The index for extracting min and max values for normalization.

    Returns:
        numpy.ndarray: Normalized data.
    """
    mins = dataset.normalizer.mins[index]
    maxs = dataset.normalizer.maxs[index]
    nonzero_i = np.abs(maxs - mins) > 0
    x[nonzero_i] = (x[nonzero_i] - mins[nonzero_i]) / (maxs[nonzero_i] - mins[nonzero_i])
    x = 2 * x - 1
    return x


def unnormalize(x, dataset, index, eps=1e-4):
    """
    Unnormalize data back to original scale.

    Args:
        x (numpy.ndarray): The normalized data.
        dataset: The dataset object containing normalizer.
        index (int): Index for accessing normalization limits.
        eps (float): A small epsilon value to prevent values from exceeding bounds.

    Returns:
        numpy.ndarray: Unnormalized data.
    """
    mins = dataset.normalizer.mins[index]
    maxs = dataset.normalizer.maxs[index]
    if x.max() > 1 + eps or x.min() < -1 - eps:
        x = np.clip(x, -1, 1)

    x = (x + 1) / 2.
    return x * (maxs - mins) + mins


def make_timesteps(batch_size, i, device):
    """
    Generate timestep tensor for the current iteration.

    Args:
        batch_size (int): Number of samples in the batch.
        i (int): The timestep index.
        device (torch.device): The device to store the tensor.

    Returns:
        torch.Tensor: Timestep tensor.
    """
    return torch.full((batch_size,), i, device=device, dtype=torch.long)


#-----------------------------------------------------------------------------#
#----------------------------- Planning Setup --------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    """
    Argument parser with defaults for dataset and config.
    """
    dataset: str = 'walker2d-medium-replay-v2'
    config: str = 'config.locomotion'

# Initialize arguments
args = Parser().parse_args('plan')
args.horizon = 1024
args.scale = 0.1

# Set seed for reproducibility
set_seed(42)

#-----------------------------------------------------------------------------#
#--------------------------- Loading Experiments ----------------------------#
#-----------------------------------------------------------------------------#

# Load the diffusion model
diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset,
    # args.diffusion_loadpath,
    f'diffusion/defaults_H{args.horizon}_T{args.n_diffusion_steps}', 
    #device = args.device,
    epoch=args.diffusion_epoch, seed=args.seed,
)

# Load value function experiment (specific to basketball)
value_experiment = utils.load_diffusion(
    "/local2/yao/diffuser/logs/" #args.loadbase
    ,"basketball_single_game_wd_TS1000000" #args.dataset
    , f'values/defaults_H{args.horizon}_T{args.n_diffusion_steps}_d{args.discount}', #device = args.device,
    epoch=args.value_epoch, seed=args.seed,
)

# ## ensure that the diffusion model and value function are compatible with each other
# utils.check_compatibility(diffusion_experiment, value_experiment)

# Retrieve models and dataset
diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

# Initialize value function guide
value_function = value_experiment.ema
guide_config = utils.Config(args.guide, #device = args.device,
                             model=value_function, verbose=False)
guide = guide_config()

# Logger configuration
logger_config = utils.Config(
    utils.Logger,
    renderer=renderer,
    logpath=args.savepath,
    vis_freq=args.vis_freq,
    max_render=args.max_render,
)

# Policy configuration
## policies are wrappers around an unconditional diffusion model and a value guide
policy_config = utils.Config(
    args.policy,
    guide=guide,
    scale=args.scale,
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    ## sampling kwargs
    sample_fn=sampling.n_step_guided_p_sample,
    n_guide_steps=args.n_guide_steps,
    t_stopgrad=args.t_stopgrad,
    scale_grad_by_std=args.scale_grad_by_std,
    verbose=False,
)


# Initialize logger and policy
logger = logger_config()
policy = policy_config()


#-----------------------------------------------------------------------------#
#--------------------------- Update Heuristics ------------------------------#
#-----------------------------------------------------------------------------#

def update_heuristics(observation, next_obs, first=False):
    """
    Update the positions of opponents based on proximity to players using heuristics.
    
    Args:
        observation (numpy.ndarray): Current observation state.
        next_obs (numpy.ndarray): Next observation state.
        first (bool): If True, initialize positions. Defaults to False.

    Returns:
        numpy.ndarray: Updated observation state.
    """
    obs = observation.reshape(11, 6)
    nxt_obs = next_obs.reshape(11, 6)

    player_positions = obs[1:6, :3]  
    opponents_positions = obs[6:11, :3] 
    nxt_player_positions = nxt_obs[1:6, :3] 
    nxt_opponents_positions = nxt_obs[6:11, :3] 

    # Euclidean distances between the player and each opposing player
    distances = cdist(opponents_positions, player_positions)

    assigned_players = np.zeros(opponents_positions.shape[0], dtype=bool)

    # Move the opponents somewhere close to the player in next_obs that they are closeest to
    for opponent_index in range(len(opponents_positions)):
        available_players = np.where(~assigned_players)[0]
        closest_available_player_index = available_players[np.argmin(distances[opponent_index, available_players])]

        # Mark player as assigned
        assigned_players[closest_available_player_index] = True

        ### These are the values for the inequality for different types of Hueristics
        #original 2.3
        #loose 3.4
        #stagnent 100000000
        if distances[opponent_index, closest_available_player_index] > 2.3:

            # Calculate the direction vector
            direction_vector = player_positions[closest_available_player_index, :3] - opponents_positions[opponent_index, :3]

            # Normalize the direction vector
            normalized_direction = direction_vector / np.linalg.norm(direction_vector)

            # Calculate the offset as 0.5 in each dimension towards the player
            offset = 0.35 * normalized_direction
            nxt_opponents_positions[opponent_index, :3] = opponents_positions[opponent_index, :3] + offset
            nxt_opponents_positions[opponent_index, 0] = np.clip(nxt_opponents_positions[opponent_index, 0], 0, 94)
            nxt_opponents_positions[opponent_index, 1] = np.clip(nxt_opponents_positions[opponent_index, 1], 0, 50)
        else:
            ### use the below if not doing original
            # nxt_opponents_positions[opponent_index, :3] = opponents_positions[opponent_index, :3]

            # if using original use below, get offset for original below
            offset = np.random.uniform(low=[2.3, -0.15, 0], high=[2.6, 0.15,0])
            nxt_opponents_positions[opponent_index, :3] = nxt_player_positions[closest_available_player_index, :3] + offset
            nxt_opponents_positions[opponent_index, 0] = np.clip(nxt_opponents_positions[opponent_index, 0], 0, 94)
            nxt_opponents_positions[opponent_index, 1] = np.clip(nxt_opponents_positions[opponent_index, 1], 0, 50)


    nxt_obs[6:11, :3] = nxt_opponents_positions
    # make the movement columns the xyz position of next_obs minus the xyz in observation
    nxt_obs[6:11, 3:] = nxt_obs[6:11, :3] - obs[6:11, :3]
    return nxt_obs.flatten()


def update_heuristics2_3(observation, next_obs, first=False):
    """
    Update the positions of opponents using a 2-3 zone defense strategy.

    Args:
        observation (numpy.ndarray): Current observation state.
        next_obs (numpy.ndarray): Next observation state.
        first (bool): If True, initialize opponent positions in predefined zones.

    Returns:
        numpy.ndarray: Updated observation state.
    """
    obs = observation.reshape(11, 6)
    nxt_obs = next_obs.reshape(11, 6)

    player_positions = obs[1:6, :3]  
    opponents_positions = obs[6:11, :3] 
    nxt_opponents_positions = nxt_obs[6:11, :3]

    # Predefined zones for 2-3 defense
    zone_boundaries = [
    (26, 47, 63, 81),   # Zone 0 (on 3 side)
    (3, 24, 63, 81),    # Zone 1 (on 4 side)
    (31, 49, 81, 93),   # Zone 2
    (1, 19, 81, 93),    # Zone 3
    (19, 31, 75, 94)  # Zone 4
    ]

    if first:
        for i, (y_min, y_max, x_min, x_max) in enumerate(zone_boundaries):
            midpoint_x = (x_min + x_max) / 2
            midpoint_y = (y_min + y_max) / 2
            nxt_obs[i + 6, 0] = midpoint_x
            nxt_obs[i + 6, 1] = midpoint_y
        return nxt_obs.flatten()

    distances = np.full((5, 5), np.inf)

    for i, (y_min, y_max, x_min, x_max) in enumerate(zone_boundaries):
        players_in_zone_mask = np.logical_and.reduce((
        player_positions[:, 0] >= x_min,
        player_positions[:, 0] <= x_max,
        player_positions[:, 1] >= y_min,
        player_positions[:, 1] <= y_max))
    
        # Extract player positions within the specified zone
        players_in_zone = player_positions[players_in_zone_mask]
        # Update distances only for the positions within the specified zone
        distances[i, players_in_zone_mask] = cdist(opponents_positions[i].reshape(1,-1), players_in_zone)


    assigned_players = np.zeros(opponents_positions.shape[0], dtype=bool)

    # Move the opponents somewhere close to the player in next_obs that they are closeest to
    for opponent_index in range(len(opponents_positions)):

        available_players = np.where(~assigned_players)[0]

        closest_available_player_index = available_players[np.argmin(distances[opponent_index, available_players])]

        if distances[opponent_index, available_players][np.argmin(distances[opponent_index, available_players])] != np.inf:
            # Mark player as assigned
            assigned_players[closest_available_player_index] = True
    
            if distances[opponent_index, closest_available_player_index] > 3:
                # Calculate the direction vector
                direction_vector = player_positions[closest_available_player_index, :3] - opponents_positions[opponent_index, :3]
    
                # Normalize the direction vector
                normalized_direction = direction_vector / np.linalg.norm(direction_vector)
    
                # Calculate the offset as 0.35 in each dimension towards the player
                offset = 0.35 * normalized_direction
                nxt_opponents_positions[opponent_index, :3] = opponents_positions[opponent_index, :3] + offset
                nxt_opponents_positions[opponent_index, 0] = np.clip(nxt_opponents_positions[opponent_index, 0], zone_boundaries[opponent_index][2], zone_boundaries[opponent_index][3])
                nxt_opponents_positions[opponent_index, 1] = np.clip(nxt_opponents_positions[opponent_index, 1], zone_boundaries[opponent_index][0], zone_boundaries[opponent_index][1])
            else:
                nxt_opponents_positions[opponent_index, :3] = opponents_positions[opponent_index, :3]

        else:
            nxt_opponents_positions[opponent_index, :3] = opponents_positions[opponent_index, :3]

    nxt_obs[6:11, :3] = nxt_opponents_positions
    # make the movement columns the xyz position of next_obs minus the xyz in observation
    nxt_obs[6:11, 3:] = nxt_obs[6:11, :3] - obs[6:11, :3]
    
    return nxt_obs.flatten()


#-----------------------------------------------------------------------------#
#------------------------- Main Sampling and Logging ------------------------#
#-----------------------------------------------------------------------------#

# Constants
SAMPLING_NUM = 1
total_reward = np.array([0]*5)
groundtruth_reward = 0

### This is the number you want to batch by, if using Hueristics
first_num_of_observations = 100

pathid = 'new_test2_cond' + str(first_num_of_observations)
path = f"./logs/guided_samples{pathid}_{args.scale}"

### Set to True If you are using hueristics
use_hue = True

folder_existed = False # used to set this to True for debugging (No saving results)

# Create directory if it doesn't exist, (To continue interrupted processes)
if not os.path.exists(path):
    os.makedirs(path)
    print(f"Directory {path} created.")
    folder_existed = False
else:
    print(f"Directory {path} already exists.")

# Log rewards if wanted
# reward_log = open(f"{path}/reward_{SUBSET*COUNT}_{SUBSET*COUNT+SUBSET}.log", "w")

# Progress bar for dataset
pbar = tqdm(range(len(dataset)), desc="Planning: ")


for index in pbar:
    print(f"posession #{index}")
    observation = dataset.observations[index, 0]

    # Accumulate ground truth rewards
    groundtruth_reward += dataset.rewards[index]
    game_info = dataset.trajectory_game_record[index].split(".npy")[0]
    
    # Save ground truth observations if the folder didn't exist before
    if not folder_existed:
        savepath = os.path.join(f'{path}', f'{game_info}-{index}-groundtruth.npy')
        if not os.path.exists(savepath):
            torch.save(dataset.observations[index, :], savepath)
    
    # Save guided observations
    savepath = os.path.join(f'{path}', f'{game_info}-{index}-guided-245K.npy')
    if use_hue:
        if not os.path.exists(savepath):
        # sample the first 6 channels and get the first frame of the 1024

            ## format current observation for conditioning
            samples = None
            observations = np.zeros((5, 1024, 66))
            actions = np.zeros((5, 1024, 0))
            values = torch.zeros(5)
            additional_conditions = {}
            num_iterations = int(np.ceil(1024 // first_num_of_observations))

            for n in range(5):
                obs = observation
                conditions = {0: observation}
                # observations[n,0] = dataset.unnormalize(obs)

                # For the below loop, the 'update hueristic function can switch from 2_3 to not by changes the name of the function manually
                for i in range(num_iterations):
                    action, temp_samples = policy(conditions, batch_size=SAMPLING_NUM, verbose=args.verbose)
                    if (i == (num_iterations - 1)):
                        for j in range(1024 - (first_num_of_observations * (num_iterations - 1))):
                            obs = update_heuristics2_3(unnormalize(obs, dataset, j + len(conditions) - 1), temp_samples.observations[0,j + len(conditions) - 1])
                            observations[n,(first_num_of_observations*i) + j] = obs
                            obs = normalize(obs, dataset,j + len(conditions) - 1)
                            # store conditions to keep to use in the next iteration
                            additional_conditions[(i*first_num_of_observations)+j+1] = obs        
                    else:
                        for j in range(first_num_of_observations):
                            if (i == 0) and (j == 0):
                                obs = update_heuristics2_3(unnormalize(obs, dataset, j + len(conditions) - 1), temp_samples.observations[0,j + len(conditions) - 1], True)
                            else:
                                obs = update_heuristics2_3(unnormalize(obs, dataset, j + len(conditions) - 1), temp_samples.observations[0,j + len(conditions) - 1])
                            observations[n,(first_num_of_observations*i) + j] = obs
                            obs = normalize(obs, dataset,j + len(conditions) - 1)
                            additional_conditions[(i*first_num_of_observations)+j + 1] = obs
                    # actions[n,i] = temp_samples.actions[0,1]

                    # Add in usable conditions
                    conditions.update(additional_conditions)
                    additional_conditions.clear()

            t = make_timesteps(5, 0, policy.diffusion_model.betas.device)

            values = policy.guide(policy.normalizer.unnormalize(torch.tensor(observations)).float().to(args.device), None, t)

            samples = Trajectories(
                        actions=actions,
                        observations = observations,
                        values= values
                        )
    else:
        if not os.path.exists(savepath):
            conditions = {0: observation}
            action, samples = policy(conditions, batch_size=5, verbose=args.verbose)

            total_reward = np.add(total_reward, samples.values.cpu().detach().numpy())
        
    if not folder_existed:
        if not os.path.exists(savepath):
            savepath = os.path.join(f'{path}', f'{game_info}-{index}-guided-245K.npy')
            torch.save(samples.observations, savepath)
            # torch.save(samples.values.detach().cpu().numpy(), savepath)
            # print(samples.values)

        # Log Rewards if wanted
        # reward_log.write(f"{game_info},{samples.values.cpu().detach().numpy()}")
        # reward_log.write("\n")

        if index > 0 and index % 300 == 0:
            print(f"[Step: {index}] [Reward: {total_reward}]")

        pbar.set_description(f"[GT reward: {groundtruth_reward}] [Reward: {total_reward}]", refresh=True)

# Final print statements
print(f"Total reward: {total_reward}")
print(f"[Mean: {total_reward.mean()}] [MAX: {total_reward.max()}] [Std: {total_reward.std()}]")
print(f"Ground truth reward: {groundtruth_reward}")

