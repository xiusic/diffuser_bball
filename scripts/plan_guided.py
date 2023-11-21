import pdb
import os
import sys
import random
import torch
import numpy as np
from tqdm import tqdm

import diffuser.sampling as sampling
import diffuser.utils as utils
from diffuser.sampling.policies import Trajectories
from scipy.spatial.distance import cdist


#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'walker2d-medium-replay-v2'
    config: str = 'config.locomotion'

args = Parser().parse_args('plan')
args.horizon = 1024
# args.t_stopgrad = 1
# args.n_guide_steps = 4
args.scale = 0.1

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

set_seed(42)

#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#

## load diffusion model and value function from disk
diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed,
)
value_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.value_loadpath,
    epoch=args.value_epoch, seed=args.seed,
)

# ## ensure that the diffusion model and value function are compatible with each other
# utils.check_compatibility(diffusion_experiment, value_experiment)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer
# print(dataset.mins.shape)
# print(dataset.normalizer.mins.shape)
# break
# print(dataset.unnormalize(dataset.observations[0, 1]))
# print(dir(dataset))
# exit()

## initialize value guide
value_function = value_experiment.ema
guide_config = utils.Config(args.guide, model=value_function, verbose=False)
guide = guide_config()

logger_config = utils.Config(
    utils.Logger,
    renderer=renderer,
    logpath=args.savepath,
    vis_freq=args.vis_freq,
    max_render=args.max_render,
)

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

logger = logger_config()
policy = policy_config()


#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#
def update_heuristics(observation, next_obs):
    obs = observation.reshape(11, 6)
    nxt_obs = next_obs.reshape(11, 6)
    # player_observation = obs[2:6, :]  
    # opponents_observation = obs[6:11, :] 

    player_positions = obs[1:6, :3]  
    opponents_positions = obs[6:11, :3] 
    nxt_player_positions = nxt_obs[1:6, :3] 
    nxt_opponents_positions = nxt_obs[6:11, :3] 

    # Euclidean distances between the player and each opposing player
    distances = cdist(opponents_positions, player_positions)

    assigned_players = np.zeros(opponents_positions.shape[0], dtype=bool)

    # move the opponents somewhere close to the player in next_obs that they are closeest to
    for opponent_index in range(len(opponents_positions)):
        available_players = np.where(~assigned_players)[0]
        closest_available_player_index = available_players[np.argmin(distances[opponent_index, available_players])]

        # Mark player as assigned
        assigned_players[closest_available_player_index] = True

        if distances[opponent_index, closest_available_player_index] > 2:
            # Calculate the direction vector
            direction_vector = player_positions[closest_available_player_index, :3] - opponents_positions[opponent_index, :3]

            # Normalize the direction vector
            normalized_direction = direction_vector / np.linalg.norm(direction_vector)

            # Calculate the offset as 0.5 in each dimension towards the player
            offset = 0.5 * normalized_direction
            nxt_opponents_positions[opponent_index, :3] = opponents_positions[opponent_index, :3] + offset
            nxt_opponents_positions[opponent_index, 0] = np.clip(nxt_opponents_positions[opponent_index, 0], 0, 94)
            nxt_opponents_positions[opponent_index, 1] = np.clip(nxt_opponents_positions[opponent_index, 1], 0, 50)
        else:
            # get offset
            offset = np.random.uniform(low=[-2.3, -0.15, -0.000001], high=[-2.0, 0.15, 0.000001])
            nxt_opponents_positions[opponent_index, :3] = nxt_player_positions[closest_available_player_index, :3] + offset
            nxt_opponents_positions[opponent_index, 0] = np.clip(nxt_opponents_positions[opponent_index, 0], 0, 94)
            nxt_opponents_positions[opponent_index, 1] = np.clip(nxt_opponents_positions[opponent_index, 1], 0, 50)

    nxt_obs[6:11, :3] = nxt_opponents_positions
    # make the movement columns the xyz position of next_obs minus the xyz in observation
    nxt_obs[6:11, 3:] = nxt_obs[6:11, :3] - obs[6:11, :3]
    # nxt_obs[6:11, 3:] = obs[6:11, 3:]
    return nxt_obs.flatten()
# 94 x 50
def normalize(x, dataset):
    mins = dataset.mins
    maxs = dataset.maxs
    ## [ 0, 1 ]
    nonzero_i = np.abs(maxs - mins) > 0
    x[nonzero_i] = (x[nonzero_i] - mins[nonzero_i]) / (maxs[nonzero_i] - mins[nonzero_i])
    ## [ -1, 1 ]
    x = 2 * x - 1
    return x


SAMPLING_NUM = 1
total_reward = np.array([0]*5)
groundtruth_reward = 0
pathid = 'hue'
path = f"./logs/guided_samples{pathid}_{args.scale}"
folder_existed = True
if not os.path.exists(path):
    os.makedirs(path)
    print(f"Directory {path} created.")
    folder_existed = False
else:
    print(f"Directory {path} already exists.")
    folder_existed = True
# SUBSET = 5000
# COUNT = 9
# reward_log = open(f"{path}/reward_{SUBSET*COUNT}_{SUBSET*COUNT+SUBSET}.log", "w")
pbar = tqdm(range(len(dataset)), desc="Planning: ")
for index in pbar:
    print(f"posession #{index}")
    observation = dataset.observations[index, 0]
    # print(dataset.observations)

    # print(dataset.observations[5])
    # print(observation.shape) (66,)
    # print(dataset.observations.shape) (68701, 1024, 66)
    # print(type(observation)) <class 'numpy.ndarray'>
    # print(type(dataset.observations))
    # print(dataset.observations[index, 0])
    # print(dataset.observations[index, 1])
    # print(dataset.observations[index, 2])
    # print(dataset.observations[index, 3])
    
    groundtruth_reward += dataset.rewards[index]
    game_info = dataset.trajectory_game_record[index].split(".npy")[0]
    
    if not folder_existed:
        savepath = os.path.join(f'{path}', f'{game_info}-{index}-groundtruth.npy')
        torch.save(dataset.observations[index, :], savepath)
    # 1/0

    # sample the first 6 channels and get the first frame of the 1024

    ## format current observation for conditioning
    samples = None
    observations = np.zeros((5, 1024, 66))
    actions = np.zeros((5, 1024, 0))
    values = torch.zeros(5)
    for n in range(5):
        obs = observation
        conditions = {0: observation}
        observations[n,0] = dataset.unnormalize(obs)
        for i in range(1,1024):
            action, temp_samples = policy(conditions, batch_size=SAMPLING_NUM, verbose=args.verbose)
            obs = update_heuristics(dataset.unnormalize(obs), temp_samples.observations[0,1])
            observations[n,i] = obs
            obs = normalize(obs, dataset)
            # actions[n,i] = temp_samples.actions[0,1]

            #replace starting condition (idea 1)
            conditions = {0: obs}
            #add in condition (idea 2)
            # conditions[i] = obs
    samples = Trajectories(
                actions=actions,
                observations = observations,
                values= values
                )

    # action, samples = policy(conditions, batch_size=SAMPLING_NUM, verbose=args.verbose)
    # print(samples.observations.shape) (5, 1024, 66) 
    # print(samples.values.shape) torch.Size([5])
    # print(samples.actions.shape) (5, 1024, 0)
    # print(samples.actions)
    # print(action)
    # print(samples)
    #uncomment below
    # total_reward = np.add(total_reward, samples.values.cpu().detach().numpy())
    
    # print("GUIDED")
    if not folder_existed:
        savepath = os.path.join(f'{path}', f'{game_info}-{index}-guided-245K.npy')
        print(savepath)
        torch.save(samples.observations, savepath)
        # print(samples.values)

    # print("NON-GUIDED")
    # savepath = os.path.join(f'{path}', f'{game_info}-{index}-nonguided-245K.npy')
    # print(savepath)
    # torch.save(non_guided_samples.observations, savepath)
    # print(non_guided_samples.values)
    # 1/0
    # reward_log.write(f"{game_info},{samples.values.cpu().detach().numpy()}")
    # reward_log.write("\n")

    # if index > 0 and index % 300 == 0:
    #     print(f"[Step: {index}] [Reward: {total_reward}]")

    # pbar.set_description(f"[GT reward: {groundtruth_reward}] [Reward: {total_reward}]", refresh=True)

# print(f"Total reward: {total_reward}")
# print(f"[Mean: {total_reward.mean()}] [MAX: {total_reward.max()}] [Std: {total_reward.std()}]")
# print(f"Ground truth reward: {groundtruth_reward}")
# print(SUBSET*COUNT, SUBSET*COUNT+SUBSET)

