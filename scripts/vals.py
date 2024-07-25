import pdb
import os
import sys
import random
import torch
import numpy as np
from tqdm import tqdm
import io

import diffuser.sampling as sampling
import diffuser.utils as utils
from diffuser.sampling.policies import Trajectories
from scipy.spatial.distance import cdist
import ipdb


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

# ## load diffusion model and value function from disk
# diffusion_experiment = utils.load_diffusion(
#     args.loadbase, args.dataset, args.diffusion_loadpath,
#     epoch=args.diffusion_epoch, seed=args.seed,
# )
# value_experiment = utils.load_diffusion(
#     args.loadbase, args.dataset, args.value_loadpath,
#     epoch=args.value_epoch, seed=args.seed,
# )

## load diffusion model and value function from disk
diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset,
    # args.diffusion_loadpath,
    f'diffusion/defaults_H{args.horizon}_T{100}', 
    #device = args.device,
    epoch=args.diffusion_epoch, seed=args.seed,
)
value_experiment = utils.load_diffusion(
    "/local2/yao/diffuser/logs/" #args.loadbase
    ,"basketball_single_game_wd_TS1000000" #args.dataset
    , f'values/defaults_H{args.horizon}_T{args.n_diffusion_steps}_d{args.discount}', #device = args.device,
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

def normalize(x, dataset):
    mins = dataset.mins
    maxs = dataset.maxs
    ## [ 0, 1 ]
    nonzero_i = np.abs(maxs - mins) > 0
    x[nonzero_i] = (x[nonzero_i] - mins[nonzero_i]) / (maxs[nonzero_i] - mins[nonzero_i])
    ## [ -1, 1 ]
    x = 2 * x - 1
    return x

# order ran too
# file_names = [
#     "guided_samplesact_(2_3)_50100_0.1",
#     "guided_samplesact_50100_0.1",
#     "guided_samplesact_loose_50100_0.1",
#     "guided_samplesact_original_50100_0.1",
#     "guided_samplesact25_loose_50100_0.1",
#     "guided_samplesact25_original_50100_0.1",
#     "guided_samplesact25_(2_3)_50100_0.1",
#     "guided_samplesact25_50_0.1"]
replace = "guided_samplesact_original_50100_0.1"

# CUDA_VISIBLE_DEVICES=5 python ./scripts/vals.py --dataset basketball_single_game_wd_act25 --logbase /local2/dmreynos/diffuser_bball/logs/ --diffusion_epoch epoch_50

path = f"/local2/dmreynos/diffuser_bball/logs/{replace}"  # Replace with the path to your .npy file
# get cases
# get gifs and values for organized hue 
# /2016.NBA.Raw.SportVU.Game.Logs12.05.2015.POR.at.MIN_dir-0-guided-245K.npy"
# Load data from the .npy file
pos = 19
with open(f"rewards_{replace}_obs_{pos}.log", "w") as f:
    # Redirect stdout to the log file
    sys.stdout = f
    all_files = os.listdir(path)
    total_reward = np.array([0]*5)
    values_list = []
    filtered_files = [file for file in all_files if file.endswith('.npy') and 'groundtruth' not in file.lower()]
    for file_name in filtered_files:
        if file_name != f"2016.NBA.Raw.SportVU.Game.Logs12.05.2015.POR.at.MIN_dir-{pos}-guided-245K.npy":
            continue  # Skip this specific file
        file_path = os.path.join(path, file_name)
        data = np.load(file_path, allow_pickle=True)
        # print(data[0].shape)
        #print(data.files)
        pickle_file = data['archive/data.pkl']
        file_obj = io.BytesIO(pickle_file)
        observations = np.load(file_obj, allow_pickle=True)
        # print(observations.shape)
        def make_timesteps(batch_size, i, device):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            return t
        t = make_timesteps(5, 0, policy.diffusion_model.betas.device)
        # values = policy.guide(torch.tensor(observations).float().to('cuda:0'), None, t)
        values = policy.guide(torch.tensor(np.apply_along_axis(lambda obs: normalize(obs, dataset), 2, observations.copy())).float().to('cuda:0'), None, t)
        total_reward = np.add(total_reward, values.cpu().detach().numpy())
        values_list.append(values.cpu().detach().numpy())
        # print(values)

    print(f"Total reward: {total_reward}")
    print(f"[Mean: {total_reward.mean()}] [MAX: {total_reward.max()}] [Std: {total_reward.std()}]")
    values_list = np.array(values_list)
    average_values = np.mean(values_list, axis = 0)
    # std_val = values_list.std()
    std_val = np.std(values_list, axis=0)
    print("average per possession is", average_values, "Std:", std_val)

sys.stdout = sys.__stdout__