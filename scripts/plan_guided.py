import pdb
import os
import random
import torch
import numpy as np
from tqdm import tqdm

import diffuser.sampling as sampling
import diffuser.utils as utils


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

SAMPLING_NUM = 5
total_reward = np.array([0]*5)
groundtruth_reward = 0
path = f"/local2/yao/diffuser/logs/guided_samples_{args.scale}"
# SUBSET = 5000
# COUNT = 9
# reward_log = open(f"{path}/reward_{SUBSET*COUNT}_{SUBSET*COUNT+SUBSET}.log", "w")
pbar = tqdm(range(len(dataset)), desc="Planning: ")
for index in pbar:
    observation = dataset.observations[index, 0]
    groundtruth_reward += dataset.rewards[index]
    game_info = dataset.trajectory_game_record[index].split(".npy")[0]

    # savepath = os.path.join(f'{path}', f'{game_info}-{index}-groundtruth.npy')
    # torch.save(dataset.observations[index, :], savepath)
    # 1/0

    # sample the first 6 channels and get the first frame of the 1024

    ## format current observation for conditioning
    conditions = {0: observation}
    action, samples = policy(conditions, batch_size=SAMPLING_NUM, verbose=args.verbose)
    total_reward = np.add(total_reward, samples.values.cpu().detach().numpy())

    # print("GUIDED")
    savepath = os.path.join(f'{path}', f'{game_info}-{index}-guided-245K.npy')
    # print(savepath)
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

    if index > 0 and index % 300 == 0:
        print(f"[Step: {index}] [Reward: {total_reward}]")

    pbar.set_description(f"[GT reward: {groundtruth_reward}] [Reward: {total_reward}]", refresh=True)

print(f"Total reward: {total_reward}")
print(f"[Mean: {total_reward.mean()}] [MAX: {total_reward.max()}] [Std: {total_reward.std()}]")
print(f"Ground truth reward: {groundtruth_reward}")
# print(SUBSET*COUNT, SUBSET*COUNT+SUBSET)

