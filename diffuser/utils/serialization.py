import os
import pickle
import glob
import torch
import pdb

from collections import namedtuple
from diffuser.datasets.sequence import BBSequenceDataset, BBwdGoalDataset, BBwDirStatSequenceDataset

DiffusionExperiment = namedtuple('Diffusion', 'dataset renderer model diffusion ema trainer epoch')

def mkdir(savepath):
    """
        returns `True` iff `savepath` is created
    """
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        return True
    else:
        return False

def get_latest_epoch(loadpath):
    states = glob.glob1(os.path.join(*loadpath), 'state_*')
    latest_epoch = -1
    for state in states:
        epoch = int(state.replace('state_', '').replace('.pt', ''))
        latest_epoch = max(epoch, latest_epoch)
    return latest_epoch

def load_config(*loadpath):
    loadpath = os.path.join(*loadpath)
    config = pickle.load(open(loadpath, 'rb'))
    print(f'[ utils/serialization ] Loaded config from {loadpath}')
    print(config)
    return config

def load_diffusion(*loadpath, epoch='latest', device='cuda:0', seed=None):
    dataset_config = load_config(*loadpath, 'dataset_config.pkl')
    render_config = load_config(*loadpath, 'render_config.pkl')
    model_config = load_config(*loadpath, 'model_config.pkl')
    diffusion_config = load_config(*loadpath, 'diffusion_config.pkl')
    trainer_config = load_config(*loadpath, 'trainer_config.pkl')

    ## remove absolute path for results loaded from azure
    ## @TODO : remove results folder from within trainer class
    trainer_config._dict['results_folder'] = os.path.join(*loadpath)

    if loadpath[1] == "basketball_single_game":
        dataset = BBSequenceDataset("")
    elif loadpath[1].startswith("basketball_single_game_wd"):
        dataset = BBwdGoalDataset("../../data/transpose_files/", reward_path="../../data/2_final_json_rewards/")
    elif loadpath[1] == "basketball_single_game_wDirStat":
        dataset = BBwDirStatSequenceDataset("")
    else:
        dataset = dataset_config(seed=seed)

    renderer = None
    # renderer = render_config()
    model = model_config()
    diffusion = diffusion_config(model, device = device)
    trainer = trainer_config(diffusion, dataset, renderer, device = device)

    if epoch == 'latest':
        epoch = get_latest_epoch(loadpath)

    print(f'\n[ utils/serialization ] Loading model epoch: {epoch}\n')

    trainer.load(epoch)

    return DiffusionExperiment(dataset, renderer, model, diffusion, trainer.ema_model, trainer, epoch)

def check_compatibility(experiment_1, experiment_2):
    '''
        returns True if `experiment_1 and `experiment_2` have
        the same normalizers and number of diffusion steps
    '''
    normalizers_1 = experiment_1.dataset.normalizer.get_field_normalizers()
    normalizers_2 = experiment_2.dataset.normalizer.get_field_normalizers()
    for key in normalizers_1:
        norm_1 = type(normalizers_1[key])
        norm_2 = type(normalizers_2[key])
        assert norm_1 == norm_2, \
            f'Normalizers should be identical, found {norm_1} and {norm_2} for field {key}'

    n_steps_1 = experiment_1.diffusion.n_timesteps
    n_steps_2 = experiment_2.diffusion.n_timesteps
    assert n_steps_1 == n_steps_2, \
        ('Number of timesteps should match between diffusion experiments, '
        f'found {n_steps_1} and {n_steps_2}')
