from collections import namedtuple
import numpy as np
import torch
import pdb
import json
import os
import pickle
import time
from tqdm import tqdm

from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset
from .normalization import DatasetNormalizer, LimitsNormalizer
from .buffer import ReplayBuffer


Batch = namedtuple('Batch', 'trajectories conditions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')


class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env='hopper-medium-replay', horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True, seed=None):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = load_environment(env)
        self.env.seed(seed)
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        itr = sequence_dataset(env, self.preprocess_fn)

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()

        print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)
        batch = Batch(trajectories, conditions)
        return batch


class BBSequenceDataset(torch.utils.data.Dataset):

    def __init__(self, filepath):
        
        self.observations = np.load(filepath, allow_pickle=True) #TODO  (N * T * d)
        self.observation_dim = 33
        self.action_dim = 0

        max_len = -1
        for obs in self.observations:
            if len(obs) > max_len:
                max_len = len(obs)

        valid_observations = []
        for i in range(len(self.observations)):
            # self.observations[i] = np.pad(self.observations[i], (0, max_len-len(self.observations[i])), 'constant')
            self.observations[i] = np.concatenate([self.observations[i], np.zeros((max_len-len(self.observations[i]), self.observations[i].shape[-1]))])
            try:
                self.observations[i] = self.observations[i].astype(np.float32)
            except:
                continue
            valid_observations.append(self.observations[i])

        self.observations = np.concatenate([np.expand_dims(ob, axis=0) for ob in valid_observations])
        self.observations  = self.observations[:, :1024, :]
        print(self.observations.shape)
        self.normalize()

        
    def normalize(self):
        self.observations  = self.observations.reshape(-1, 33)
        self.mins = self.observations.min(axis=0)
        self.maxs = self.observations.max(axis=0)
        ## [ 0, 1 ]
        nonzero_i = np.abs(self.maxs - self.mins) > 0
        self.observations[:, nonzero_i] = (self.observations[:, nonzero_i] - self.mins[nonzero_i]) / (self.maxs[nonzero_i] - self.mins[nonzero_i])
        ## [ -1, 1 ]
        self.observations = 2 * self.observations - 1
        self.observations = self.observations.reshape(-1, 1024, 33)

    def unnormalize(self, x, eps=1e-4):
        '''
            x : [ -1, 1 ]
        '''
        # print(eps)
        if x.max() > 1 + eps or x.min() < -1 - eps:
            # print(f'[ datasets/mujoco ] Warning: sample out of range | ({x.min():.4f}, {x.max():.4f})')
            x = np.clip(x, -1, 1)

        ## [ -1, 1 ] --> [ 0, 1 ]
        x = (x + 1) / 2.

        return x * (self.maxs - self.mins) + self.mins


    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        #normalization?
        observations = np.asarray(self.observations[idx]) # T * d
        conditions = self.get_conditions(observations)
        trajectories = observations
        batch = Batch(trajectories, conditions)
        return batch



class BBwdSequenceDataset(torch.utils.data.Dataset):

    def __init__(self, filepath, reward_path=None):
        self.observation_dim = 66
        self.action_dim = 0

        ### hard code
        self.max_path_length = 1024

        # 480 files
        observation_paths = [os.path.join(filepath, f) for f in os.listdir(filepath) if "2016" in f and "dir" in f]
        # observation_paths = [os.path.join(filepath, f) for f in os.listdir(filepath) if "2015" in f and "dir" in f]

        print(f"file count: {len(observation_paths)}")

        if reward_path is not None:
            reward_paths = [os.path.join(reward_path, f) for f in os.listdir(reward_path) if "2016" in f]
            # reward_paths = [os.path.join(reward_path, f) for f in os.listdir(reward_path) if "2015" in f]
            print(f"reward count: {len(reward_paths)}")
        
        # load the processed file if already processed it
        observation_file = f"{filepath}processed_observations_test_unnormalized.pkl"
        reward_file = f"{reward_path}processed_rewards_test_unnormalized.pkl"
        trajectory_game_file = f"{reward_path}processed_trajectory_game_test_unnormalized.pkl"
        if not os.path.isfile(observation_file):
            valid_observations, valid_rewards = [], []
            trajectory_game_record = []
            for index, observation_path in tqdm(enumerate(observation_paths), total=len(observation_paths), desc="processing files: "):
                # process observation file
                observations = np.load(observation_path, allow_pickle=True)

                if reward_path is not None:
                    game_info = observation_path.split("/")[-1].split("_dir")[0]            # 11.06.2015.TOR.at.ORL
                    with open(f"{reward_path}{game_info}.7z_rewards.json") as f:
                        rewards = json.load(f)

                for i in range(len(observations)):
                    try:
                        last_observation = observations[i][-1].astype(np.float32)
                    except:
                        continue

                    # pad if less than self.max_path_length
                    if len(observations[i]) >= self.max_path_length:
                        observations[i] = observations[i][:self.max_path_length]
                    else:
                        paddings = np.zeros((self.max_path_length-len(observations[i]), observations[i].shape[-1]))
                        paddings += last_observation
                        observations[i] = np.concatenate([observations[i], paddings])
                    try:
                        observations[i] = observations[i].astype(np.float32)
                    except:
                        continue

                    valid_observations.append(observations[i])
                    trajectory_game_record.append(observation_path.split("/")[-1])      # 2016.NBA.Raw.SportVU.Game.Logs12.05.2015.POR.at.MIN_dir.npy

                    # process reward file
                    if reward_path is not None:
                        valid_rewards.append(rewards[str(i)])

            self.observations = np.array(valid_observations)
            self.observations = self.observations[:, :1024, :]
            self.normalize()
            self.trajectory_game_record = np.array(trajectory_game_record)
            if reward_path is not None:
                self.rewards = np.array(valid_rewards)
                with open(reward_file, 'wb') as file:
                    pickle.dump(self.rewards, file)

            with open(observation_file, 'wb') as file:
                pickle.dump(self.observations, file)
            with open(trajectory_game_file, 'wb') as file:
                pickle.dump(self.trajectory_game_record, file)
        else:
            with open(observation_file, 'rb') as file:
                self.observations = pickle.load(file)
            with open(trajectory_game_file, 'rb') as file:
                self.trajectory_game_record = pickle.load(file)

            if reward_path is not None:
                with open(reward_file, 'rb') as file:
                    self.rewards = pickle.load(file)

        print(self.observations.shape, self.trajectory_game_record.shape)
        # 210952 (train); 68701 (test)]

        self.normalizer = LimitsNormalizer(self.observations)
        self.normalize()                                        # used when the input file is unnormalized
        
    def normalize(self):
        self.observations  = self.observations.reshape(-1, 66)
        self.mins = self.observations.min(axis=0)
        self.maxs = self.observations.max(axis=0)
        ## [ 0, 1 ]
        nonzero_i = np.abs(self.maxs - self.mins) > 0
        self.observations[:, nonzero_i] = (self.observations[:, nonzero_i] - self.mins[nonzero_i]) / (self.maxs[nonzero_i] - self.mins[nonzero_i])
        ## [ -1, 1 ]
        self.observations = 2 * self.observations - 1
        self.observations = self.observations.reshape(-1, 1024, 66)

    def unnormalize(self, x, eps=1e-4):
        '''
            x : [ -1, 1 ]
        '''
        # print(eps)
        if x.max() > 1 + eps or x.min() < -1 - eps:
            # print(f'[ datasets/mujoco ] Warning: sample out of range | ({x.min():.4f}, {x.max():.4f})')
            x = np.clip(x, -1, 1)

        ## [ -1, 1 ] --> [ 0, 1 ]
        x = (x + 1) / 2.

        return x * (self.maxs - self.mins) + self.mins


    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        #normalization?
        observations = np.asarray(self.observations[idx]) # T * d
        conditions = self.get_conditions(observations)
        trajectories = observations
        batch = Batch(trajectories, conditions)
        return batch


class BBwdValueDataset(BBwdSequenceDataset):
    '''
        adds a value field to the datapoints for training the value function
    '''

    def __init__(self, *args, discount=0.99, normed=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:,None]
        self.normed = False
        if normed:
            self.vmin, self.vmax = self._get_bounds()
            self.normed = True

    def _get_bounds(self):
        print('[ datasets/sequence ] Getting value dataset bounds...', end=' ', flush=True)
        vmin = np.inf
        vmax = -np.inf
        for i in range(len(self.indices)):
            value = self.__getitem__(i).values.item()
            vmin = min(value, vmin)
            vmax = max(value, vmax)
        print('✓')
        return vmin, vmax

    def normalize_value(self, value):
        ## [0, 1]
        normed = (value - self.vmin) / (self.vmax - self.vmin)
        ## [-1, 1]
        normed = normed * 2 - 1
        return normed

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        
        rewards = np.array([self.rewards[idx]], dtype=np.float32)

        discounts = self.discounts[:len(rewards)]
        value = (discounts * rewards).sum()
        if self.normed:
            value = self.normalize_value(value)
        value = np.array([value], dtype=np.float32)
        value_batch = ValueBatch(*batch, value)
        return value_batch
    

class BBwdGoalDataset(BBwdSequenceDataset):

    def get_conditions(self, observations):
        '''
            condition on both the current observation and the last observation in the plan
        '''
        return {
            0: observations[0],
            self.horizon - 1: observations[-1],
        }


class BBwDirStatSequenceDataset(torch.utils.data.Dataset):

    def __init__(self, filepath):
        
        self.observations = np.load(filepath, allow_pickle=True) #TODO  (N * T * d)
        self.observation_dim = 11*22
        self.action_dim = 0

        max_len = -1
        for obs in self.observations:
            if len(obs) > max_len:
                max_len = len(obs)

        valid_observations = []
        for i in range(len(self.observations)):
            # self.observations[i] = np.pad(self.observations[i], (0, max_len-len(self.observations[i])), 'constant')
            # self.observations[i] = np.concatenate([self.observations[i], np.zeros((max_len-len(self.observations[i]), self.observations[i].shape[-1]))])
            try:
                last_observation = self.observations[i][-1].astype(np.float32)
            except:
                continue
            
            paddings = np.zeros((max_len-len(self.observations[i]), self.observations[i].shape[-1]))
            
            paddings += last_observation
            self.observations[i] = np.concatenate([self.observations[i], paddings])
            try:
                self.observations[i] = self.observations[i].astype(np.float32)
            except:
                continue
            valid_observations.append(self.observations[i])

        self.observations = np.concatenate([np.expand_dims(ob, axis=0) for ob in valid_observations])
        self.observations  = self.observations[:, :1024, :]
        print(self.observations.shape)
        self.normalize()

        
    def normalize(self):
        self.observations  = self.observations.reshape(-1, 11*22)
        self.mins = self.observations.min(axis=0)
        self.maxs = self.observations.max(axis=0)
        ## [ 0, 1 ]
        nonzero_i = np.abs(self.maxs - self.mins) > 0
        self.observations[:, nonzero_i] = (self.observations[:, nonzero_i] - self.mins[nonzero_i]) / (self.maxs[nonzero_i] - self.mins[nonzero_i])
        ## [ -1, 1 ]
        self.observations = 2 * self.observations - 1
        self.observations = self.observations.reshape(-1, 1024, 11*22)

    def unnormalize(self, x, eps=1e-4):
        '''
            x : [ -1, 1 ]
        '''
        # print(eps)
        if x.max() > 1 + eps or x.min() < -1 - eps:
            # print(f'[ datasets/mujoco ] Warning: sample out of range | ({x.min():.4f}, {x.max():.4f})')
            x = np.clip(x, -1, 1)

        ## [ -1, 1 ] --> [ 0, 1 ]
        x = (x + 1) / 2.

        return x * (self.maxs - self.mins) + self.mins


    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        #normalization?
        observations = np.asarray(self.observations[idx]) # T * d
        conditions = self.get_conditions(observations)
        trajectories = observations
        batch = Batch(trajectories, conditions)
        return batch

class GoalDataset(SequenceDataset):

    def get_conditions(self, observations):
        '''
            condition on both the current observation and the last observation in the plan
        '''
        return {
            0: observations[0],
            self.horizon - 1: observations[-1],
        }


class ValueDataset(SequenceDataset):
    '''
        adds a value field to the datapoints for training the value function
    '''

    def __init__(self, *args, discount=0.99, normed=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:,None]
        self.normed = False
        if normed:
            self.vmin, self.vmax = self._get_bounds()
            self.normed = True

    def _get_bounds(self):
        print('[ datasets/sequence ] Getting value dataset bounds...', end=' ', flush=True)
        vmin = np.inf
        vmax = -np.inf
        for i in range(len(self.indices)):
            value = self.__getitem__(i).values.item()
            vmin = min(value, vmin)
            vmax = max(value, vmax)
        print('✓')
        return vmin, vmax

    def normalize_value(self, value):
        ## [0, 1]
        normed = (value - self.vmin) / (self.vmax - self.vmin)
        ## [-1, 1]
        normed = normed * 2 - 1
        return normed

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        path_ind, start, end = self.indices[idx]
        rewards = self.fields['rewards'][path_ind, start:]
        discounts = self.discounts[:len(rewards)]
        value = (discounts * rewards).sum()
        if self.normed:
            value = self.normalize_value(value)
        value = np.array([value], dtype=np.float32)
        value_batch = ValueBatch(*batch, value)     # trajectories: (T, 23); reward: (1, )
        return value_batch

