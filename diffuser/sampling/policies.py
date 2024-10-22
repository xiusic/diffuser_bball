from collections import namedtuple
import torch
import einops
import pdb
import numpy as np
import ipdb

import diffuser.utils as utils
from diffuser.datasets.preprocessing import get_policy_preprocess_fn


Trajectories = namedtuple('Trajectories', 'actions observations values')


class GuidedPolicy:

    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        self.guide = guide
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = diffusion_model.action_dim
        self.preprocess_fn = get_policy_preprocess_fn(preprocess_fns)
        self.sample_kwargs = sample_kwargs

    def __call__(self, conditions, batch_size=1, verbose=True):
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        conditions = self._format_conditions(conditions, batch_size)

        def make_timesteps(batch_size, i, device):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            return t

        ## run reverse diffusion process
        samples = self.diffusion_model(conditions, guide=self.guide, verbose=verbose, **self.sample_kwargs)
        trajectories = utils.to_np(samples.trajectories)
        # print(trajectories.shape)
        # print(np.dtype(trajectories))
        t = make_timesteps(batch_size, 0, samples.trajectories.device)
        values = self.guide(samples.trajectories, None, t)

        # ### non-guided
        # samples = self.diffusion_model(conditions)
        # trajectories = utils.to_np(samples.trajectories)
        # t = make_timesteps(batch_size, 0, samples.trajectories.device)
        # values = self.guide(samples.trajectories, None, t)

        ## extract action [ batch_size x horizon x transition_dim ]
        actions = trajectories[:, :, :self.action_dim]
        # actions = self.normalizer.unnormalize(actions)                             # [BUG]: zero-size array cannot normalize

        ## extract first action
        action = actions[0, 0]

        normed_observations = trajectories[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations)

        trajectories = Trajectories(actions, observations, values)
        return action, trajectories

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        # [BUG]: self.normalizer is a string instead of a class
        # conditions = utils.apply_dict(
        #     self.normalizer.normalize,
        #     conditions,
        # )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda:0')
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions
