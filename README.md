# Planning with Diffusion &nbsp;&nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YajKhu-CUIGBJeQPehjVPJcK_b38a8Nc?usp=sharing)


Training and visualizing of diffusion models from [Planning with Diffusion for Flexible Behavior Synthesis](https://diffusion-planning.github.io/).

The [main branch](https://github.com/jannerm/diffuser/tree/main) contains code for training diffusion models and planning via value-function guided sampling on the D4RL locomotion environments.
The [kuka branch](https://github.com/jannerm/diffuser/tree/kuka) contains block-stacking experiments.
The [maze2d branch](https://github.com/jannerm/diffuser/tree/maze2d) contains goal-reaching via inpainting in the Maze2D environments.

<p align="center">
    <img src="https://diffusion-planning.github.io/images/diffuser-card.png" width="60%" title="Diffuser model">
</p>

**Update 10/17/2022:** A bug in the value function scaling has been fixed in [this commit](https://github.com/jannerm/diffuser/commit/3d7361c2d028473b601cc04f5eecd019e14eb4eb). Thanks to [Philemon Brakel](https://scholar.google.com/citations?user=Q6UMpRYAAAAJ&hl=en) for catching it!

## Quickstart

Load a pretrained diffusion model and sample from it in your browser with [scripts/diffuser-sample.ipynb](https://colab.research.google.com/drive/1YajKhu-CUIGBJeQPehjVPJcK_b38a8Nc?usp=sharing).


## Installation

```
conda env create -f environment.yml
conda activate diffuser
pip install -e .
```

## Using pretrained models

### Downloading weights

Download pretrained diffusion models and value functions with:
```
./scripts/download_pretrained.sh
```

This command downloads and extracts a [tarfile](https://drive.google.com/file/d/1srTq0OFQtWIv9A7fwm3fwh1StA__qr6y/view?usp=sharing) containing [this directory](https://drive.google.com/drive/folders/1ie6z3toz9OjcarJuwjQwXXzDwh1XnS02?usp=sharing) to `logs/pretrained`. The models are organized according to the following structure:
```
└── logs/pretrained
    ├── ${environment_1}
    │   ├── diffusion
    │   │   └── ${experiment_name}
    │   │       ├── state_${epoch}.pt
    │   │       ├── sample-${epoch}-*.png
    │   │       └── {dataset, diffusion, model, render, trainer}_config.pkl
    │   └── values
    │       └── ${experiment_name}
    │           ├── state_${epoch}.pt
    │           └── {dataset, diffusion, model, render, trainer}_config.pkl
    ├── ${environment_2}
    │   └── ...
```

The `state_${epoch}.pt` files contain the network weights and the `config.pkl` files contain the instantation arguments for the relevant classes.
The png files contain samples from different points during training of the diffusion model.

### Planning

To plan with guided sampling, run:
```
python scripts/plan_guided.py --dataset halfcheetah-medium-expert-v2 --logbase logs/pretrained
```

The `--logbase` flag points the [experiment loaders](scripts/plan_guided.py#L22-L30) to the folder containing the pretrained models.
You can override planning hyperparameters with flags, such as `--batch_size 8`, but the default
hyperparameters are a good starting point.

**Results.** The current codebase performs a few points better (averaged over environments) than
described in the arxiv v1 paper due to small tweaks to the architecture and objective. It is also
somewhat faster. The arxiv paper will be updated shortly to reflect these changes.

## Training from scratch

1. Train a diffusion model with:
```
python scripts/train.py --dataset halfcheetah-medium-expert-v2
```

The default hyperparameters are listed in [locomotion:diffusion](config/locomotion.py#L22-L65).
You can override any of them with flags, eg, `--n_diffusion_steps 100`.

2. Train a value function with:
```
python scripts/train_values.py --dataset halfcheetah-medium-expert-v2
```
See [locomotion:values](config/locomotion.py#L67-L108) for the corresponding default hyperparameters.


3. Plan using your newly-trained models with the same command as in the pretrained planning section, simply replacing the logbase to point to your new models:
```
python scripts/plan_guided.py --dataset halfcheetah-medium-expert-v2 --logbase logs
```
See [locomotion:plans](config/locomotion.py#L110-L149) for the corresponding default hyperparameters.

**Deferred f-strings.** Note that some planning script arguments, such as `--n_diffusion_steps` or `--discount`,
do not actually change any logic during planning, but simply load a different model using a deferred f-string.
For example, the following flags:
```
---horizon 32 --n_diffusion_steps 20 --discount 0.997
--value_loadpath 'f:values/defaults_H{horizon}_T{n_diffusion_steps}_d{discount}'
```
will resolve to a value checkpoint path of `values/defaults_H32_T20_d0.997`. It is possible to
change the horizon of the diffusion model after training (see [here](https://colab.research.google.com/drive/1YajKhu-CUIGBJeQPehjVPJcK_b38a8Nc?usp=sharing) for an example),
but not for the value function.

## Docker

1. Build the image:
```
docker build -f Dockerfile . -t diffuser
```

2. Test the image:
```
docker run -it --rm --gpus all \
    --mount type=bind,source=$PWD,target=/home/code \
    --mount type=bind,source=$HOME/.d4rl,target=/root/.d4rl \
    diffuser \
    bash -c \
    "export PYTHONPATH=$PYTHONPATH:/home/code && \
    python /home/code/scripts/train.py --dataset halfcheetah-medium-expert-v2 --logbase logs"
```

## Singularity

1. Build the image:
```
singularity build --fakeroot diffuser.sif Singularity.def
```

2. Test the image:
```
singularity exec --nv --writable-tmpfs diffuser.sif \
        bash -c \
        "pip install -e . && \
        python scripts/train.py --dataset halfcheetah-medium-expert-v2 --logbase logs"
```


## Running on Azure

#### Setup

1. Tag the Docker image (built in the [Docker section](#Docker)) and push it to Docker Hub:
```
export DOCKER_USERNAME=$(docker info | sed '/Username:/!d;s/.* //')
docker tag diffuser ${DOCKER_USERNAME}/diffuser:latest
docker image push ${DOCKER_USERNAME}/diffuser
```

3. Update [`azure/config.py`](azure/config.py), either by modifying the file directly or setting the relevant [environment variables](azure/config.py#L47-L52). To set the `AZURE_STORAGE_CONNECTION` variable, navigate to the `Access keys` section of your storage account. Click `Show keys` and copy the `Connection string`.

4. Download [`azcopy`](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10): `./azure/download.sh`

#### Usage

Launch training jobs with `python azure/launch.py`. The launch script takes no command-line arguments; instead, it launches a job for every combination of hyperparameters in [`params_to_sweep`](azure/launch_train.py#L36-L38).


#### Viewing results

To rsync the results from the Azure storage container, run `./azure/sync.sh`.

To mount the storage container:
1. Create a blobfuse config with `./azure/make_fuse_config.sh`
2. Run `./azure/mount.sh` to mount the storage container to `~/azure_mount`

To unmount the container, run `sudo umount -f ~/azure_mount; rm -r ~/azure_mount`


## Reference
```
@inproceedings{janner2022diffuser,
  title = {Planning with Diffusion for Flexible Behavior Synthesis},
  author = {Michael Janner and Yilun Du and Joshua B. Tenenbaum and Sergey Levine},
  booktitle = {International Conference on Machine Learning},
  year = {2022},
}
```


## Acknowledgements

The diffusion model implementation is based on Phil Wang's [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) repo.
The organization of this repo and remote launcher is based on the [trajectory-transformer](https://github.com/jannerm/trajectory-transformer) repo.
