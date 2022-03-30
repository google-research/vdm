# Variational Diffusion Models

Jax/Flax Code for implementing Variational Diffusion Models (https://arxiv.org/abs/2107.00630)

## Installation

```pip install -r requirements.txt```

## Training

### (Optional) Setting up a `v3-8` machine on Google Cloud Engine (GCE)

This code was tested on a `v3-8` TPU machine on Google Cloud Engine (GCE). The machine was created using:
```gcloud alpha compute tpus tpu-vm create --project=[project] --zone=[zone] [machine-name] --accelerator-type=v3-8 --version=v2-alpha```

SSH to the instance:
```gcloud alpha compute tpus tpu-vm ssh --project=[project] --zone=[zone] [machine-name]```

Then, we copied (or cloned through git) the code to a directory `~/vdm` on the machine. Then, on the instance, we installed libraries:
```cd ~; bash vdm/sh/setup-vm-tpuv3.sh```

### CIFAR-10 without data augmentation

To train a continuous-time model on CIFAR-10:
```python3 -m vdm.main --config=vdm/configs/cifar10.py --workdir=[workdir]```

Where `[workdir]` can be a local dir or a Google Cloud Storage address (`gs://[your-address]`).

### CIFAR-10 with data augmentation

We also provide code for training a continuous-time model of CIFAR-10 with data augmentation. The model has some minor differences with the one described in the paper, but achieves similar performance.

To train a continuous-time model on CIFAR-10:
```python3 -m vdm.main --config=vdm/configs/cifar10_aug.py --workdir=[workdir]```

However, at the time of writing, this is too slow and memory-intensive to run on a single TPU or GPU machine. Therefore, we run on a `v3-64` TPU pod, as follows.

#### Training on a TPU pod on GCE

Create the machine with:
```gcloud alpha compute tpus tpu-vm create --project=[project] --zone=[zone] [machine-name] --accelerator-type=v3-64 --version=v2-alpha```

Copy (or clone through git) the code to a directory `~/vdm` on the machine. Then, install libraries:
```gcloud alpha compute tpus tpu-vm ssh --project=[project] --zone=[zone] [machine-name] --worker=all --command="$(<~/vdm/sh/setup-tpu-machine-v3.sh)"```

Start training with:
```gcloud alpha compute tpus tpu-vm ssh --project=[project] --zone=[zone] [machine-name] --worker=all --command="~/.local/bin/ipython vdm/main.py -- --workdir=[dir_to_save_logs] --config=vdm/configs/cifar10_aug.py"```

Stop training on all machines by running:
```gcloud alpha compute tpus tpu-vm ssh --project=[project] --zone=[zone] [machine-name] --worker=all --command="killall main.py"```
