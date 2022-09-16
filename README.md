# Variational Diffusion Models

Jax/Flax Code for reproducing some key results of Variational Diffusion Models (https://arxiv.org/abs/2107.00630).

## Standalone Colabs

At `colab/SimpleDiffusionColab.ipynb` you will find an independent and stand-alone Colab implementation of a Variational Diffusion Model (VDM), serving as an easy-to-understand demonstration of the code and principles behind the paper. [Link to open in Colab](https://colab.research.google.com/github/google-research/vdm/blob/main/colab/SimpleDiffusionColab.ipynb). (Thanks a lot to [Alex Alemi](https://www.alexalemi.com/) and [Ben Poole](https://cs.stanford.edu/~poole/) for this implementation.)

At `colab/2D_VDM_Example.ipynb` you will find an even more basic implementation, on a 2D swirl dataset and using MLPs. [Link to open in Colab](https://colab.research.google.com/github/google-research/vdm/blob/main/colab/2D_VDM_Example.ipynb).

## Setup: Installing required libraries

This code was tested on a TPU-v3 machine. For instructions on how to launch such a machine, see 'Setting up a `v3-8` machine' below.

To install the required libraries on a TPU machine:
```
pip3 install -U pip
sudo pip uninstall -y jax jaxlib
pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip3 install --upgrade -r requirements.txt
```

Alternatively, on a GPU machine, this should work:
```
pip3 install -U pip
pip3 install --upgrade jax jaxlib
pip3 install --upgrade -r requirements.txt
```

## Train/evaluate: CIFAR-10 without data augmentation

The commands below assume that the code is checked out at the `./vdm` directory, such that this README is located at `./vdm/README'.

To evaluate from a pre-trained checkpoint:
```
python3 -m vdm.main --mode=eval --config=vdm/configs/cifar10.py --workdir=[workdir] --checkpoint='gs://gresearch/vdm/cifar10/checkpoint-final/checkpoints-0'
```
where `[workdir]` is a directory to write results to, such as `'/tmp/vdm-workdir'`, or a Google Cloud Storage address (`gs://[your-address]`). Running the command above will print out a bunch of statistics, including `eval_bpd=2.637`, which matches the result in the paper (2.64).

To train:
```
python3 -m vdm.main --config=vdm/configs/cifar10.py --workdir=[workdir]
```

## CIFAR-10 with data augmentation

We also provide code for training a continuous-time VDM of CIFAR-10 with data augmentation. The model has some minor differences with the one described in the paper, but achieves similar performance.

To evaluate:
```
python3 -m vdm.main --mode=eval --config=vdm/configs/cifar10_aug.py --workdir=[workdir] --checkpoint='gs://gresearch/vdm/cifar10/aug-checkpoint-final/checkpoints-0'
```
This reports a bpd results slightly worse than the paper: eval_bpd=2.522, versus the paper's result of 2.49. We suspect this is due to small the fact that this open source version of the model was trained on a smaller batch size, in addition to a small difference in the score network implementation, namely how it is conditioned.

To train:
```
python3 -m vdm.main --config=vdm/configs/cifar10_aug.py --workdir=[workdir]
```

At the time of writing, training this model is too slow and memory-intensive to run on a single TPU or GPU machine. Therefore, we run on a `v3-64` TPU pod, as explained below.

## (Optional) Setting up a `v3-8` machine on Google Cloud Engine (GCE)

This code was tested on a `v3-8` TPU machine on Google Cloud Engine (GCE). The machine was created using:
```
gcloud alpha compute tpus tpu-vm create --project=[project] --zone=[zone] [machine-name] --accelerator-type=v3-8 --version=v2-alpha
```

SSH to the instance:
```
gcloud alpha compute tpus tpu-vm ssh --project=[project] --zone=[zone] [machine-name]
```

Then, we copied (or cloned through git) the code to a directory `~/vdm` on the machine. Then, on the instance, we installed libraries:
```
cd ~; bash vdm/sh/setup-vm-tpuv3.sh
```

## Training on a TPU pod on GCE

Create the machine with:
```
gcloud alpha compute tpus tpu-vm create --project=[project] --zone=[zone] [machine-name] --accelerator-type=v3-64 --version=v2-alpha
```

Copy (or clone through git) the code to a directory `~/vdm` on the machine. Then, install libraries:
```
gcloud alpha compute tpus tpu-vm ssh --project=[project] --zone=[zone] [machine-name] --worker=all --command="$(<~/vdm/sh/setup-tpu-machine-v3.sh)"
```

Start training with:
```
gcloud alpha compute tpus tpu-vm ssh --project=[project] --zone=[zone] [machine-name] --worker=all --command="~/.local/bin/ipython vdm/main.py -- --workdir=[dir_to_save_logs] --config=vdm/configs/cifar10_aug.py"
```

Stop training on all machines by running:
```
gcloud alpha compute tpus tpu-vm ssh --project=[project] --zone=[zone] [machine-name] --worker=all --command="killall main.py"
```

## Acknowledgments

We thank [Ruiqi Gao](https://ruiqigao.github.io/) for substantial contributions to this public VDM codebase, and to [Alex Alemi](https://www.alexalemi.com/) and [Ben Poole](https://cs.stanford.edu/~poole/) for implementing `colab/SimpleDiffusionColab.ipynb`.

## Disclaimer

This is not an officially supported Google product.
