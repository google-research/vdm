# Copyright 2022 The VDM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Get configuration
import ml_collections


def d(**kwargs):
  """Helper of creating a config dict."""
  return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
  """Get the hyperparameters for the model"""
  config = ml_collections.ConfigDict()
  config.exp_name = "exp_vdm"
  config.model_type = "model_vdm"
  config.ckpt_restore_dir = 'None'

  config.data = d(
      dataset='imagenet64',  # cifar10/cifar10_aug/imagenet32/imagenet64
      ignore_cache=False,
  )

  config.model = d(
      vocab_size=256,
      sample_softmax=False,
      antithetic_time_sampling=True,
      with_fourier_features=True,
      with_attention=False,

      # configurations of the noise schedule
      gamma_type='learnable_scalar',  # learnable_scalar / learnable_nnet / fixed
      gamma_min=-13.3,
      gamma_max=5.,

      # configurations of the score model
      sm_n_timesteps=0,
      sm_n_embd=256,
      sm_n_layer=64,
      sm_pdrop=0.1,
  )

  config.training = d(
      seed=1,
      substeps=1000,
      num_steps_lr_warmup=100,
      num_steps_train=10_000_000,
      num_steps_eval=100,
      batch_size_train=128,
      batch_size_eval=128,
      steps_per_logging=1000,
      steps_per_eval=10_000,
      steps_per_save=10_000,
      profile=False,
  )

  config.optimizer = d(
      name='adamw',
      args=d(
          b1=0.9,
          b2=0.99,
          eps=1e-8,
          weight_decay=0.01,
      ),
      learning_rate=2e-4,
      lr_decay=False,
      ema_rate=0.9999,
  )

  return config
