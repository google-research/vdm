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

from abc import ABC, abstractmethod
import functools
import os
from typing import Any, Tuple

from absl import logging
import chex
from clu import periodic_actions
from clu import parameter_overview
from clu import metric_writers
from clu import checkpoint
from flax.core.frozen_dict import unfreeze, FrozenDict
import flax.jax_utils as flax_utils
import flax
from jax._src.random import PRNGKey
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from optax._src import base
import tensorflow as tf

import vdm.train_state
import vdm.utils as utils
import vdm.dataset as dataset


class Experiment(ABC):
  """Boilerplate for training and evaluating VDM models."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.config = config

    # Set seed before initializing model.
    seed = config.training.seed
    self.rng = utils.with_verbosity("ERROR", lambda: jax.random.PRNGKey(seed))

    # initialize dataset
    logging.warning('=== Initializing dataset ===')
    self.rng, data_rng = jax.random.split(self.rng)
    self.train_iter, self.eval_iter = dataset.create_dataset(config, data_rng)

    # initialize model
    logging.warning('=== Initializing model ===')
    self.rng, model_rng = jax.random.split(self.rng)
    self.model, params = self.get_model_and_params(model_rng)
    parameter_overview.log_parameter_overview(params)

    # initialize train state
    logging.info('=== Initializing train state ===')
    self.state = vdm.train_state.TrainState.create(
        apply_fn=self.model.apply,
        variables=params,
        optax_optimizer=self.get_optimizer)
    self.lr_schedule = self.get_lr_schedule()

    # Restore from checkpoint
    ckpt_restore_dir = self.config.get('ckpt_restore_dir', 'None')
    if ckpt_restore_dir != 'None':
      ckpt_restore = checkpoint.Checkpoint(ckpt_restore_dir)
      checkpoint_to_restore = ckpt_restore.get_latest_checkpoint_to_restore_from()
      assert checkpoint_to_restore
      state_restore_dict = ckpt_restore.restore_dict(checkpoint_to_restore)
      self.state = restore_partial(self.state, state_restore_dict)
      del state_restore_dict, ckpt_restore, checkpoint_to_restore

    # initialize train/eval step
    logging.info('=== Initializing train/eval step ===')
    self.rng, train_rng = jax.random.split(self.rng)
    self.p_train_step = functools.partial(self.train_step, train_rng)
    self.p_train_step = functools.partial(jax.lax.scan, self.p_train_step)
    self.p_train_step = jax.pmap(self.p_train_step, "batch")

    self.rng, eval_rng, sample_rng = jax.random.split(self.rng, 3)
    self.p_eval_step = functools.partial(self.eval_step, eval_rng)
    self.p_eval_step = jax.pmap(self.p_eval_step, "batch")
    self.p_sample = functools.partial(
        self.sample_fn,
        dummy_inputs=next(self.eval_iter)["images"][0],
        rng=sample_rng,
    )
    self.p_sample = utils.dist(
        self.p_sample, accumulate='concat', axis_name='batch')

    logging.info('=== Done with Experiment.__init__ ===')

  def get_lr_schedule(self):
    learning_rate = self.config.optimizer.learning_rate
    config_train = self.config.training
    # Create learning rate schedule
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=learning_rate,
        transition_steps=config_train.num_steps_lr_warmup
    )

    if self.config.optimizer.lr_decay:
      decay_fn = optax.linear_schedule(
          init_value=learning_rate,
          end_value=0,
          transition_steps=config_train.num_steps_train - config_train.num_steps_lr_warmup,
      )
      schedule_fn = optax.join_schedules(
          schedules=[warmup_fn, decay_fn], boundaries=[
              config_train.num_steps_lr_warmup]
      )
    else:
      schedule_fn = warmup_fn

    return schedule_fn

  def get_optimizer(self, lr: float) -> base.GradientTransformation:
    """Get an optax optimizer. Can be overided. """
    config = self.config.optimizer

    def decay_mask_fn(params):
      flat_params = flax.traverse_util.flatten_dict(unfreeze(params))
      flat_mask = {
          path: (path[-1] != "bias" and path[-2:]
                 not in [("layer_norm", "scale"), ("final_layer_norm", "scale")])
          for path in flat_params
      }
      return FrozenDict(flax.traverse_util.unflatten_dict(flat_mask))

    if config.name == "adamw":
      optimizer = optax.adamw(
          learning_rate=lr,
          mask=decay_mask_fn,
          **config.args,
      )
      if hasattr(config, "gradient_clip_norm"):
        clip = optax.clip_by_global_norm(config.gradient_clip_norm)
        optimizer = optax.chain(clip, optimizer)
    else:
      raise Exception('Unknow optimizer.')

    return optimizer

  @abstractmethod
  def get_model_and_params(self, rng: PRNGKey):
    """Return the model and initialized parameters."""
    ...

  @abstractmethod
  def sample_fn(self, *, dummy_inputs, rng, params) -> chex.Array:
    """Generate a batch of samples in [0, 255]. """
    ...

  @abstractmethod
  def loss_fn(self, params, batch, rng, is_train) -> Tuple[float, Any]:
    """Loss function and metrics."""
    ...

  def train_and_evaluate(self, workdir: str):
    logging.warning('=== Experiment.train_and_evaluate() ===')
    logging.info('Workdir: '+workdir)

    #if jax.process_index() == 0:
    #  if not tf.io.gfile.exists(workdir):
    #    tf.io.gfile.mkdir(workdir)

    config = self.config.training
    logging.info('num_steps_train=%d', config.num_steps_train)

    # Get train state
    state = self.state

    # Set up checkpointing of the model and the input pipeline.
    checkpoint_dir = os.path.join(workdir, 'checkpoints')
    ckpt = checkpoint.MultihostCheckpoint(checkpoint_dir, max_to_keep=5)
    checkpoint_to_restore = ckpt.get_latest_checkpoint_to_restore_from()
    if checkpoint_to_restore:
      state = ckpt.restore_or_initialize(state, checkpoint_to_restore)
    initial_step = int(state.step)

    # Distribute training.
    state = flax_utils.replicate(state)

    # Create logger/writer
    writer = utils.create_custom_writer(workdir, jax.process_index())
    if initial_step == 0:
      writer.write_hparams(dict(self.config))

    hooks = []
    report_progress = periodic_actions.ReportProgress(
        num_train_steps=config.num_steps_train, writer=writer)
    if jax.process_index() == 0:
      hooks += [report_progress]
      if config.profile:
        hooks += [periodic_actions.Profile(num_profile_steps=5,
                                           logdir=workdir)]

    step = initial_step
    substeps = config.substeps

    with metric_writers.ensure_flushes(writer):
      logging.info('=== Start of training ===')
      # the step count starts from 1 to num_steps_train
      while step < config.num_steps_train:
        is_last_step = step + substeps >= config.num_steps_train
        # One training step
        with jax.profiler.StepTraceAnnotation('train', step_num=step):
          batch = jax.tree_map(jnp.asarray, next(self.train_iter))
          state, _train_metrics = self.p_train_step(state, batch)

        # Quick indication that training is happening.
        logging.log_first_n(
            logging.WARNING, 'Finished training step %d.', 3, step)
        for h in hooks:
          h(step)

        new_step = int(state.step[0])
        assert new_step == step + substeps
        step = new_step

        if step % config.steps_per_logging == 0 or is_last_step:
          logging.debug('=== Writing scalars ===')
          metrics = flax_utils.unreplicate(_train_metrics['scalars'])

          def avg_over_substeps(x):
            assert x.shape[0] == substeps
            return float(x.mean(axis=0))

          metrics = jax.tree_map(avg_over_substeps, metrics)
          writer.write_scalars(step, metrics)

        if step % config.steps_per_eval == 0 or is_last_step or step == 1000:
          logging.debug('=== Running eval ===')
          with report_progress.timed('eval'):
            eval_metrics = []
            for eval_step in range(config.num_steps_eval):
              batch = self.eval_iter.next()
              batch = jax.tree_map(jnp.asarray, batch)
              metrics = self.p_eval_step(
                  state.ema_params, batch, flax_utils.replicate(eval_step))
              eval_metrics.append(metrics['scalars'])

            # average over eval metrics
            eval_metrics = utils.get_metrics(eval_metrics)
            eval_metrics = jax.tree_map(jnp.mean, eval_metrics)
            writer.write_scalars(step, eval_metrics)

            # print out a batch of images
            metrics = flax_utils.unreplicate(metrics)
            images = metrics['images']
            samples = self.p_sample(params=state.ema_params)
            samples = utils.generate_image_grids(samples)[None, :, :, :]
            images['samples'] = samples.astype(np.uint8)
            writer.write_images(step, images)

        if step % config.steps_per_save == 0 or is_last_step:
          with report_progress.timed('checkpoint'):
            ckpt.save(flax_utils.unreplicate(state))

  def evaluate(self, logdir, checkpoint_dir):
    """Perform one evaluation."""
    logging.info('=== Experiment.evaluate() ===')

    ckpt = checkpoint.Checkpoint(checkpoint_dir)
    state_dict = ckpt.restore_dict()
    params = flax.core.FrozenDict(state_dict['ema_params'])
    step = int(state_dict['step'])

    # Distribute training.
    params = flax_utils.replicate(params)

    eval_logdir = os.path.join(logdir, 'eval')
    tf.io.gfile.makedirs(eval_logdir)
    writer = metric_writers.create_default_writer(
        eval_logdir, just_logging=jax.process_index() > 0)

    eval_metrics = []

    for eval_step in range(self.config.training.num_steps_eval):
      batch = self.eval_iter.next()
      batch = jax.tree_map(jnp.asarray, batch)
      metrics = self.p_eval_step(
          params, batch, flax_utils.replicate(eval_step))
      eval_metrics.append(metrics['scalars'])

    # average over eval metrics
    eval_metrics = utils.get_metrics(eval_metrics)
    eval_metrics = jax.tree_map(jnp.mean, eval_metrics)

    writer.write_scalars(step, eval_metrics)

    # sample a batch of images
    samples = self.p_sample(params=params)
    samples = utils.generate_image_grids(samples)[None, :, :, :]
    samples = {'samples': samples.astype(np.uint8)}
    writer.write_images(step, samples)

  def train_step(self, base_rng, state, batch):
    rng = jax.random.fold_in(base_rng, jax.lax.axis_index('batch'))
    rng = jax.random.fold_in(rng, state.step)

    grad_fn = jax.value_and_grad(self.loss_fn, has_aux=True)
    (_, metrics), grads = grad_fn(state.params, batch, rng=rng, is_train=True)
    grads = jax.lax.pmean(grads, "batch")

    learning_rate = self.lr_schedule(state.step)
    new_state = state.apply_gradients(
        grads=grads, lr=learning_rate, ema_rate=self.config.optimizer.ema_rate)

    metrics['scalars'] = jax.tree_map(
        lambda x: jax.lax.pmean(x, axis_name="batch"), metrics['scalars'])
    metrics['scalars'] = {"train_" +
                          k: v for (k, v) in metrics['scalars'].items()}

    metrics['images'] = jax.tree_map(
        lambda x: utils.generate_image_grids(x)[None, :, :, :],
        metrics['images'])

    return new_state, metrics

  def eval_step(self, base_rng, params, batch, eval_step=0):
    rng = jax.random.fold_in(base_rng, jax.lax.axis_index('batch'))
    rng = jax.random.fold_in(rng, eval_step)

    _, metrics = self.loss_fn(params, batch, rng=rng, is_train=False)

    # summarize metrics
    metrics['scalars'] = jax.lax.pmean(
        metrics['scalars'], axis_name="batch")
    metrics['scalars'] = {
        "eval_" + k: v for (k, v) in metrics['scalars'].items()}

    metrics['images'] = jax.tree_map(
        lambda x: utils.generate_image_grids(x)[None, :, :, :],
        metrics['images'])

    return metrics


def copy_dict(dict1, dict2):
  if not isinstance(dict1, dict):
    assert not isinstance(dict2, dict)
    return dict2
  for key in dict1.keys():
    if key in dict2:
      dict1[key] = copy_dict(dict1[key], dict2[key])

  return dict1


def restore_partial(state, state_restore_dict):
  state_dict = flax.serialization.to_state_dict(state)
  state_dict = copy_dict(state_dict, state_restore_dict)
  state = flax.serialization.from_state_dict(state, state_dict)

  return state
