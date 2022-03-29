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


import os
import functools
import sys
import time
from typing import Any, Mapping

from absl import logging
from clu import metric_writers
from clu.metric_writers.async_writer import AsyncMultiWriter
from clu.metric_writers.async_writer import AsyncWriter
from clu.metric_writers.logging_writer import LoggingWriter
from clu.metric_writers.summary_writer import SummaryWriter
from clu.metric_writers.interface import MetricWriter
from clu.metric_writers.multi_writer import MultiWriter
import chex
import flax
import jax
import jax.numpy as jnp
import numpy as np
from pprint import pformat


def get_workdir():
  argv = sys.argv
  config_list = []
  config_list.append(time.strftime('%Y%m%d-%H%M%S'))
  for i in range(1, len(argv)):
    if argv[i].startswith('--config='):
      config_file = argv[i].split('/')[-1]
      config_file = config_file.split('.py')[0]
    elif argv[i].startswith('--workdir=') or argv[i].startswith('--config.ckpt_restore_dir='):
      continue
    elif argv[i].startswith('--config'):
      cfgs = argv[i].split('.')
      cfg = cfgs[-1]
      if cfg.isnumeric() or len(cfg) == 0:
        cfg = cfgs[-2]+'.'+cfgs[-1]
      config_list.append(cfg)
  workdir = "-".join(config_list)

  return os.path.join(config_file, workdir)


def dist(fn, accumulate: str, axis_name='batch'):
  """Wrap a function in pmap and device_get(unreplicate(.)) its return value."""

  if accumulate == 'concat':
    accumulate_fn = functools.partial(
        allgather_and_reshape, axis_name=axis_name)
  elif accumulate == 'mean':
    accumulate_fn = functools.partial(
        jax.lax.pmean, axis_name=axis_name)
  elif accumulate == 'none':
    accumulate_fn = None
  else:
    raise NotImplementedError(accumulate)

  @functools.partial(jax.pmap, axis_name=axis_name)
  def pmapped_fn(*args, **kwargs):
    out = fn(*args, **kwargs)
    return out if accumulate_fn is None else jax.tree_map(accumulate_fn, out)

  def wrapper(*args, **kwargs):
    return jax.device_get(
        flax.jax_utils.unreplicate(pmapped_fn(*args, **kwargs)))

  return wrapper


def allgather_and_reshape(x, axis_name='batch'):
  """Allgather and merge the newly inserted axis w/ the original batch axis."""
  y = jax.lax.all_gather(x, axis_name=axis_name)
  assert y.shape[1:] == x.shape
  return y.reshape(y.shape[0] * x.shape[0], *x.shape[1:])


def generate_image_grids(images: chex.Array):
  """Simple helper to generate a single image from a mini batch."""

  def image_grid(nrow, ncol, imagevecs, imshape):
    images = iter(imagevecs.reshape((-1,) + imshape))
    return jnp.squeeze(
        jnp.vstack([
            jnp.hstack([next(images)
                        for _ in range(ncol)][::-1])
            for _ in range(nrow)
        ]))

  batch_size = images.shape[0]
  grid_size = int(np.floor(np.sqrt(batch_size)))

  image_shape = images.shape[1:]
  return image_grid(
      nrow=grid_size,
      ncol=grid_size,
      imagevecs=images[0:grid_size**2],
      imshape=image_shape,
  )


def clip_by_global_norm(pytree, clip_norm, use_norm=None):
  if use_norm is None:
    use_norm = global_norm(pytree)
    assert use_norm.shape == ()  # pylint: disable=g-explicit-bool-comparison
  scale = clip_norm * jnp.minimum(1.0 / use_norm, 1.0 / clip_norm)
  return jax.tree_map(lambda x: x * scale, pytree), use_norm


def global_norm(pytree):
  return jnp.sqrt(jnp.sum(jnp.asarray(
      [jnp.sum(jnp.square(x)) for x in jax.tree_leaves(pytree)])))


def apply_ema(decay, avg, new):
  return jax.tree_multimap(lambda a, b: decay * a + (1. - decay) * b, avg, new)


""" Get metrics """


def get_metrics(device_metrics):
  # We select the first element of x in order to get a single copy of a
  # device-replicated metric.
  _device_metrics = jax.tree_map(lambda x: x[0], device_metrics)
  metrics_np = jax.device_get(_device_metrics)
  return stack_forest(metrics_np)


def stack_forest(forest):
  stack_args = lambda *args: np.stack(args)
  return jax.tree_multimap(stack_args, *forest)


def average_appended_metrics(metrics):
  ks = metrics[0].keys()
  result = {k: np.mean([metrics[i][k]
                       for i in range(len(metrics))]) for k in ks}
  return result


""" Custom logging that is command-line friendly """


def create_custom_writer(logdir: str, process_index: int,
                         asynchronous=True) -> MetricWriter:
  """Adapted from clu.metric_writers.__init__"""
  if process_index > 0:
    if asynchronous:
      return AsyncWriter(CustomLoggingWriter())
    else:
      return CustomLoggingWriter()
  writers = [CustomLoggingWriter(), SummaryWriter(logdir)]
  if asynchronous:
    return AsyncMultiWriter(writers)
  return MultiWriter(writers)


class CustomLoggingWriter(LoggingWriter):
  def write_scalars(self, step: int, scalars: Mapping[str, metric_writers.interface.Scalar]):
    keys = sorted(scalars.keys())
    if step == 1:
      logging.info("%s", ", ".join(["Step"]+keys))
    values = [scalars[key] for key in keys]
    # Convert jax DeviceArrays and numpy ndarrays to python native type
    values = [np.array(v).item() for v in values]
    # Print floats
    values = [f"{v:.4f}" if isinstance(v, float) else f"{v}" for v in values]
    logging.info("%d, %s", step, ", ".join(values))

  def write_texts(self, step: int, texts: Mapping[str, str]):
    logging.info("[%d] Got texts: %s.", step, texts)

  def write_hparams(self, hparams: Mapping[str, Any]):
    logging.info("Hyperparameters:\n%s", pformat(hparams))

  def write_images(self, step: int, images: Mapping[str, Any]):
    logging.info("[%d] Got images: %s.", step,
                 {k: v.shape for k, v in images.items()})


""" Run with temporary verbosity """


def with_verbosity(temporary_verbosity_level, fn):
  old_verbosity_level = logging.get_verbosity()
  logging.set_verbosity(temporary_verbosity_level)
  result = fn()
  logging.set_verbosity(old_verbosity_level)
  return result
