# Copyright 2022 The VDM Authors, Flax Authors.
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

"""
Adapted from
https://flax.readthedocs.io/en/latest/_modules/flax/
training/train_state.html#TrainState.

But with added EMA of the parameters.
"""

import copy
from typing import Any, Callable, Optional

from flax import core
from flax import struct
import jax
import optax


class TrainState(struct.PyTreeNode):
  """Simple train state for the common case with a single Optax optimizer.

  Synopsis:

    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx)
    grad_fn = jax.grad(make_loss_fn(state.apply_fn))
    for batch in data:
      grads = grad_fn(state.params, batch)
      state = state.apply_gradients(grads=grads)

  Note that you can easily extend this dataclass by subclassing it for storing
  additional data (e.g. additional variable collections).

  For more exotic usecases (e.g. multiple optimizers) it's probably best to
  fork the class and modify it.

  Attributes:
    step: Counter starts at 0 and is incremented by every call to
      `.apply_gradients()`.
    apply_fn: Usually set to `model.apply()`. Kept in this dataclass for
      convenience to have a shorter params list for the `train_step()` function
      in your training loop.
    tx: An Optax gradient transformation.
    opt_state: The state for `tx`.
  """
  step: int
  params: core.FrozenDict[str, Any]
  ema_params: core.FrozenDict[str, Any]
  opt_state: optax.OptState
  tx_fn: Callable[[float], optax.GradientTransformation] = struct.field(
      pytree_node=False)
  apply_fn: Callable = struct.field(pytree_node=False)

  def apply_gradients(self, *, grads, lr, ema_rate, **kwargs):
    """Updates `step`, `params`, `opt_state` and `**kwargs` in return value.

    Note that internally this function calls `.tx.update()` followed by a call
    to `optax.apply_updates()` to update `params` and `opt_state`.

    Args:
      grads: Gradients that have the same pytree structure as `.params`.
      **kwargs: Additional dataclass attributes that should be `.replace()`-ed.

    Returns:
      An updated instance of `self` with `step` incremented by one, `params`
      and `opt_state` updated by applying `grads`, and additional attributes
      replaced as specified by `kwargs`.
    """
    tx = self.tx_fn(lr)
    updates, new_opt_state = tx.update(
        grads, self.opt_state, self.params)
    new_params = optax.apply_updates(self.params, updates)
    new_ema_params = jax.tree_multimap(
        lambda x, y: x + (1. - ema_rate) * (y - x),
        self.ema_params,
        new_params,
    )

    return self.replace(
        step=self.step + 1,
        params=new_params,
        ema_params=new_ema_params,
        opt_state=new_opt_state,
        **kwargs,
    )

  @classmethod
  def create(_class, *, apply_fn, variables, optax_optimizer, **kwargs):
    """Creates a new instance with `step=0` and initialized `opt_state`."""
    # _class is the TrainState class
    params = variables["params"]
    opt_state = optax_optimizer(1.).init(params)
    ema_params = copy.deepcopy(params)
    return _class(
        step=0,
        apply_fn=apply_fn,
        params=params,
        ema_params=ema_params,
        tx_fn=optax_optimizer,
        opt_state=opt_state,
        **kwargs,
    )
