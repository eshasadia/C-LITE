# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
from functools import partial
from typing import Any

import dinov2.distributed as distributed
import torch
from fvcore.common.checkpoint import Checkpointer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp._runtime_utils import _reshard
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

logger = logging.getLogger("dinov2")


def get_fsdp_wrapper(model_cfg, modules_to_wrap=set()):
    sharding_strategy_dict = {
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
    }

    dtype_dict = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }

    mixed_precision_config = MixedPrecision(
        param_dtype=dtype_dict[model_cfg.mixed_precision.param_dtype],
        reduce_dtype=dtype_dict[model_cfg.mixed_precision.reduce_dtype],
        buffer_dtype=dtype_dict[model_cfg.mixed_precision.buffer_dtype],
    )

    sharding_strategy_config = sharding_strategy_dict[model_cfg.sharding_strategy]

    local_rank = distributed.get_local_rank()

    fsdp_wrapper = partial(
        FSDP,
        sharding_strategy=sharding_strategy_config,
        mixed_precision=mixed_precision_config,
        device_id=local_rank,
        sync_module_states=True,
        use_orig_params=True,
        auto_wrap_policy=ModuleWrapPolicy(modules_to_wrap),
    )
    return fsdp_wrapper


def is_fsdp(x):
    return isinstance(x, FSDP)


def is_sharded_fsdp(x):
    return is_fsdp(x) and x.sharding_strategy is not ShardingStrategy.NO_SHARD



_reshard_unavailable_warned = False


def _warn_reshard_unavailable(detail: str):
    global _reshard_unavailable_warned
    if _reshard_unavailable_warned:
        return
    logger.warning(
        "reshard_fsdp_model: could not perform the manual early-reshard "
        "optimization on this PyTorch version (%s). This does not affect "
        "training correctness -- FSDP reshards parameters automatically "
        "after each forward pass regardless -- it only means gathered "
        "parameters may be freed slightly later than intended, using "
        "marginally more peak GPU memory. This warning is only logged "
        "once.",
        detail,
    )
    _reshard_unavailable_warned = True


def _get_fsdp_handles(x):
    """
    Returns a list of FlatParamHandle-like objects for `x`, trying every
    attribute name FSDP has used across PyTorch versions. Returns an empty
    list if none are found.
    """
    handles = getattr(x, "_handles", None)
    if handles:
        return list(handles)

    handle = getattr(x, "_handle", None)
    if handle is not None:
        return [handle]

    return []


def free_if_fsdp(x):
    if not is_sharded_fsdp(x):
        return

    handles = _get_fsdp_handles(x)
    if not handles:
        _warn_reshard_unavailable(
            "no `_handles` or `_handle` attribute found on the FSDP-wrapped module"
        )
        return

    true_list = [True for _ in handles]

    # Try the calling convention `_reshard` used pre-2.1 (list of handles,
    # list of free-flags) first, then fall back to the singular-handle form
    # some later versions expect.
    try:
        _reshard(x, handles, true_list)
        return
    except (TypeError, AttributeError):
        pass

    try:
        _reshard(x, handles[0], true_list[0])
        return
    except (TypeError, AttributeError, IndexError) as e:
        _warn_reshard_unavailable(f"`_reshard()` call failed with: {e!r}")
        return


def get_fsdp_modules(x):
    return FSDP.fsdp_modules(x)


def reshard_fsdp_model(x):
    for m in get_fsdp_modules(x):
        free_if_fsdp(m)


def rankstr():
    return f"rank_{distributed.get_global_rank()}"


class FSDPCheckpointer(Checkpointer):
    
    def save(self, name: str, **kwargs: Any) -> None:
        """
        Dump model and checkpointables to a file.

        Args:
            name (str): name of the file.
            kwargs (dict): extra arbitrary data to save.
        """
        if not self.save_dir or not self.save_to_disk:
            return

        data = {}
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):  # LOCAL_STATE_DICT
            data["model"] = self.model.state_dict()

        # data["model"] = self.model.state_dict()
        for key, obj in self.checkpointables.items():
            data[key] = obj.state_dict()
        data.update(kwargs)

        basename = f"{name}.{rankstr()}.pth"
        save_file = os.path.join(self.save_dir, basename)
        assert os.path.basename(save_file) == basename, basename
        self.logger.info("Saving checkpoint to {}".format(save_file))
        with self.path_manager.open(save_file, "wb") as f:
            torch.save(data, f)
        self.tag_last_checkpoint(basename)

    def load(self, *args, **kwargs):
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):  # LOCAL_STATE_DICT
            return super().load(*args, **kwargs)

    def has_checkpoint(self) -> bool:
        """
        Returns:
            bool: whether a checkpoint exists in the target directory.
        """
        save_file = os.path.join(self.save_dir, f"last_checkpoint.{rankstr()}")
        return self.path_manager.exists(save_file)

    def get_checkpoint_file(self) -> str:
        """
        Returns:
            str: The latest checkpoint file in target directory.
        """
        save_file = os.path.join(self.save_dir, f"last_checkpoint.{rankstr()}")
        try:
            with self.path_manager.open(save_file, "r") as f:
                last_saved = f.read().strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            return ""
        # pyre-fixme[6]: For 2nd param expected `Union[PathLike[str], str]` but got
        #  `Union[bytes, str]`.
        return os.path.join(self.save_dir, last_saved)

    def tag_last_checkpoint(self, last_filename_basename: str) -> None:
        """
        Tag the last checkpoint.

        Args:
            last_filename_basename (str): the basename of the last filename.
        """
        if distributed.is_enabled():
            torch.distributed.barrier()
        save_file = os.path.join(self.save_dir, f"last_checkpoint.{rankstr()}")
        with self.path_manager.open(save_file, "w") as f:
            f.write(last_filename_basename)  # pyre-ignore


ShardedGradScaler = ShardedGradScaler