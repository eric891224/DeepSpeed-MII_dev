# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# PASLAB
import torch

# DeepSpeed Team
from functools import wraps

from deepspeed.accelerator import get_accelerator

from mii.logging import logger


def sync_debug(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.sync_debug:
            get_accelerator().synchronize()
            logger.debug(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
        result = func(self, *args, **kwargs)
        if self.sync_debug:
            get_accelerator().synchronize()
            logger.debug(f"Finished calling {func.__name__}")
        return result

    return wrapper


def profiler(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.profile_model_time:
            return func(self, *args, **kwargs)

        # unknown bubble
        if self._iters == 0:
            return func(self, *args, **kwargs)

        # evaluation
        elif self._iters == 1:
            stage = "evaluate"
            torch.cuda.nvtx.range_push("Evaluation")
        # generation
        else:
            stage = "generate"
            torch.cuda.nvtx.range_push("Generation")

        self._timers(stage).start()
        # self._timers(func.__name__).start()
        result = func(self, *args, **kwargs)
        # self._timers(func.__name__).stop()
        self._timers(stage).stop()
        torch.cuda.nvtx.range_pop()

        self._profiled_times[stage].append(self._timers(stage).elapsed(reset=True))

        # self._profiled_times[func.__name__].append(
        #     self._timers(func.__name__).elapsed(reset=True)
        # )
        return result

    return wrapper
