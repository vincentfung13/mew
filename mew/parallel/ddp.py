import logging
import weakref
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.autograd import Variable

LOGGER = logging.getLogger(__name__)


class DistributedDataParallel(nn.Module):
    def __init__(
        self,
        module: torch.nn.Module,
    ):
        super().__init__()

        self.module = module

        # This flag is used to control whether we trigger the grad all_reduce,
        # in the case where grad_accumulation is enabled, we only need to sync
        # on the step where there's an optim step
        self.sync_grad_flag = True

        LOGGER.info("Broadcasting weights to all workers...")
        with torch.no_grad():
            _broadcast_handles = []
            for parameter in module.parameters():
                # Broadcast the model weights to all ranks from rank 0
                _handle = dist.broadcast(parameter, src=0, async_op=True)
                _broadcast_handles.append(_handle)
            LOGGER.info("Waiting for the broadcast ops to finish...")
            for handle in _broadcast_handles:
                handle.wait()

        # Using weakref here to handle the hooks, so that gc can remove the hooks
        # when the DDP ref is gc-ed
        # This is not needed in normal training runs but needs to be taken care of in the
        # scenarios where the model might outlive the DDP wrapper (e.g profiling)
        self_ref = weakref.ref(self)

        def _on_grad_ready_hook(param):
            watcher = self_ref()
            if watcher is None:
                return
            watcher._on_grad_ready(param)

        # Register all-reduce hooks
        LOGGER.info("Registering all reduce hooks for all grads...")
        self._grad_all_reduce_handles = []
        self._hook_handles = []
        for parameter in module.parameters():
            if parameter.requires_grad:
                _handle = parameter.register_post_accumulate_grad_hook(
                    _on_grad_ready_hook
                )
                self._hook_handles.append(_handle)

        self._callback_queued = False

    def forward(self, *args, **kwargs):
        self._callback_queued = False
        return self.module(*args, **kwargs)

    def __del__(self):
        _hook_handles = getattr(self, "_hook_handles", [])
        for _handle in _hook_handles:
            _handle.remove()

    def _sync_grads(self):
        # This function waits for all grads to finish all reducing.
        # it is queued into the autograd callback queue
        # and automatically called by the autograd engine before loss.backward returns
        # this happens in line ~96
        for handle in self._grad_all_reduce_handles:
            handle.wait()
        # clear handles buffer
        self._grad_all_reduce_handles = []
        self._callback_queued = False

    @contextmanager
    def no_sync(self):
        previous = self.sync_grad_flag
        self.sync_grad_flag = False
        try:
            yield
        finally:
            self.sync_grad_flag = previous

    def _on_grad_ready(self, parameter: nn.parameter.Parameter):
        if not self.sync_grad_flag:
            return

        # Queue sync grads call back ONLY IF sync_grad_flag is set
        if not self._callback_queued:
            Variable._execution_engine.queue_callback(self._sync_grads)
            self._callback_queued = True

        # All reduce the gradient
        backend = dist.get_backend()
        if backend == dist.Backend.NCCL:
            _handle = dist.all_reduce(
                parameter.grad, op=dist.ReduceOp.AVG, async_op=True
            )
        else:
            parameter.grad /= dist.get_world_size()
            _handle = dist.all_reduce(
                parameter.grad, op=dist.ReduceOp.SUM, async_op=True
            )
        self._grad_all_reduce_handles.append(_handle)
