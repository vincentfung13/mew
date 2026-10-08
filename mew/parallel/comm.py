from typing import Iterable

import torch
import torch.distributed as dist


def broadcast_tensors(tensors: Iterable, sync: bool):
    with torch.no_grad():
        _broadcast_handles = []
        for _to_broadcast in tensors:
            # Broadcast the model weights to all ranks from rank 0
            _handle = dist.broadcast(_to_broadcast, src=0, async_op=True)
            _broadcast_handles.append(_handle)
        if sync:
            for handle in _broadcast_handles:
                handle.wait()
