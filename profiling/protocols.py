from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from enum import Enum

import torch

from profiling.cases import ProfilingCase


class ProfilingProtocol(str, Enum):
    FORWARD_ONLY = "forward_only"
    FULL_TRAINING_STEP = "full_training_step"
    REPEAT_BACKWARD_ON_SAME_GRAPH = "repeat_backward_on_same_graph"


Stage = Callable[[str], AbstractContextManager]


def parse_protocol(value: str) -> ProfilingProtocol:
    try:
        return ProfilingProtocol(value)
    except ValueError as error:
        supported = ", ".join(protocol.value for protocol in ProfilingProtocol)
        raise ValueError(
            f"Unsupported profiling protocol: {value}. Supported protocols: {supported}"
        ) from error


def run_protocol(
    protocol: ProfilingProtocol,
    *,
    steps: int,
    module: torch.nn.Module,
    case: ProfilingCase,
    optimizer: torch.optim.Optimizer,
    stage: Stage = lambda _name: nullcontext(),
) -> None:
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")

    if protocol is ProfilingProtocol.FORWARD_ONLY:
        for _ in range(steps):
            with stage("total"):
                with torch.no_grad(), stage("forward"):
                    module(*case.inputs)
        return

    if protocol is ProfilingProtocol.FULL_TRAINING_STEP:
        for _ in range(steps):
            with stage("total"):
                optimizer.zero_grad()
                with stage("forward"):
                    output = module(*case.inputs)
                    loss = case.loss_fn(output)
                with stage("backward"):
                    loss.backward()
                with stage("optimizer_step"):
                    optimizer.step()
        return

    if protocol is ProfilingProtocol.REPEAT_BACKWARD_ON_SAME_GRAPH:
        with stage("total"):
            optimizer.zero_grad()
            with stage("forward_phase"):
                for _ in range(steps):
                    with stage("forward"):
                        output = module(*case.inputs)
                        loss = case.loss_fn(output)
            with stage("backward_phase"):
                for ind in range(steps):
                    # Backprop only for the last loss
                    with stage("backward"):
                        loss.backward(retain_graph=ind < steps - 1)
        return

    raise AssertionError(f"Unhandled profiling protocol: {protocol}")
