from contextlib import contextmanager

import pytest
import torch

from profiling.cases import ProfilingCase
from profiling.protocols import ProfilingProtocol, parse_protocol, run_protocol


class RecordingModule(torch.nn.Module):
    def __init__(self, events):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.events = events

    def forward(self, inputs):
        self.events.append("forward")
        output = self.weight * inputs
        if output.requires_grad:
            output.register_hook(lambda _gradient: self.events.append("backward"))
        return output


class RecordingSGD(torch.optim.SGD):
    def __init__(self, params, events):
        super().__init__(params, lr=0.1)
        self.events = events

    def zero_grad(self, *args, **kwargs):
        self.events.append("zero_grad")
        return super().zero_grad(*args, **kwargs)

    def step(self, closure=None):
        self.events.append("optimizer_step")
        return super().step(closure)


def _build_workload(events):
    module = RecordingModule(events)
    case = ProfilingCase(
        module=module,
        inputs=(torch.tensor(2.0),),
        loss_fn=lambda output: output.square(),
    )
    optimizer = RecordingSGD(module.parameters(), events)
    return module, case, optimizer


def test_forward_only_runs_without_gradients():
    events = []
    module, case, optimizer = _build_workload(events)

    run_protocol(
        ProfilingProtocol.FORWARD_ONLY,
        steps=3,
        module=module,
        case=case,
        optimizer=optimizer,
    )

    assert events == ["forward"] * 3
    assert module.weight.grad is None


def test_full_training_step_runs_complete_steps():
    events = []
    module, case, optimizer = _build_workload(events)

    run_protocol(
        ProfilingProtocol.FULL_TRAINING_STEP,
        steps=2,
        module=module,
        case=case,
        optimizer=optimizer,
    )

    assert events == [
        "zero_grad",
        "forward",
        "backward",
        "optimizer_step",
        "zero_grad",
        "forward",
        "backward",
        "optimizer_step",
    ]


def test_repeat_backward_protocol_runs_all_forwards_before_backwards():
    events = []
    module, case, optimizer = _build_workload(events)

    run_protocol(
        ProfilingProtocol.REPEAT_BACKWARD_ON_SAME_GRAPH,
        steps=3,
        module=module,
        case=case,
        optimizer=optimizer,
    )

    assert events == ["zero_grad"] + ["forward"] * 3 + ["backward"] * 3
    assert module.weight.grad.item() == pytest.approx(24.0)


def test_repeat_backward_protocol_records_operation_and_phase_stages():
    events = []
    stages = []
    module, case, optimizer = _build_workload(events)

    @contextmanager
    def record_stage(name):
        stages.append(("start", name))
        yield
        stages.append(("end", name))

    run_protocol(
        ProfilingProtocol.REPEAT_BACKWARD_ON_SAME_GRAPH,
        steps=2,
        module=module,
        case=case,
        optimizer=optimizer,
        stage=record_stage,
    )

    assert [event for event in stages if event == ("start", "forward")] == [
        ("start", "forward"),
        ("start", "forward"),
    ]
    assert [event for event in stages if event == ("start", "backward")] == [
        ("start", "backward"),
        ("start", "backward"),
    ]
    assert ("start", "forward_phase") in stages
    assert ("start", "backward_phase") in stages
    assert stages[0] == ("start", "total")
    assert stages[-1] == ("end", "total")


def test_parse_protocol_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unsupported profiling protocol"):
        parse_protocol("unknown")
