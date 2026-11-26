import pytest

from augbench.run_config import TrainingConfig
from augbench.training.step import TrainSession

torch = pytest.importorskip("torch")

CLASSES = 10


def _tiny_model(_name: str, _device: str) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3),
        torch.nn.BatchNorm2d(4),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(4, CLASSES),
    )


def _session() -> TrainSession:
    config = TrainingConfig(
        model="resnet18",
        compile=False,
        learning_rate=0.1,
        momentum=0.9,
        weight_decay=1e-4,
        ceiling_warmup_steps=1,
        ceiling_steps=2,
        label_classes=CLASSES,
    )
    return TrainSession(config, batch_size=4, device="cpu", model_factory=_tiny_model)


def _batch() -> torch.Tensor:
    return torch.randn(4, 3, 8, 8, dtype=torch.float16)


def _weights(session: TrainSession) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in session.model.parameters()]


def test_a_step_returns_the_batch_size_and_a_finite_loss() -> None:
    session = _session()
    session.reset(seed=137)

    assert session.step(_batch()) == 4
    assert session.loss_is_finite()


def test_reset_restores_the_initial_weights_in_the_same_parameters() -> None:
    session = _session()
    session.reset(seed=137)
    initial = _weights(session)
    parameters = [id(parameter) for parameter in session.model.parameters()]

    session.step(_batch())
    assert any(not torch.equal(a, b) for a, b in zip(initial, _weights(session), strict=True))

    session.reset(seed=138)
    assert all(torch.equal(a, b) for a, b in zip(initial, _weights(session), strict=True))
    assert [id(parameter) for parameter in session.model.parameters()] == parameters


def test_reset_starts_a_fresh_optimizer() -> None:
    session = _session()
    session.reset(seed=137)
    session.step(_batch())
    assert session.optimizer_state_size() > 0

    session.reset(seed=137)

    assert session.optimizer_state_size() == 0


def test_labels_depend_only_on_the_seed() -> None:
    session = _session()
    session.reset(seed=137)
    first = session.labels.clone()
    session.reset(seed=138)
    other = session.labels.clone()
    session.reset(seed=137)

    assert torch.equal(first, session.labels)
    assert not torch.equal(first, other)
    assert int(first.max()) < CLASSES


def test_ceiling_reports_a_positive_rate_on_synthetic_input() -> None:
    session = _session()
    session.reset(seed=137)

    assert session.ceiling(_batch(), steps=2, warmup=1) > 0
