import pytest

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def test_deferred_normalization_rejects_cpu_batches() -> None:
    torch = pytest.importorskip("torch")
    normalizer = pytest.importorskip("augbench.implementations.gpu_normalize").GpuBatchNormalize
    batch = torch.zeros((1, 2, 2, 3), dtype=torch.uint8)

    with pytest.raises(RuntimeError, match="CUDA batch"):
        normalizer(mean=_MEAN, std=_STD, input_layout="BHWC")(batch)


def test_deferred_normalization_makes_float16_bchw_on_cuda() -> None:
    torch = pytest.importorskip("torch")
    normalizer = pytest.importorskip("augbench.implementations.gpu_normalize").GpuBatchNormalize
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    batch = torch.zeros((2, 3, 2, 2), dtype=torch.float32, device="cuda")

    output = normalizer(mean=_MEAN, std=_STD, input_layout="BCHW")(batch)

    expected = -torch.tensor(_MEAN, dtype=torch.float16).view(1, 3, 1, 1) / torch.tensor(
        _STD,
        dtype=torch.float16,
    ).view(1, 3, 1, 1)
    assert output.device.type == "cuda"
    assert output.dtype == torch.float16
    assert torch.allclose(output, expected.expand_as(output))
