from craton.hardware import GPU, HardwareReport, architecture_name, parse_nvidia_smi, plan_precision

RTX_2060 = GPU("NVIDIA GeForce RTX 2060", 6.0, (7, 5))
RTX_3090 = GPU("NVIDIA GeForce RTX 3090", 24.0, (8, 6))


def report(*gpus: GPU, source: str = "torch") -> HardwareReport:
    return HardwareReport(
        python="3.11",
        os="test",
        torch_version="2.14.1",
        torch_cuda="12.6" if gpus else None,
        gpus=gpus,
        ram_gib=16.0,
        source=source,  # type: ignore[arg-type]
    )


def test_turing_has_fp16_but_no_bf16() -> None:
    assert RTX_2060.architecture == "Turing"
    assert RTX_2060.fp16_tensor_cores
    assert not RTX_2060.bf16


def test_bf16_on_turing_becomes_fp16_with_scaler_and_warning() -> None:
    plan = plan_precision("bf16", report(RTX_2060))
    assert (plan.dtype, plan.device, plan.grad_scaler) == ("fp16", "cuda", True)
    assert any("no native bf16" in w for w in plan.warnings)


def test_bf16_on_ampere_is_kept_without_scaler() -> None:
    plan = plan_precision("bf16", report(RTX_3090))
    assert (plan.dtype, plan.grad_scaler, plan.warnings) == ("bf16", False, ())


def test_fp16_always_uses_a_scaler() -> None:
    assert plan_precision("fp16", report(RTX_2060)).grad_scaler


def test_no_gpu_means_fp32_on_cpu() -> None:
    plan = plan_precision("fp16", report())
    assert (plan.dtype, plan.device, plan.grad_scaler) == ("fp32", "cpu", False)
    assert plan.warnings


def test_gpu_seen_only_by_nvidia_smi_is_not_usable() -> None:
    r = report(RTX_2060, source="nvidia-smi")
    assert not r.cuda_usable
    assert plan_precision("fp16", r).device == "cpu"


def test_parse_nvidia_smi() -> None:
    out = "NVIDIA GeForce RTX 2060, 6144, 7.5\nbroken line\nNVIDIA A100, 81920, 8.0\n"
    gpus = parse_nvidia_smi(out)
    assert gpus == (RTX_2060, GPU("NVIDIA A100", 80.0, (8, 0)))


def test_unknown_architecture_name() -> None:
    assert architecture_name((99, 1)) == "sm_991"
