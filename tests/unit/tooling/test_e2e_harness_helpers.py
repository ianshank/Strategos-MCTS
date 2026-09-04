"""Invariants for the end-to-end harness's own helpers.

``tests/utils/device_matrix.py`` and ``tests/utils/e2e_process.py`` are the foundation the
whole e2e suite stands on, which makes them a single point of *silent* failure. If
``device_params`` ever returned an empty list, or ``requested_devices`` silently returned
``None``, the device matrix would collapse to nothing: the e2e tests would collect no
device cases and the run would still be green. That is precisely the "green but not
checked" defect the e2e program exists to eliminate, so the foundation gets the same
treatment ``tests/unit/test_ci_workflow_invariants.py`` gives its own parsers.

The same reasoning covers ``hermetic_env``: it is the guard that keeps a child process
from reaching a paid provider, and this suite also runs in a post-merge workflow that
exports real credentials. A stripping bug there is invisible until it bills someone, so
the strip list is asserted directly rather than trusted.

Everything here is pure: no subprocess is spawned and no device is required.
"""

from __future__ import annotations

from pathlib import Path

from _pytest.outcomes import Failed
import pytest

from tests.utils.device_matrix import (
    ACCELERATOR_DEVICES,
    CPU_DEVICE,
    CUDA_DEVICE,
    DEVICE_MATRIX,
    DEVICE_MATRIX_ENV,
    GPU_MARKER_NAME,
    MPS_DEVICE,
    NO_ACCELERATOR_ID,
    DeviceCase,
    device_available,
    device_cases,
    device_params,
    requested_devices,
    require_case,
)
from tests.utils.e2e_process import (
    DEFAULT_SUBPROCESS_TIMEOUT_SECONDS,
    HERMETIC_ENV_DEFAULTS,
    STRIPPED_ENV_VARS,
    SUBPROCESS_TIMEOUT_ENV,
    ProcessResult,
    hermetic_env,
    subprocess_timeout_seconds,
)

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path("/repo")


@pytest.fixture(autouse=True)
def _unpinned_device_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run these invariants against the *implicit* matrix, whatever the shell holds.

    ``device_params()`` and ``device_cases()`` read the process environment by default, so
    a developer who exported ``E2E_DEVICES`` (say, after ``E2E_DEVICES=cuda make
    test-e2e``) would see three of these tests fail while the helpers were entirely
    correct. A unit test whose verdict depends on ambient state is not an invariant.

    Cases that exercise the override pass their own mapping explicitly, so clearing the
    variable here narrows nothing.
    """
    monkeypatch.delenv(DEVICE_MATRIX_ENV, raising=False)


# --------------------------------------------------------------- the matrix itself


def test_cpu_is_always_in_the_matrix_and_is_not_an_accelerator() -> None:
    """CPU is the one device every host has, so it anchors the matrix."""
    assert CPU_DEVICE in DEVICE_MATRIX
    assert CPU_DEVICE not in ACCELERATOR_DEVICES
    assert set(ACCELERATOR_DEVICES) == set(DEVICE_MATRIX) - {CPU_DEVICE}
    assert {CUDA_DEVICE, MPS_DEVICE} <= set(DEVICE_MATRIX)


def test_the_implicit_matrix_is_never_empty() -> None:
    """The guard against a silent collapse.

    Every other assertion in the e2e suite is parametrized off this list. If it ever came
    back empty the suite would collect zero device cases and still report success, which
    is the failure this whole program exists to prevent.
    """
    params = device_params()
    assert params, "the device matrix collapsed to nothing; the e2e suite would silently check no device"
    assert len(params) == len(DEVICE_MATRIX)
    assert [param.id for param in params] == list(DEVICE_MATRIX)


def test_cpu_is_reported_available_without_consulting_torch() -> None:
    """CPU availability must not depend on the optional neural extra."""
    assert device_available(CPU_DEVICE) is True


def test_an_unknown_device_is_rejected_rather_than_assumed_absent() -> None:
    """A typo must fail loudly; treating it as 'unavailable' would silently skip."""
    with pytest.raises(ValueError, match="unknown device"):
        device_available("tpu")


# ------------------------------------------------------------ E2E_DEVICES parsing


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", None),
        ("   ", None),
        ("cpu", (CPU_DEVICE,)),
        ("cuda,cpu", (CPU_DEVICE, CUDA_DEVICE)),  # normalized to matrix order
        ("CUDA", (CUDA_DEVICE,)),  # case-insensitive
        (" cpu , cpu ", (CPU_DEVICE,)),  # de-duplicated
        ("cpu,cuda,mps", DEVICE_MATRIX),
    ],
)
def test_requested_devices_parses_the_override(raw: str, expected: tuple[str, ...] | None) -> None:
    assert requested_devices({DEVICE_MATRIX_ENV: raw}) == expected


def test_requested_devices_rejects_an_unknown_name() -> None:
    """A pinned matrix naming a device that does not exist is a configuration error."""
    with pytest.raises(ValueError, match="unknown device"):
        requested_devices({DEVICE_MATRIX_ENV: "cpu,gpu0"})


def test_an_unset_override_yields_the_whole_matrix_as_not_required() -> None:
    cases = device_cases({})
    assert [case.name for case in cases] == list(DEVICE_MATRIX)
    assert all(case.requested is False for case in cases)


def test_an_explicit_override_marks_every_named_device_required() -> None:
    cases = device_cases({DEVICE_MATRIX_ENV: "cuda"})
    assert [case.name for case in cases] == [CUDA_DEVICE]
    assert cases[0].requested is True


# ------------------------------------------------------- skip versus fail semantics


def test_an_available_device_neither_skips_nor_fails() -> None:
    case = DeviceCase(name=CPU_DEVICE, available=True, requested=False)
    assert case.skip_reason is None
    assert case.failure_reason is None
    assert require_case(case) == CPU_DEVICE


def test_an_absent_device_skips_when_it_was_not_asked_for() -> None:
    """The implicit matrix reports the gap; it does not manufacture a failure."""
    case = DeviceCase(name=CUDA_DEVICE, available=False, requested=False)
    assert case.failure_reason is None
    assert case.skip_reason is not None
    # The reason has to tell a reader how to turn the skip into a hard requirement.
    assert CUDA_DEVICE in case.skip_reason
    assert DEVICE_MATRIX_ENV in case.skip_reason


def test_an_absent_device_fails_when_it_was_explicitly_required() -> None:
    """A pinned matrix that cannot be honoured is a broken host, not a skip.

    Without this the ``E2E_DEVICES=cuda`` contract would be unfalsifiable: an operator
    pinning a device would get a green run that verified nothing on it.
    """
    case = DeviceCase(name=CUDA_DEVICE, available=False, requested=True)
    assert case.skip_reason is None, "a required device must not be quietly skipped"
    assert case.failure_reason is not None
    with pytest.raises(Failed, match=DEVICE_MATRIX_ENV):
        require_case(case)


def test_an_available_device_is_returned_even_when_required() -> None:
    case = DeviceCase(name=CUDA_DEVICE, available=True, requested=True)
    assert require_case(case) == CUDA_DEVICE


# ------------------------------------------------------------------ param marking


def test_accelerator_params_carry_the_gpu_marker_and_cpu_does_not() -> None:
    """``-m "not gpu"`` must deselect exactly the accelerator cases."""
    by_id = {param.id: param for param in device_params()}
    assert GPU_MARKER_NAME not in {mark.name for mark in by_id[CPU_DEVICE].marks}
    for accelerator in ACCELERATOR_DEVICES:
        assert GPU_MARKER_NAME in {mark.name for mark in by_id[accelerator].marks}


def test_unavailable_cases_are_marked_skip_at_collection() -> None:
    """Attached as a mark, not raised in the fixture, so junit records them as skips."""
    cases = (
        DeviceCase(name=CPU_DEVICE, available=True, requested=False),
        DeviceCase(name=CUDA_DEVICE, available=False, requested=False),
    )
    by_id = {param.id: param for param in device_params(cases)}
    assert "skip" not in {mark.name for mark in by_id[CPU_DEVICE].marks}
    assert "skip" in {mark.name for mark in by_id[CUDA_DEVICE].marks}


def test_a_required_case_is_not_pre_marked_skip() -> None:
    """It must reach the fixture body so ``require_case`` can fail it with the operator's own config."""
    cases = (DeviceCase(name=CUDA_DEVICE, available=False, requested=True),)
    marks = {mark.name for mark in device_params(cases)[0].marks}
    assert "skip" not in marks


def test_accelerators_only_drops_cpu() -> None:
    """The cross-device comparison needs two devices, so CPU alone is not a case."""
    ids = [param.id for param in device_params(accelerators_only=True)]
    assert CPU_DEVICE not in ids
    assert ids == list(ACCELERATOR_DEVICES)


def test_an_accelerator_free_matrix_still_names_its_reason() -> None:
    """``E2E_DEVICES=cpu`` must not degrade to pytest's bare "got empty parameter set".

    An empty ``params=`` is reported without any explanation of why the case vanished,
    which is the unreasoned skip this suite exists to eliminate. One explicitly reasoned
    placeholder is emitted instead.
    """
    cpu_only = (DeviceCase(name=CPU_DEVICE, available=True, requested=True),)
    params = device_params(cpu_only, accelerators_only=True)

    assert len(params) == 1, "an accelerator-free matrix must still yield exactly one reported case"
    assert params[0].id == NO_ACCELERATOR_ID
    marks = {mark.name for mark in params[0].marks}
    assert "skip" in marks
    assert GPU_MARKER_NAME in marks, "the placeholder must stay deselectable by -m 'not gpu'"

    reason = next(mark for mark in params[0].marks if mark.name == "skip").kwargs["reason"]
    assert DEVICE_MATRIX_ENV in reason, "the skip must tell the reader how to get an accelerator into the matrix"


def test_the_accelerator_placeholder_carries_an_unavailable_accelerator() -> None:
    """Its failure direction is deliberate.

    If the skip mark were ever dropped, the consuming test must fail on a missing device
    rather than quietly comparing CPU against CPU and reporting agreement.
    """
    cpu_only = (DeviceCase(name=CPU_DEVICE, available=True, requested=False),)
    case = device_params(cpu_only, accelerators_only=True)[0].values[0]
    assert case.is_accelerator
    assert case.available is False


# ------------------------------------------------------------- the hermetic environment


#: Credentials a child must never inherit, named here INDEPENDENTLY of the strip list itself.
#:
#: This duplication is deliberate and load-bearing. Building the input from
#: ``STRIPPED_ENV_VARS`` and then asserting nothing in ``STRIPPED_ENV_VARS`` survives is a
#: tautology: it proves the function strips what the list names, and can never detect a
#: credential *missing from the list*. That is not hypothetical — it is exactly how
#: ``AWS_ACCESS_KEY_ID`` and ``AWS_SECRET_ACCESS_KEY`` went unstripped while a test asserting
#: "every declared secret is stripped" stayed green, with ``aioboto3`` a core dependency.
SENSITIVE_ENV_VARS_THAT_MUST_NOT_REACH_A_CHILD = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "LANGSMITH_API_KEY",
    "LANGCHAIN_API_KEY",
    "WANDB_API_KEY",
    "PINECONE_API_KEY",
    "BRAINTRUST_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
)


def test_no_sensitive_credential_reaches_a_child_environment() -> None:
    """The network guard, asserted against an independently-named list.

    This suite also runs in a post-merge workflow that exports real provider keys, so a
    hole here is invisible until it bills someone — or, for the AWS pair, until a child
    process reaches S3 with production credentials.
    """
    base = dict.fromkeys(SENSITIVE_ENV_VARS_THAT_MUST_NOT_REACH_A_CHILD, "a-real-looking-secret")
    env = hermetic_env(repo_root=REPO_ROOT, base=base)
    leaked = sorted(name for name in SENSITIVE_ENV_VARS_THAT_MUST_NOT_REACH_A_CHILD if name in env)
    assert not leaked, f"these credentials would reach the child process: {leaked}"


def test_the_strip_list_covers_every_name_this_test_module_knows_about() -> None:
    """Fails when a credential is added here but not to the helper — the real regression.

    Kept separate from the behavioural test above so the failure message distinguishes
    "the stripping is broken" from "the list is incomplete".
    """
    missing = sorted(set(SENSITIVE_ENV_VARS_THAT_MUST_NOT_REACH_A_CHILD) - set(STRIPPED_ENV_VARS))
    assert not missing, f"tests/utils/e2e_process.py STRIPPED_ENV_VARS is missing: {missing}"


def test_the_strip_list_has_no_duplicates() -> None:
    """A list with repeats reads as carefully assembled when it was not.

    It had two, because the shared ``LLM_PROVIDER_CREDENTIAL_ENV_VARS`` constant already
    carried names the local tuple repeated.
    """
    assert len(STRIPPED_ENV_VARS) == len(set(STRIPPED_ENV_VARS)), f"duplicates in {STRIPPED_ENV_VARS}"


def test_the_offline_posture_is_pinned() -> None:
    """A child must not phone home to a hub or a tracker even if the parent would."""
    env = hermetic_env(repo_root=REPO_ROOT, base={"HF_HUB_OFFLINE": "0", "WANDB_MODE": "online"})
    for name, value in HERMETIC_ENV_DEFAULTS.items():
        assert env[name] == value


def test_overrides_are_applied_last_and_none_deletes() -> None:
    env = hermetic_env(
        repo_root=REPO_ROOT,
        base={"KEEP": "1", "DROP": "1"},
        overrides={"ADDED": "yes", "DROP": None, "HF_HUB_OFFLINE": "0"},
    )
    assert env["KEEP"] == "1"
    assert env["ADDED"] == "yes"
    assert "DROP" not in env
    # An explicit override beats the pinned default: the DDP test needs exactly this to
    # blank CUDA_VISIBLE_DEVICES and select the gloo backend.
    assert env["HF_HUB_OFFLINE"] == "0"


def test_repo_root_is_prepended_to_pythonpath() -> None:
    """``python -m src...`` has to resolve from whatever cwd the child is given."""
    fresh = hermetic_env(repo_root=REPO_ROOT, base={})
    assert fresh["PYTHONPATH"] == str(REPO_ROOT)

    inherited = hermetic_env(repo_root=REPO_ROOT, base={"PYTHONPATH": "/existing"})
    assert inherited["PYTHONPATH"].split(":")[0] == str(REPO_ROOT)
    assert "/existing" in inherited["PYTHONPATH"]


# ---------------------------------------------------------------- the child timeout


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", DEFAULT_SUBPROCESS_TIMEOUT_SECONDS),
        ("45", 45.0),
        ("12.5", 12.5),
        ("not-a-number", DEFAULT_SUBPROCESS_TIMEOUT_SECONDS),  # malformed must not disable the bound
        ("0", DEFAULT_SUBPROCESS_TIMEOUT_SECONDS),  # nor may zero
        ("-1", DEFAULT_SUBPROCESS_TIMEOUT_SECONDS),  # nor a negative
    ],
)
def test_a_malformed_timeout_override_never_removes_the_bound(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: float
) -> None:
    """An unbounded child could hold a hung process group past the CI job's own timeout."""
    monkeypatch.setenv(SUBPROCESS_TIMEOUT_ENV, raw)
    assert subprocess_timeout_seconds() == expected


# ------------------------------------------------------------------ failure reporting


def test_a_failure_describes_itself_completely() -> None:
    """``describe()`` is the assertion message, so a red CI log must explain itself."""
    result = ProcessResult(
        argv=("python", "-m", "src.training.self_play_convergence"),
        returncode=2,
        stdout="the last thing it printed",
        stderr="the traceback",
        duration_seconds=1.5,
    )
    assert result.ok is False
    rendered = result.describe()
    assert "python -m src.training.self_play_convergence" in rendered
    assert "exit 2" in rendered
    assert "the last thing it printed" in rendered
    assert "the traceback" in rendered


def test_a_timed_out_run_is_not_ok_even_with_a_zero_exit_code() -> None:
    """``ok`` must consider both halves.

    A killed child can report 0 depending on how the kill lands, so ``returncode == 0``
    alone is not success. Without this case, deleting ``and not self.timed_out`` from
    ``ProcessResult.ok`` leaves the whole suite green.
    """
    assert (
        ProcessResult(argv=("x",), returncode=0, stdout="", stderr="", duration_seconds=1.0, timed_out=True).ok is False
    )


def test_describe_keeps_the_END_of_a_long_stream() -> None:
    """The tail, not the head: a failure's cause is at the end of the log, not the start."""
    body = "\n".join(f"line-{index}" for index in range(200))
    rendered = ProcessResult(argv=("x",), returncode=1, stdout=body, stderr="", duration_seconds=1.0).describe(
        tail_lines=5
    )
    assert "line-199" in rendered, "describe() dropped the end of the stream, where the failure is"
    assert "line-0" not in rendered, "describe() kept the head instead of the tail"


def test_a_timeout_is_reported_as_a_timeout_not_as_an_exit_code() -> None:
    """A killed child has a misleading returncode; the rendering must not imply it exited."""
    result = ProcessResult(argv=("sleep",), returncode=-9, stdout="", stderr="", duration_seconds=240.0, timed_out=True)
    assert result.ok is False
    assert "TIMED OUT" in result.describe()


def test_empty_streams_render_as_a_placeholder() -> None:
    """An empty section must read as 'nothing was printed', not as a truncated log."""
    result = ProcessResult(argv=("true",), returncode=0, stdout="", stderr="", duration_seconds=0.1)
    assert result.ok is True
    assert "<empty>" in result.describe()
