from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from silenttwin.agentdojo.runtime_integrity import (
    RuntimeIntegrityError,
    derive_learned_runtime_fingerprint,
    make_learned_runtime_provenance,
    not_applicable_learned_runtime_provenance,
    validate_learned_runtime_provenance,
    validate_locked_distributions,
    verify_agentdojo_distribution,
)


class _RecordHash:
    def __init__(self, value: str) -> None:
        self.mode = "sha256"
        self.value = value


class _File:
    def __init__(self, path: str, *, identity: str = "record-value", size: int = 1) -> None:
        self.path = path
        self.hash = _RecordHash(identity)
        self.size = size

    def __str__(self) -> str:
        return self.path


class _Distribution:
    def __init__(
        self,
        name: str,
        version: str,
        *,
        root: Path | None = None,
        files: list[_File] | None = None,
    ) -> None:
        self.metadata = {"Name": name}
        self.version = version
        self.root = root
        self.files = files

    def locate_file(self, item: object) -> Path:
        assert self.root is not None
        return self.root / str(item)


def _payload_fingerprint(files: dict[str, bytes]) -> str:
    rows = [
        {
            "path": path,
            "sha256": hashlib.sha256(content).hexdigest(),
            "size": len(content),
        }
        for path, content in sorted(files.items())
    ]
    encoded = json.dumps(
        rows, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _installed_distribution(tmp_path: Path) -> tuple[_Distribution, dict[str, bytes]]:
    files = {
        "agentdojo/__init__.py": b"__version__ = '0.1.35'\n",
        "agentdojo-0.1.35.dist-info/METADATA": b"Name: agentdojo\nVersion: 0.1.35\n",
    }
    for relative, content in files.items():
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    distribution = _Distribution(
        "agentdojo",
        "0.1.35",
        root=tmp_path,
        files=[_File(path, size=len(content)) for path, content in files.items()],
    )
    return distribution, files


def test_installed_payload_mode_does_not_claim_wheel_archive_verification(
    tmp_path: Path,
) -> None:
    distribution, files = _installed_distribution(tmp_path)
    report = verify_agentdojo_distribution(
        distribution=distribution,
        expected_payload_sha256=_payload_fingerprint(files),
        expected_payload_file_count=len(files),
    )
    assert report.verification_mode == (
        "installed_payload_against_frozen_wheel_payload_manifest"
    )
    assert report.wheel_artifact_sha256 is None
    assert report.wheel_artifact_verified is False


def test_verified_wheel_is_compared_with_installed_payload(tmp_path: Path) -> None:
    installed_root = tmp_path / "installed"
    distribution, files = _installed_distribution(installed_root)
    wheel = tmp_path / "agentdojo-0.1.35-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for relative, content in files.items():
            archive.writestr(relative, content)
        archive.writestr("agentdojo-0.1.35.dist-info/RECORD", b"")
    wheel_digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    report = verify_agentdojo_distribution(
        distribution=distribution,
        wheel_artifact=wheel,
        expected_wheel_sha256=wheel_digest,
        expected_payload_sha256=_payload_fingerprint(files),
        expected_payload_file_count=len(files),
    )
    assert report.verification_mode == (
        "verified_published_wheel_and_matching_installed_payload"
    )
    assert report.wheel_artifact_verified is True

    (installed_root / "agentdojo/__init__.py").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeIntegrityError, match="payload bytes differ"):
        verify_agentdojo_distribution(
            distribution=distribution,
            wheel_artifact=wheel,
            expected_wheel_sha256=wheel_digest,
            expected_payload_sha256=_payload_fingerprint(files),
            expected_payload_file_count=len(files),
        )


def test_wrong_operator_wheel_digest_fails_closed(tmp_path: Path) -> None:
    distribution, files = _installed_distribution(tmp_path / "installed")
    wheel = tmp_path / "agentdojo.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for relative, content in files.items():
            archive.writestr(relative, content)
    with pytest.raises(RuntimeIntegrityError, match="wheel SHA-256"):
        verify_agentdojo_distribution(
            distribution=distribution,
            wheel_artifact=wheel,
            expected_wheel_sha256="0" * 64,
            expected_payload_sha256=_payload_fingerprint(files),
            expected_payload_file_count=len(files),
        )


def test_exact_lock_validates_every_installed_core_version(tmp_path: Path) -> None:
    lock = tmp_path / "requirements.lock"
    lock.write_text("agentdojo==0.1.35\ntorch==2.7.1\n", encoding="utf-8")
    versions = {"agentdojo": "0.1.35", "torch": "2.7.1"}
    report = validate_locked_distributions(
        lock,
        version_resolver=versions.__getitem__,
        expected_count=None,
    )
    assert report.distribution_count == 2

    versions["torch"] = "2.7.2"
    with pytest.raises(RuntimeIntegrityError, match="torch:2.7.2"):
        validate_locked_distributions(
            lock,
            version_resolver=versions.__getitem__,
            expected_count=None,
        )


def test_runtime_fingerprint_is_deterministic_and_detects_manifest_drift(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "requirements.lock"
    lock.write_text("agentdojo==0.1.35\n", encoding="utf-8")
    versions = {"agentdojo": "0.1.35"}
    agentdojo = _Distribution(
        "agentdojo", "0.1.35", files=[_File("agentdojo/__init__.py", identity="a")]
    )
    torch = _Distribution(
        "torch", "2.7.1", files=[_File("torch/__init__.py", identity="b")]
    )
    transformers = _Distribution(
        "transformers",
        "4.55.0",
        files=[_File("transformers/__init__.py", identity="c")],
    )
    python_identity = {
        "implementation": "cpython",
        "version": [3, 11, 15],
        "cache_tag": "cpython-311",
    }
    first = derive_learned_runtime_fingerprint(
        lock,
        distributions=[agentdojo, torch, transformers],
        version_resolver=versions.__getitem__,
        expected_lock_count=None,
        python_identity=python_identity,
    )
    reordered = derive_learned_runtime_fingerprint(
        lock,
        distributions=[transformers, agentdojo, torch],
        version_resolver=versions.__getitem__,
        expected_lock_count=None,
        python_identity=python_identity,
    )
    assert first.fingerprint == reordered.fingerprint
    assert first.fingerprint.startswith("sha256:")

    drifted_transformers = _Distribution(
        "transformers",
        "4.55.0",
        files=[_File("transformers/__init__.py", identity="changed")],
    )
    drifted = derive_learned_runtime_fingerprint(
        lock,
        distributions=[agentdojo, torch, drifted_transformers],
        version_resolver=versions.__getitem__,
        expected_lock_count=None,
        python_identity=python_identity,
    )
    assert drifted.fingerprint != first.fingerprint


def test_runtime_manifest_sorts_an_unsorted_core_lock(tmp_path: Path) -> None:
    lock = tmp_path / "requirements.lock"
    lock.write_text(
        "torch==2.7.1\nagentdojo==0.1.35\n", encoding="utf-8"
    )
    versions = {"agentdojo": "0.1.35", "torch": "2.7.1"}
    report = derive_learned_runtime_fingerprint(
        lock,
        distributions=[
            _Distribution("transformers", "4.55.0", files=[]),
            _Distribution("torch", "2.7.1", files=[]),
            _Distribution("agentdojo", "0.1.35", files=[]),
        ],
        version_resolver=versions.__getitem__,
        expected_lock_count=None,
        python_identity={
            "implementation": "cpython",
            "version": [3, 11, 15],
            "cache_tag": "cpython-311",
        },
    )
    assert report.manifest["locked_core"] == [
        {"name": "agentdojo", "version": "0.1.35"},
        {"name": "torch", "version": "2.7.1"},
    ]
    assert [
        row["name"] for row in report.manifest["installed_distributions"]
    ] == ["agentdojo", "torch", "transformers"]


def test_learned_fingerprint_requires_local_inference_stack(tmp_path: Path) -> None:
    lock = tmp_path / "requirements.lock"
    lock.write_text("agentdojo==0.1.35\n", encoding="utf-8")
    agentdojo = _Distribution("agentdojo", "0.1.35", files=[])
    with pytest.raises(RuntimeIntegrityError, match="torch, transformers"):
        derive_learned_runtime_fingerprint(
            lock,
            distributions=[agentdojo],
            version_resolver=lambda _: "0.1.35",
            expected_lock_count=None,
        )


def test_retained_runtime_manifest_recomputes_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "requirements.lock"
    lock.write_text("agentdojo==0.1.35\n", encoding="utf-8")
    report = derive_learned_runtime_fingerprint(
        lock,
        distributions=[
            _Distribution(
                "agentdojo",
                "0.1.35",
                files=[_File("agentdojo/__init__.py", identity="a")],
            ),
            _Distribution(
                "torch",
                "2.7.1+site",
                files=[_File("torch/__init__.py", identity="b")],
            ),
            _Distribution(
                "transformers",
                "4.55.0",
                files=[_File("transformers/__init__.py", identity="c")],
            ),
        ],
        version_resolver=lambda _: "0.1.35",
        expected_lock_count=None,
        python_identity={
            "implementation": "cpython",
            "version": [3, 11, 15],
            "cache_tag": "cpython-311",
            "abi_flags": "",
            "soabi": "cpython-311-x86_64-linux-gnu",
            "byteorder": "little",
            "system": "Linux",
            "machine": "x86_64",
        },
    )
    provenance = make_learned_runtime_provenance(report)
    assert provenance["runtime_fingerprint"] == report.fingerprint
    assert provenance["distribution_count"] == 3
    assert [
        row["name"] for row in provenance["manifest"]["installed_distributions"]
    ] == ["agentdojo", "torch", "transformers"]
    assert validate_learned_runtime_provenance(
        provenance,
        expected_runtime_fingerprints={report.fingerprint},
    ) == report.fingerprint

    tampered = json.loads(json.dumps(provenance))
    tampered["manifest"]["installed_distributions"][2]["version"] = "4.55.1"
    with pytest.raises(RuntimeIntegrityError, match="does not match its manifest"):
        validate_learned_runtime_provenance(
            tampered,
            expected_runtime_fingerprints={report.fingerprint},
        )

    model_free = not_applicable_learned_runtime_provenance()
    assert validate_learned_runtime_provenance(
        model_free, expected_runtime_fingerprints=set()
    ) == "not_applicable"
    with pytest.raises(RuntimeIntegrityError, match="cannot use not_applicable"):
        validate_learned_runtime_provenance(
            model_free,
            expected_runtime_fingerprints={report.fingerprint},
        )


def test_runtime_validation_uses_derived_identity_not_arbitrary_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from silenttwin.agentdojo import runtime_validation

    report_lock = tmp_path / "report.lock"
    report_lock.write_text("agentdojo==0.1.35\n", encoding="utf-8")
    report = derive_learned_runtime_fingerprint(
        report_lock,
        distributions=[
            _Distribution("agentdojo", "0.1.35", files=[]),
            _Distribution("torch", "2.7.1", files=[]),
            _Distribution("transformers", "4.55.0", files=[]),
        ],
        version_resolver=lambda _: "0.1.35",
        expected_lock_count=None,
        python_identity={
            "implementation": "cpython",
            "version": [3, 11, 15],
            "cache_tag": "cpython-311",
            "abi_flags": "",
            "soabi": "cpython-311-x86_64-linux-gnu",
            "byteorder": "little",
            "system": "Linux",
            "machine": "x86_64",
        },
    )
    expected = report.fingerprint
    calls: list[str] = []
    monkeypatch.setattr(
        runtime_validation,
        "verify_agentdojo_distribution",
        lambda **kwargs: calls.append(f"wheel={kwargs['wheel_artifact']}"),
    )
    monkeypatch.setattr(
        runtime_validation,
        "validate_locked_distributions",
        lambda path: calls.append(f"lock={path}"),
    )
    monkeypatch.setattr(
        runtime_validation,
        "derive_learned_runtime_fingerprint",
        lambda *args, **kwargs: report,
    )
    monkeypatch.setenv("AGENTDOJO_WHEEL_ARTIFACT", "/persistent/agentdojo.whl")
    monkeypatch.setenv("AGENTDOJO_RUNTIME_FINGERPRINT", expected)
    retained = runtime_validation.validate_environment_integrity(
        dependency_lock_path=tmp_path / "lock",
        fixture_mode=False,
        runtime_fingerprints={expected},
    )
    assert retained["status"] == "captured"
    assert retained["runtime_fingerprint"] == expected
    assert retained["manifest"] == report.manifest
    assert calls == [
        "wheel=/persistent/agentdojo.whl",
        f"lock={tmp_path / 'lock'}",
    ]

    monkeypatch.setenv("AGENTDOJO_RUNTIME_FINGERPRINT", "sha256:" + "b" * 64)
    with pytest.raises(runtime_validation.RuntimeArtifactError, match="derived active"):
        runtime_validation.validate_environment_integrity(
            dependency_lock_path=tmp_path / "lock",
            fixture_mode=False,
            runtime_fingerprints={expected},
        )


def test_runtime_validation_rejects_derived_environment_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from silenttwin.agentdojo import runtime_validation

    monkeypatch.setattr(runtime_validation, "verify_agentdojo_distribution", lambda **_: None)
    monkeypatch.setattr(runtime_validation, "validate_locked_distributions", lambda _: None)
    monkeypatch.setattr(
        runtime_validation,
        "derive_learned_runtime_fingerprint",
        lambda *args, **kwargs: SimpleNamespace(fingerprint="sha256:" + "c" * 64),
    )
    monkeypatch.setenv("AGENTDOJO_RUNTIME_FINGERPRINT", "sha256:" + "a" * 64)
    with pytest.raises(runtime_validation.RuntimeArtifactError, match="scientific model"):
        runtime_validation.validate_environment_integrity(
            dependency_lock_path=tmp_path / "lock",
            fixture_mode=False,
            runtime_fingerprints={"sha256:" + "a" * 64},
        )


def test_pair_observation_preflight_binds_profiles_before_client_construction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from silenttwin.agentdojo import runner

    expected = "sha256:" + "d" * 64
    captured: dict[str, object] = {}
    checkpoint = tmp_path / "persistent-monitor-checkpoint"
    checkpoint.mkdir()
    for variable in (
        "SLURM_TMPDIR",
        "AGENTDOJO_MONITOR_CHECKPOINT",
        "AGENTDOJO_MONITOR_CHECKPOINT_MONITOR_A",
        "AGENTDOJO_MONITOR_CHECKPOINT_MONITOR_B",
        "AGENTDOJO_MODEL_CACHE",
        "HF_HOME",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("AGENTDOJO_MONITOR_CHECKPOINT", str(checkpoint))

    def _capture(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(runner, "validate_environment_integrity", _capture)
    runner._preflight_pair_observation_environment(
        strategy_catalog={
            "monitor_profiles": [
                {
                    "profile_id": "monitor-a",
                    "implementation": "local_transformers",
                    "runtime_fingerprint": expected,
                },
                {
                    "profile_id": "monitor-b",
                    "implementation": "local_transformers",
                    "runtime_fingerprint": expected,
                },
            ]
        },
        dependency_lock_path=tmp_path / "requirements.lock",
    )
    assert captured == {
        "dependency_lock_path": tmp_path / "requirements.lock",
        "fixture_mode": False,
        "runtime_fingerprints": {expected},
    }


def test_pair_observation_preflight_rejects_unbound_learned_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from silenttwin.agentdojo import runner

    monkeypatch.setattr(
        runner,
        "validate_environment_integrity",
        lambda **_: pytest.fail("runtime validation must not receive an invalid identity"),
    )
    with pytest.raises(runner.AgentDojoRunnerError, match="frozen runtime fingerprint"):
        runner._preflight_pair_observation_environment(
            strategy_catalog={
                "monitor_profiles": [
                    {
                        "implementation": "local_transformers",
                        "runtime_fingerprint": "operator-text",
                    }
                ]
            },
            dependency_lock_path=tmp_path / "requirements.lock",
        )


def test_checkpoint_path_inside_slurm_scratch_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from silenttwin.agentdojo import runtime_validation

    scratch = tmp_path / "slurm-scratch"
    checkpoint = scratch / "attacker-model"
    checkpoint.mkdir(parents=True)
    monkeypatch.setenv("SLURM_TMPDIR", str(scratch))
    monkeypatch.setenv("AGENTDOJO_ATTACKER_CHECKPOINT", str(checkpoint))
    with pytest.raises(runtime_validation.RuntimeArtifactError, match="SLURM_TMPDIR"):
        runtime_validation._validate_checkpoint_paths({"attacker"})

    persistent = tmp_path / "persistent" / "attacker-model"
    persistent.mkdir(parents=True)
    monkeypatch.setenv("AGENTDOJO_ATTACKER_CHECKPOINT", str(persistent))
    runtime_validation._validate_checkpoint_paths({"attacker"})


def test_cli_fingerprint_only_emits_directly_freezable_value(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from silenttwin.agentdojo import runtime_integrity

    expected = "sha256:" + "e" * 64
    report = SimpleNamespace(
        learned_runtime=SimpleNamespace(fingerprint=expected),
        to_dict=lambda **_: pytest.fail("JSON rendering is not used in fingerprint-only mode"),
    )
    monkeypatch.setattr(runtime_integrity, "verify_runtime_environment", lambda *_, **__: report)
    assert runtime_integrity.main(["--fingerprint-only"]) == 0
    assert capsys.readouterr().out == expected + "\n"
