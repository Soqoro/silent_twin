from __future__ import annotations

from dataclasses import replace
import unittest

from silenttwin.attackers import (
    BlackBoxAttacker,
    LLMAttacker,
    ModelResponse,
    ModelUsage,
    PublicTrialContext,
    TrainingExample,
    TrainingProvenance,
    TranscriptSnapshot,
    make_attacker,
    parse_probe_response,
    parse_prediction_response,
)
from silenttwin.attackers.llm_attacker import (
    StructuredProbeSelectionError,
    StructuredPredictionError,
)
from silenttwin.model_clients import (
    LocalModelConfig,
    LocalTransformersModelClient,
    checkpoint_fingerprint_manifest_path,
    fingerprint_local_checkpoint,
    prepare_local_checkpoint_fingerprint,
)
from silenttwin.schemas import VisibleEvent
from silenttwin.worlds import build_world_pair


def context(*, template_id: str = "heldout", split: str = "test") -> PublicTrialContext:
    pair = build_world_pair("email", seed=31)
    return PublicTrialContext.from_public_projection(
        pair.public_projection,
        query_budget=1,
        public_instance_hash=pair.public_instance_hash,
        runtime_name="generic_failure",
        feedback_protocol="genuine",
        template_id=template_id,
        dataset_split=split,
        dataset_revision="revision-1",
    )


def snapshot(status: str, error_code: str | None = None) -> TranscriptSnapshot:
    return TranscriptSnapshot.from_events(
        (
            VisibleEvent(
                event_type="tool_result",
                sequence=0,
                action_id="probe_cloud_link",
                tool_name="tool",
                status=status,
                error={"code": error_code, "message": "failed"} if error_code else None,
            ),
        )
    )


class InvalidClient:
    def complete(self, prompt: str, *, seed: int = 0, max_tokens: int = 128) -> ModelResponse:
        del prompt, seed, max_tokens
        return ModelResponse(
            text="theta1 perhaps",
            usage=ModelUsage(input_tokens=4, output_tokens=2),
            metadata={"model_id": "invalid-test-client", "retries": 0},
        )


class FailingClient:
    def complete(self, prompt: str, *, seed: int = 0, max_tokens: int = 128) -> ModelResponse:
        del prompt, seed, max_tokens
        raise RuntimeError("generation failed")

    def failure_metadata(self):
        return {
            "model_id": "failed-model",
            "model_revision": "a" * 40,
            "tokenizer_revision": "b" * 40,
            "torch_version": "test-torch",
            "gpu_name": "test-gpu",
            "local_files_only": True,
        }


class AdaptiveClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.seeds: list[int] = []

    def complete(self, prompt: str, *, seed: int = 0, max_tokens: int = 128) -> ModelResponse:
        del max_tokens
        self.prompts.append(prompt)
        self.seeds.append(seed)
        if "REQUEST_TYPE:probe_selection" in prompt:
            text = '{"probe_action_id":"probe_encoded_attachment"}'
        else:
            text = '{"prediction":"theta1","posterior":{"theta0":0.2,"theta1":0.8}}'
        return ModelResponse(
            text=text,
            usage=ModelUsage(input_tokens=10, output_tokens=4),
            metadata={
                "model_id": "adaptive-test",
                "model_revision": "deadbeef",
                "tokenizer_revision": "feedface",
                "retries": 0,
            },
        )


class RestrictedAttackerTests(unittest.TestCase):
    def test_public_actions_omit_policy_and_are_deeply_immutable(self) -> None:
        ctx = context()
        action = ctx.probes[0]
        self.assertNotIn("policy_tag", action.to_dict())
        with self.assertRaises(TypeError):
            action.arguments["new"] = "value"
        with self.assertRaises(TypeError):
            ctx.public_environment["new"] = "value"

    def test_structured_parser_rejects_invalid_or_partial_outputs(self) -> None:
        with self.assertRaises(StructuredPredictionError):
            parse_prediction_response("theta0")
        with self.assertRaises(StructuredPredictionError):
            parse_prediction_response(
                '{"prediction":"theta0","posterior":{"theta0":0.6,"theta1":0.6}}'
            )
        prediction, posterior = parse_prediction_response(
            '{"prediction":"theta1","posterior":{"theta0":0.25,"theta1":0.75}}'
        )
        self.assertEqual("theta1", prediction)
        self.assertEqual({"theta0": 0.25, "theta1": 0.75}, posterior)

        with self.assertRaises(StructuredProbeSelectionError):
            parse_probe_response('{"probe_action_id":"private-action"}', {"public"})
        self.assertEqual(
            "public",
            parse_probe_response('{"probe_action_id":"public"}', {"public"}),
        )

    def test_real_llm_selects_probe_from_public_context_and_defines_hypotheses(self) -> None:
        client = AdaptiveClient()
        attacker = LLMAttacker(client)
        selected = attacker.choose_probe(context(), TranscriptSnapshot(), 1)
        self.assertEqual("probe_encoded_attachment", selected)
        self.assertIn("candidate_hypotheses", client.prompts[0])
        self.assertIn("exploit_candidates", client.prompts[0])
        self.assertNotIn("actual_state", client.prompts[0])

        prediction = attacker.predict_hidden_state(context(), snapshot("ok"))
        self.assertEqual("theta1", prediction.prediction)
        self.assertEqual(20, prediction.telemetry.input_tokens)
        self.assertEqual(8, prediction.telemetry.output_tokens)
        self.assertEqual(1, len(prediction.telemetry.metadata["adaptive_probe_calls"]))

    def test_llm_phase_seeds_and_probe_prefixes_are_budget_independent(self) -> None:
        q4_client = AdaptiveClient()
        q16_client = AdaptiveClient()
        q4 = LLMAttacker(q4_client, seed=73)
        q16 = LLMAttacker(q16_client, seed=73)
        q4_context = replace(context(), query_budget=4)
        q16_context = replace(context(), query_budget=16)

        q4.choose_probe(q4_context, TranscriptSnapshot(), 4)
        q16.choose_probe(q16_context, TranscriptSnapshot(), 16)
        self.assertEqual(q4_client.prompts[0], q16_client.prompts[0])
        self.assertEqual(q4_client.seeds[0], q16_client.seeds[0])

        # The prediction payload can legitimately differ after different
        # numbers of observations; its fixed phase seed must not.
        for remaining in (15, 14, 13):
            q16.choose_probe(q16_context, TranscriptSnapshot(), remaining)
        q4.predict_hidden_state(q4_context, snapshot("ok"))
        q16.predict_hidden_state(q16_context, snapshot("ok"))
        self.assertEqual(q4_client.seeds[-1], q16_client.seeds[-1])

    def test_llm_invalid_output_is_recorded_without_a_guessed_label(self) -> None:
        attacker = LLMAttacker(InvalidClient())
        prediction = attacker.predict_hidden_state(context(), snapshot("ok"))
        self.assertFalse(prediction.valid)
        self.assertIsNone(prediction.prediction)
        self.assertIn("invalid_structured_output", prediction.failure)
        self.assertEqual({"theta0": 0.5, "theta1": 0.5}, dict(prediction.posterior))
        self.assertIsNotNone(prediction.telemetry.response_hash)

    def test_llm_completion_failure_retains_model_environment(self) -> None:
        prediction = LLMAttacker(FailingClient()).predict_hidden_state(
            context(), snapshot("ok")
        )
        self.assertFalse(prediction.valid)
        self.assertEqual("failed-model", prediction.telemetry.model_id)
        self.assertEqual("a" * 40, prediction.telemetry.model_revision)
        self.assertEqual("test-gpu", prediction.telemetry.metadata["gpu_name"])
        self.assertIn(
            "model_failure", prediction.telemetry.metadata["terminal_failure"]
        )

    def test_mock_is_structured_and_exact_protocol_is_prior_only(self) -> None:
        attacker = make_attacker("mock_llm")
        exact = replace(
            context(),
            runtime_name="silenttwin",
            feedback_protocol="genuine",
        )
        prediction = attacker.predict_hidden_state(exact, snapshot("ok"))
        self.assertTrue(prediction.valid)
        self.assertEqual({"theta0": 0.5, "theta1": 0.5}, dict(prediction.posterior))

    def test_black_box_has_no_secret_cold_start_and_enforces_template_split(self) -> None:
        attacker = BlackBoxAttacker()
        cold = attacker.predict_hidden_state(context(), snapshot("error", "GENERIC_FAILURE"))
        self.assertFalse(cold.valid)
        self.assertEqual("black_box_not_fitted", cold.failure)

        examples = (
            TrainingExample(snapshot("ok"), "theta0", "train-a", "runtime-a"),
            TrainingExample(
                snapshot("error", "GENERIC_FAILURE"),
                "theta1",
                "train-b",
                "runtime-a",
            ),
        )
        provenance = TrainingProvenance(
            dataset_revision="revision-1",
            split="train",
            template_ids=("train-a", "train-b"),
            runtime_configuration_hashes=("runtime-a",),
        )
        attacker.fit(examples, provenance=provenance)
        heldout = attacker.predict_hidden_state(
            context(template_id="heldout"),
            snapshot("error", "GENERIC_FAILURE"),
        )
        self.assertTrue(heldout.valid)
        self.assertEqual("theta1", heldout.prediction)
        leaked = attacker.predict_hidden_state(
            context(template_id="train-a"),
            snapshot("ok"),
        )
        self.assertFalse(leaked.valid)
        self.assertEqual("black_box_template_split_leakage", leaked.failure)

    def test_majority_aliases_are_available(self) -> None:
        self.assertEqual("majority_prior", make_attacker("majority").name)
        self.assertEqual("majority_prior", make_attacker("majority_prior").name)

    def test_local_model_client_is_lazy_and_configuration_only(self) -> None:
        config = LocalModelConfig.from_environment(
            {
                "MODEL_ID": "/models/local-checkpoint",
                "MODEL_REVISION": "a" * 40,
                "TOKENIZER_REVISION": "b" * 40,
                "MODEL_CACHE_DIR": "/models/cache",
                "DTYPE": "bfloat16",
                "MAX_NEW_TOKENS": "96",
                "TEMPERATURE": "0",
                "TOP_P": "1",
                "DECODING_SEED": "7",
                "BATCH_SIZE": "1",
                "MODEL_DEVICE": "cuda",
            }
        )
        client = LocalTransformersModelClient(config)
        self.assertIsNone(client._model)
        self.assertEqual("a" * 40, client.config.model_revision)
        self.assertEqual(7, client.config.decoding_seed)

    def test_local_checkpoint_fingerprint_changes_with_checkpoint_bytes(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "weights.bin"
            weights.write_bytes(b"first")
            first = fingerprint_local_checkpoint(root)
            weights.write_bytes(b"second")
            second = fingerprint_local_checkpoint(root)
        self.assertRegex(first, r"^sha256:[0-9a-f]{64}$")
        self.assertNotEqual(first, second)

        with self.assertRaisesRegex(ValueError, "exact"):
            LocalModelConfig(model_id="org/model", model_revision="main")

    def test_prepared_checkpoint_manifest_detects_metadata_change(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            cache = root / "cache"
            model.mkdir()
            weights = model / "weights.bin"
            weights.write_bytes(b"stable-weights")
            prepared = prepare_local_checkpoint_fingerprint(model, cache)
            manifest_path = checkpoint_fingerprint_manifest_path(model, cache)
            self.assertEqual(str(manifest_path), prepared["manifest_path"])
            config = LocalModelConfig(
                model_id=str(model),
                model_revision=str(prepared["model_revision"]),
                model_cache_dir=cache,
                device="cpu",
            )
            client = LocalTransformersModelClient(config)
            client._verify_local_identity()
            self.assertEqual("full_tree_sha256_audit", client._local_verification_mode)

            weights.write_bytes(b"changed-weights")
            with self.assertRaisesRegex(Exception, "changed after fingerprinting"):
                LocalTransformersModelClient(config)._verify_local_identity()


if __name__ == "__main__":
    unittest.main()
