# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""Unit tests for Oruk Fourier 2B engines, transcription pipelines, and aliases."""

from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from openbench.dataset import TranscriptionSample
from openbench.engine.oruk_engine import (
    DEFAULT_ORUK_API_BASE_URL,
    DEFAULT_ORUK_API_MODEL,
    DEFAULT_ORUK_HF_MODEL_ID,
    DEFAULT_ORUK_HF_REVISION,
    ORUK_HF_NATIVE_FILES,
    OrukApi,
    OrukFourierEngine,
    build_fourier_generation_kwargs,
    normalize_oruk_api_language,
    parse_qwen3_asr_text,
    split_audio_chunks,
)
from openbench.pipeline import PipelineRegistry
from openbench.pipeline.transcription.transcription_oruk import (
    OrukApiTranscriptionConfig,
    OrukApiTranscriptionPipeline,
    OrukFourierTranscriptionConfig,
    OrukFourierTranscriptionPipeline,
)
from openbench.pipeline_prediction import Transcript
from openbench.types import PipelineType


class TestOrukHelpers(unittest.TestCase):
    def test_normalize_oruk_api_language(self) -> None:
        self.assertIsNone(normalize_oruk_api_language(None))
        self.assertIsNone(normalize_oruk_api_language("   "))
        self.assertEqual(normalize_oruk_api_language("en"), "en")
        self.assertEqual(normalize_oruk_api_language("EN-us"), "en-US")
        self.assertEqual(normalize_oruk_api_language("en-gb"), "en-GB")
        self.assertEqual(normalize_oruk_api_language("English"), "english")
        self.assertEqual(normalize_oruk_api_language("AUTO"), "auto")
        with self.assertRaises(ValueError):
            normalize_oruk_api_language("fr")

    def test_parse_qwen3_asr_text(self) -> None:
        self.assertEqual(parse_qwen3_asr_text("language English<asr_text>Hello world."), "Hello world.")
        self.assertEqual(parse_qwen3_asr_text("  Plain transcript  "), "Plain transcript")
        self.assertEqual(parse_qwen3_asr_text("language English<asr_text>"), "")

    def test_split_audio_chunks(self) -> None:
        self.assertEqual(split_audio_chunks(np.array([], dtype=np.float32)), [])
        short = np.zeros(16_000 * 10, dtype=np.float32)
        self.assertEqual(len(split_audio_chunks(short, sample_rate=16_000, chunk_length_s=30.0)), 1)
        long_wav = np.zeros(16_000 * 65, dtype=np.float32)
        chunks = split_audio_chunks(long_wav, sample_rate=16_000, chunk_length_s=30.0)
        self.assertEqual([c.size for c in chunks], [16_000 * 30, 16_000 * 30, 16_000 * 5])
        with self.assertRaises(ValueError):
            split_audio_chunks(short, chunk_length_s=0.0)

    def test_build_fourier_generation_kwargs(self) -> None:
        self.assertEqual(build_fourier_generation_kwargs("published"), {})
        self.assertEqual(build_fourier_generation_kwargs("published", max_new_tokens=256), {"max_new_tokens": 256})
        self.assertEqual(
            build_fourier_generation_kwargs("greedy", max_new_tokens=128),
            {"do_sample": False, "num_beams": 1, "max_new_tokens": 128},
        )
        with self.assertRaises(ValueError):
            build_fourier_generation_kwargs("beam")  # type: ignore[arg-type]


class TestOrukApi(unittest.TestCase):
    def test_missing_api_key_raises(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                OrukApi()
            self.assertIn("ORUK_API_KEY", str(ctx.exception))

    @patch("openbench.engine.oruk_engine.OpenAI")
    def test_transcribe_sends_strict_schema_kwargs(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = types.SimpleNamespace(text="  Good morning everyone. ")
        mock_openai_cls.return_value = mock_client

        with patch.dict(os.environ, {"ORUK_API_KEY": "oruk_test_dummy"}, clear=True):
            api = OrukApi()
            self.assertEqual(api.base_url, DEFAULT_ORUK_API_BASE_URL)
            mock_openai_cls.assert_called_once_with(
                api_key="oruk_test_dummy",
                base_url=DEFAULT_ORUK_API_BASE_URL,
                timeout=300.0,
            )

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(b"RIFFstub")
                audio_path = Path(tmp.name)
            try:
                text = api.transcribe(audio_path, language="en-us")
            finally:
                audio_path.unlink(missing_ok=True)

        self.assertEqual(text, "Good morning everyone.")
        _, kwargs = mock_client.audio.transcriptions.create.call_args
        self.assertEqual(set(kwargs.keys()), {"file", "model", "response_format", "language"})
        self.assertEqual(kwargs["model"], DEFAULT_ORUK_API_MODEL)
        self.assertEqual(kwargs["response_format"], "json")
        self.assertEqual(kwargs["language"], "en-US")

    @patch("openbench.engine.oruk_engine.OpenAI")
    def test_custom_base_url_from_env(self, mock_openai_cls: MagicMock) -> None:
        with patch.dict(
            os.environ,
            {"ORUK_API_KEY": "oruk_test_dummy", "ORUK_BASE_URL": "https://staging.example/v1"},
            clear=True,
        ):
            api = OrukApi()
        self.assertEqual(api.base_url, "https://staging.example/v1")
        mock_openai_cls.assert_called_once_with(
            api_key="oruk_test_dummy",
            base_url="https://staging.example/v1",
            timeout=300.0,
        )


class _FakeBatch(dict):
    def to(self, device: object, dtype: object) -> "_FakeBatch":
        self.moved_to = (device, dtype)
        return self


class TestOrukFourierEngine(unittest.TestCase):
    def test_raises_clear_error_when_transformers_lacks_qwen3_asr(self) -> None:
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.__version__ = "4.48.2"  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            with self.assertRaises(ImportError) as ctx:
                OrukFourierEngine()
            self.assertIn("transformers>=5.21.0", str(ctx.exception))

    def test_downloads_pinned_root_files_and_transcribes_chunks_with_keywords(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_snapshot = MagicMock(return_value=tmpdir)

            fake_processor = MagicMock()

            def _apply_req(**kwargs: object) -> _FakeBatch:
                return _FakeBatch({"input_ids": torch.tensor([[1, 2, 3]])})

            fake_processor.apply_transcription_request.side_effect = _apply_req
            fake_processor.decode.side_effect = [
                ["language English<asr_text>first chunk"],
                ["second chunk"],
            ]

            fake_model = MagicMock()
            fake_model.dtype = torch.float32
            fake_model.to.return_value = fake_model
            fake_model.eval.return_value = fake_model
            fake_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

            fake_transformers = types.ModuleType("transformers")
            fake_transformers.__version__ = "5.24.1"  # type: ignore[attr-defined]
            fake_transformers.Qwen3ASRForConditionalGeneration = object  # type: ignore[attr-defined]
            fake_transformers.AutoProcessor = MagicMock()  # type: ignore[attr-defined]
            fake_transformers.AutoProcessor.from_pretrained.return_value = fake_processor
            fake_transformers.AutoModelForMultimodalLM = MagicMock()  # type: ignore[attr-defined]
            fake_transformers.AutoModelForMultimodalLM.from_pretrained.return_value = fake_model

            fake_hf_hub = types.ModuleType("huggingface_hub")
            fake_hf_hub.snapshot_download = mock_snapshot  # type: ignore[attr-defined]

            waveform = np.zeros(16_000 * 40, dtype=np.float32)

            with (
                patch.dict(sys.modules, {"transformers": fake_transformers, "huggingface_hub": fake_hf_hub}),
                patch("librosa.load", return_value=(waveform, 16_000)),
            ):
                engine = OrukFourierEngine(device="cpu", dtype="float32", decode_mode="published")
                text = engine.transcribe("clip.wav", prompt="Argmax, Fourier", language="en")

            self.assertEqual(text, "first chunk second chunk")
            mock_snapshot.assert_called_once_with(
                repo_id=DEFAULT_ORUK_HF_MODEL_ID,
                revision=DEFAULT_ORUK_HF_REVISION,
                allow_patterns=list(ORUK_HF_NATIVE_FILES),
                cache_dir=None,
            )
            self.assertIn("config.json", mock_snapshot.call_args.kwargs["allow_patterns"])
            self.assertEqual(fake_processor.apply_transcription_request.call_count, 2)
            first_call_kwargs = fake_processor.apply_transcription_request.call_args_list[0].kwargs
            self.assertEqual(first_call_kwargs["prompt"], "Argmax, Fourier")
            self.assertEqual(first_call_kwargs["language"], "en")


class TestOrukPipelinesAndAliases(unittest.TestCase):
    def test_aliases_registered(self) -> None:
        pipelines = PipelineRegistry.list_pipelines()
        self.assertIn("oruk-fourier-2b", pipelines)
        self.assertIn("oruk-fourier-2b-api", pipelines)
        self.assertTrue(PipelineRegistry.is_alias("oruk-fourier-2b"))
        self.assertTrue(PipelineRegistry.is_alias("oruk-fourier-2b-api"))

        hf_cls = PipelineRegistry.get_pipeline_class("oruk-fourier-2b")
        api_cls = PipelineRegistry.get_pipeline_class("oruk-fourier-2b-api")
        self.assertIs(hf_cls, OrukFourierTranscriptionPipeline)
        self.assertIs(api_cls, OrukApiTranscriptionPipeline)
        self.assertEqual(hf_cls.pipeline_type, PipelineType.TRANSCRIPTION)
        self.assertEqual(api_cls.pipeline_type, PipelineType.TRANSCRIPTION)

        hf_info = PipelineRegistry.get_alias_info("oruk-fourier-2b")
        self.assertEqual(hf_info.default_config["model_id"], DEFAULT_ORUK_HF_MODEL_ID)
        self.assertEqual(hf_info.default_config["revision"], DEFAULT_ORUK_HF_REVISION)

        api_info = PipelineRegistry.get_alias_info("oruk-fourier-2b-api")
        self.assertEqual(api_info.default_config["model_version"], DEFAULT_ORUK_API_MODEL)

    @patch("openbench.pipeline.transcription.transcription_oruk.OrukApi")
    def test_api_pipeline_end_to_end_and_temp_cleanup(self, mock_api_cls: MagicMock) -> None:
        mock_api = MagicMock()
        mock_api.transcribe.return_value = "hello openbench world"
        mock_api_cls.return_value = mock_api

        pipeline = OrukApiTranscriptionPipeline(OrukApiTranscriptionConfig(force_language=True, use_keywords=False))
        sample = TranscriptionSample(
            audio_name="sample_0",
            waveform=np.zeros(16_000, dtype=np.float32),
            sample_rate=16_000,
            reference=Transcript.from_words_info(words=["hello", "openbench", "world"]),
            extra_info={"language": "en"},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("openbench.pipeline.transcription.transcription_oruk.TEMP_AUDIO_DIR", Path(tmpdir)):
                result = pipeline(sample)
        self.assertEqual(result.prediction.get_words(), ["hello", "openbench", "world"])
        called_path = mock_api.transcribe.call_args.args[0]
        self.assertFalse(Path(called_path).exists())
        self.assertEqual(mock_api.transcribe.call_args.kwargs["language"], "en")

    @patch("openbench.pipeline.transcription.transcription_oruk.OrukFourierEngine")
    def test_hf_pipeline_passes_keywords_and_cleans_up_on_error(self, mock_engine_cls: MagicMock) -> None:
        mock_engine = MagicMock()
        mock_engine.transcribe.return_value = "Oruk Fourier benchmark"
        mock_engine_cls.return_value = mock_engine

        pipeline = OrukFourierTranscriptionPipeline(
            OrukFourierTranscriptionConfig(use_keywords=True, force_language=True)
        )
        sample = TranscriptionSample(
            audio_name="sample_kw",
            waveform=np.zeros(16_000, dtype=np.float32),
            sample_rate=16_000,
            reference=Transcript.from_words_info(words=["Oruk", "Fourier", "benchmark"]),
            extra_info={"language": "en", "dictionary": ["Oruk", "Fourier"]},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("openbench.pipeline.transcription.transcription_oruk.TEMP_AUDIO_DIR", Path(tmpdir)):
                result = pipeline(sample)
                self.assertEqual(result.prediction.get_words(), ["Oruk", "Fourier", "benchmark"])
                self.assertEqual(mock_engine.transcribe.call_args.kwargs["prompt"], "Oruk, Fourier")
                self.assertEqual(mock_engine.transcribe.call_args.kwargs["language"], "en")
                self.assertFalse(Path(mock_engine.transcribe.call_args.args[0]).exists())

                # Verify temp audio file is still deleted when engine raises.
                mock_engine.transcribe.side_effect = RuntimeError("boom")
                with self.assertRaises(RuntimeError):
                    pipeline(sample)
                self.assertFalse(Path(mock_engine.transcribe.call_args.args[0]).exists())


if __name__ == "__main__":
    unittest.main()
