# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""Oruk Fourier 2B engines (local Hugging Face Transformers and hosted API)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
from argmaxtools.utils import get_logger
from openai import OpenAI


logger = get_logger(__name__)

DEFAULT_ORUK_HF_MODEL_ID = "oruk/oruk-fourier-2b"
DEFAULT_ORUK_HF_REVISION = "8c2a8871f1304aa2f634da4fb95e24438c4945b9"
DEFAULT_ORUK_API_BASE_URL = "https://speech-api.oruk.ai/v1"
DEFAULT_ORUK_API_MODEL = "oruk-fourier-2b"
TARGET_SAMPLE_RATE = 16_000

# Strictly the 7 native root checkpoint files (skips auxiliary GGUF/ONNX/MLX subfolders
# while fetching repo-root config.json directly from Hugging Face).
ORUK_HF_NATIVE_FILES: tuple[str, ...] = (
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)

ORUK_API_LANGUAGE_MAP: dict[str, str] = {
    "en": "en",
    "en-us": "en-US",
    "en-gb": "en-GB",
    "english": "english",
    "auto": "auto",
}


def normalize_oruk_api_language(language: str | None) -> str | None:
    """Validate and normalize an optional language hint for Oruk Fourier-2B API."""
    if language is None:
        return None
    stripped = language.strip()
    if not stripped:
        return None
    normalized = ORUK_API_LANGUAGE_MAP.get(stripped.lower())
    if normalized is None:
        accepted = ", ".join(ORUK_API_LANGUAGE_MAP.values())
        raise ValueError(f"Oruk Fourier-2B is English-only; unsupported language {language!r} (accepted: {accepted})")
    return normalized


def parse_qwen3_asr_text(raw_text: str) -> str:
    """Strip an optional Qwen3-ASR ``language <Name><asr_text>`` prefix."""
    text = raw_text.strip()
    if "<asr_text>" in text:
        text = text.rsplit("<asr_text>", 1)[-1]
    return text.strip()


def split_audio_chunks(
    waveform: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    chunk_length_s: float = 30.0,
) -> list[np.ndarray]:
    """Split a mono waveform into sequential chunks of at most ``chunk_length_s`` seconds."""
    if chunk_length_s <= 0:
        raise ValueError(f"chunk_length_s must be positive, got {chunk_length_s}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    if waveform.size == 0:
        return []
    max_samples = max(1, int(round(chunk_length_s * sample_rate)))
    if waveform.size <= max_samples:
        return [waveform]
    return [waveform[start : start + max_samples] for start in range(0, waveform.size, max_samples)]


def build_fourier_generation_kwargs(
    decode_mode: Literal["published", "greedy"] = "published",
    max_new_tokens: int | None = None,
) -> dict[str, Any]:
    """Build ``model.generate`` kwargs matching published (3-beam) or greedy decoding."""
    if decode_mode not in {"published", "greedy"}:
        raise ValueError(f"Unsupported decode_mode={decode_mode!r}; expected 'published' or 'greedy'")

    kwargs: dict[str, Any] = {}
    if decode_mode == "greedy":
        kwargs["do_sample"] = False
        kwargs["num_beams"] = 1
    if max_new_tokens is not None:
        kwargs["max_new_tokens"] = int(max_new_tokens)
    return kwargs


def _resolve_torch_device(torch_module: Any, requested: str | None) -> Any:
    if requested is not None:
        return torch_module.device(requested)
    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return torch_module.device("mps")
    return torch_module.device("cpu")


def _resolve_torch_dtype(torch_module: Any, device: Any, requested: str) -> Any:
    if requested == "auto":
        return torch_module.bfloat16 if device.type in {"cuda", "mps"} else torch_module.float32
    dtype_map = {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }
    if requested not in dtype_map:
        raise ValueError(f"Unsupported dtype={requested!r}; expected one of auto, bfloat16, float16, float32")
    return dtype_map[requested]


class OrukApi:
    """Hosted Oruk Speech API client for OpenAI-compatible transcription."""

    def __init__(
        self,
        model: str = DEFAULT_ORUK_API_MODEL,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        self.model = model
        resolved_key = api_key or os.getenv("ORUK_API_KEY")
        if not resolved_key:
            raise ValueError("`ORUK_API_KEY` is not set")

        resolved_base_url = base_url or os.getenv("ORUK_BASE_URL") or DEFAULT_ORUK_API_BASE_URL
        self.base_url = resolved_base_url
        self.client = OpenAI(api_key=resolved_key, base_url=resolved_base_url, timeout=timeout)

    def transcribe(self, audio_path: Path | str, language: str | None = None) -> str:
        """Transcribe a local audio file via ``POST /v1/audio/transcriptions``."""
        path = Path(audio_path)
        normalized_language = normalize_oruk_api_language(language)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "response_format": "json",
        }
        if normalized_language is not None:
            kwargs["language"] = normalized_language

        with path.open("rb") as audio_file:
            kwargs["file"] = audio_file
            response = self.client.audio.transcriptions.create(**kwargs)

        if isinstance(response, str):
            return response.strip()
        return (getattr(response, "text", "") or "").strip()


class OrukFourierEngine:
    """Local Hugging Face Transformers engine for ``oruk/oruk-fourier-2b`` (Qwen3-ASR)."""

    def __init__(
        self,
        model_id: str = DEFAULT_ORUK_HF_MODEL_ID,
        revision: str | None = DEFAULT_ORUK_HF_REVISION,
        device: str | None = None,
        dtype: Literal["auto", "bfloat16", "float16", "float32"] = "auto",
        decode_mode: Literal["published", "greedy"] = "published",
        max_new_tokens: int | None = None,
        chunk_length_s: float = 30.0,
        cache_dir: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.decode_mode = decode_mode
        self.max_new_tokens = max_new_tokens
        self.chunk_length_s = chunk_length_s
        self.cache_dir = cache_dir
        self._gen_kwargs = build_fourier_generation_kwargs(
            decode_mode=decode_mode,
            max_new_tokens=max_new_tokens,
        )

        try:
            import torch
            import transformers
        except ImportError as exc:
            raise ImportError("Local Oruk Fourier 2B inference requires `torch` and `transformers>=5.21.0`.") from exc

        if not hasattr(transformers, "Qwen3ASRForConditionalGeneration"):
            installed = getattr(transformers, "__version__", "unknown")
            raise ImportError(
                f"Installed transformers ({installed}) does not include Qwen3ASRForConditionalGeneration. "
                "Install `transformers>=5.21.0` for local `oruk/oruk-fourier-2b` inference, "
                "or use the `oruk-fourier-2b-api` pipeline."
            )

        self._torch = torch
        local_dir = self._resolve_checkpoint_dir()
        self.local_dir = local_dir
        self.device = _resolve_torch_device(torch, device)
        self.dtype = _resolve_torch_dtype(torch, self.device, dtype)

        logger.info(f"Loading Oruk Fourier 2B from '{local_dir}' on device '{self.device}' (dtype={self.dtype})")
        self.processor = transformers.AutoProcessor.from_pretrained(str(local_dir), local_files_only=True)
        self.model = (
            transformers.AutoModelForMultimodalLM.from_pretrained(
                str(local_dir),
                local_files_only=True,
                dtype=self.dtype,
            )
            .to(self.device)
            .eval()
        )

    def _resolve_checkpoint_dir(self) -> Path:
        candidate = Path(self.model_id)
        if candidate.is_dir():
            return candidate

        from huggingface_hub import snapshot_download

        downloaded = snapshot_download(
            repo_id=self.model_id,
            revision=self.revision,
            allow_patterns=list(ORUK_HF_NATIVE_FILES),
            cache_dir=self.cache_dir,
        )
        return Path(downloaded)

    def _transcribe_chunk(
        self,
        chunk: np.ndarray,
        prompt: str | None = None,
        language: str | None = None,
    ) -> str:
        request_kwargs: dict[str, Any] = {
            "audio": chunk,
            "return_tensors": "pt",
        }
        if language is not None:
            request_kwargs["language"] = language
        if prompt is not None:
            request_kwargs["prompt"] = prompt

        inputs = self.processor.apply_transcription_request(**request_kwargs)
        inputs = inputs.to(self.device, self.model.dtype)
        with self._torch.inference_mode():
            output_ids = self.model.generate(**inputs, **self._gen_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, prompt_len:]
        decoded = self.processor.decode(generated_ids, return_format="transcription_only")
        if isinstance(decoded, list):
            raw = decoded[0] if decoded else ""
        else:
            raw = str(decoded)
        return parse_qwen3_asr_text(raw)

    def transcribe(
        self,
        audio_path: Path | str,
        prompt: str | None = None,
        language: str | None = None,
    ) -> str:
        """Load audio at 16 kHz mono, chunk to <= ``chunk_length_s``, and transcribe."""
        import librosa

        waveform, _ = librosa.load(str(audio_path), sr=TARGET_SAMPLE_RATE, mono=True)
        waveform = np.asarray(waveform, dtype=np.float32)
        chunks = split_audio_chunks(
            waveform,
            sample_rate=TARGET_SAMPLE_RATE,
            chunk_length_s=self.chunk_length_s,
        )
        if not chunks:
            return ""

        parts = [text for chunk in chunks if (text := self._transcribe_chunk(chunk, prompt=prompt, language=language))]
        return " ".join(parts).strip()
