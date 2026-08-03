# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

from pathlib import Path

import numpy as np
import soundfile as sf
from typing_extensions import TypedDict

from ..pipeline_prediction import Transcript
from .dataset_base import BaseDataset, BaseSample


class SpeechGenerationExtraInfo(TypedDict, total=False):
    """Extra info for speech generation samples.

    For voice-cloning datasets (e.g. seedTTS), `ref_audio` points at the
    target-speaker prompt clip and `ref_text` is its transcript. The prototype
    pipeline consumes these to drive `tts-cli --mode voice_clone`. `sample_idx`
    carries the dataset's stable per-sample id (e.g. the source file name).

    SIM compares the generated clip against `sim_audio` when set (refclone /
    reference-length study: fixed REAL target wav). Otherwise it falls back to
    `ref_audio` / the sample waveform (seedTTS-style reconstruction).

    TED long-prompt WER rows optionally carry ``prompt_length`` (word count) and
    ``prompt_duration_minutes`` (nominal estimated talking time). Presence of
    ``prompt_duration_minutes`` selects the WER-only HF results schema.
    """

    language: str
    ref_audio: str
    ref_text: str
    sim_audio: str
    sample_idx: str
    prompt_length: int
    prompt_duration_minutes: int


class SpeechGenerationRow(TypedDict, total=False):
    """Expected row structure for speech generation.

    The text to synthesize (also the WER ground truth) comes from `prompt_text`
    (voice-clone datasets) or, for plain TTS datasets, the legacy `text` column.
    For voice cloning, `target_text` is the transcript of the reference clip
    (`audio`), used as `--ref-text`.
    """

    prompt_text: str
    text: str
    target_text: str
    language: str
    sample_idx: str
    prompt_length: int
    prompt_duration_minutes: int


class SpeechGenerationSample(BaseSample[Transcript, SpeechGenerationExtraInfo]):
    """Sample for speech-generation tasks.

    The reference `Transcript` is constructed from the prompt text. The
    pipeline synthesizes audio from this prompt and returns a
    `GeneratedAudio` prediction; the WER metric transcribes that audio
    and compares against this reference.
    """

    @property
    def text(self) -> str:
        """The original text prompt."""
        return self.reference.get_transcript_string()


class SpeechGenerationDataset(BaseDataset[SpeechGenerationSample]):
    """Dataset for speech-generation pipelines.

    Expects column: 'text' (the prompt string). No audio column is
    required — audio is produced by the pipeline, and a dummy waveform is
    supplied to satisfy the base sample structure.

    For voice-cloning eval datasets that also ship a reference clip in an
    `audio` column, that clip is loaded into the sample waveform (and its
    transcript exposed as `ref_text`) so cloning pipelines can use it as the
    target speaker. The runner still reads the generated-audio duration off the
    pipeline output, not this input audio.
    """

    # No hard-required column: the synthesis text may be `prompt_text` (voice-clone
    # datasets) or the legacy `text` (plain TTS datasets); validated in prepare_sample.
    _expected_columns: list[str] = []
    _sample_class = SpeechGenerationSample

    @staticmethod
    def _has_reference_audio(row: dict) -> bool:
        audio = row.get("audio")
        return isinstance(audio, dict) and audio.get("array") is not None

    def _extract_audio_info(self, row: dict) -> tuple[str, np.ndarray, int]:
        """Load a reference clip if present, else a placeholder waveform."""
        # Prefer an explicit id (refclone uses L{tag}); fall back to audio path stem.
        audio_name = f"sample_{row['idx']}"
        if "audio_name" in row and row["audio_name"]:
            audio_name = str(row["audio_name"])

        if self._has_reference_audio(row):
            audio = row["audio"]
            if not ("audio_name" in row and row["audio_name"]) and audio.get("path"):
                audio_name = Path(audio["path"]).stem
            return audio_name, np.asarray(audio["array"], dtype=np.float32), int(audio["sampling_rate"])

        dummy_waveform = np.zeros(1, dtype=np.float32)
        dummy_sample_rate = 16000
        return audio_name, dummy_waveform, dummy_sample_rate

    def prepare_sample(self, row: SpeechGenerationRow) -> tuple[Transcript, SpeechGenerationExtraInfo]:
        """Build the synthesis reference transcript and the per-sample extra info.

        The reference `Transcript` is the text to synthesize and the WER ground
        truth: it comes from `prompt_text` (voice-clone datasets) or the legacy
        `text` column.

        Voice-clone fields:
        * ``ref_text`` — transcript of the reference clip. An explicit
          ``ref_text`` column wins (refclone); otherwise voice-clone datasets
          provide it as ``target_text``; plain datasets with a reference clip
          reuse the synthesis text.
        * ``ref_audio`` — optional explicit path to the ICL prompt clip.
        * ``sim_audio`` — optional SIM yardstick path (refclone: real target wav).
        """
        synth_text = row.get("prompt_text")
        if synth_text is None:
            synth_text = row.get("text")
        if synth_text is None:
            raise ValueError(
                "Speech-generation dataset row must provide a 'prompt_text' (or legacy 'text') column "
                "with the text to synthesize."
            )
        reference = Transcript.from_words_info(words=synth_text.split())

        extra_info: SpeechGenerationExtraInfo = {}
        if row.get("language") is not None:
            extra_info["language"] = row["language"]
        # Stable per-sample id (e.g. source file name) for downstream result rows.
        if row.get("sample_idx") is not None:
            extra_info["sample_idx"] = str(row["sample_idx"])
        # TED long-prompt metadata → WER-only results schema (no SIM yardstick).
        if row.get("prompt_length") is not None:
            extra_info["prompt_length"] = int(row["prompt_length"])
        if row.get("prompt_duration_minutes") is not None:
            extra_info["prompt_duration_minutes"] = int(row["prompt_duration_minutes"])
        # Explicit ICL ref transcript (refclone: local datasets ship a ref_text column).
        ref_text = row.get("ref_text")
        if isinstance(ref_text, str) and ref_text.strip():
            extra_info["ref_text"] = ref_text.strip()
        elif self._has_reference_audio(row):
            # Voice-clone schema: the reference clip's transcript is `target_text`;
            # plain datasets with a reference clip reuse the synthesis text.
            extra_info["ref_text"] = row.get("target_text") or synth_text

        # Embedded SIM yardstick (voiceclone-eval `target_audio`): a held-out REAL
        # recording of the synthesis text. Materialize it to a temp WAV so SIM is
        # not computed against the clip the model conditioned on (same-channel /
        # ICL-continuation bias inflates that score). An explicit `sim_audio` path
        # column (refclone local datasets) takes precedence below.
        # Skip when the row is TED long-prompt (prompt_duration_minutes): there is
        # no held-out target audio for the synthesis text.
        target_audio = row.get("target_audio")
        if (
            row.get("prompt_duration_minutes") is None
            and isinstance(target_audio, dict)
            and target_audio.get("array") is not None
        ):
            sim_dir = Path("./temp_dataset_sim_audio")
            sim_dir.mkdir(parents=True, exist_ok=True)
            sim_name = str(row.get("sample_idx") or f"sample_{row.get('idx', 'unknown')}")
            sim_path = (sim_dir / f"{sim_name}.wav").resolve()
            sf.write(str(sim_path), np.asarray(target_audio["array"], dtype=np.float32), int(target_audio["sampling_rate"]))
            extra_info["sim_audio"] = str(sim_path)

        ref_audio = row.get("ref_audio")  # type: ignore[attr-defined]
        if isinstance(ref_audio, str) and ref_audio.strip():
            extra_info["ref_audio"] = ref_audio.strip()

        sim_audio = row.get("sim_audio")  # type: ignore[attr-defined]
        if isinstance(sim_audio, str) and sim_audio.strip() and row.get("prompt_duration_minutes") is None:
            extra_info["sim_audio"] = sim_audio.strip()

        return reference, extra_info
