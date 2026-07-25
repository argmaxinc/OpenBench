# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""Windowed speaker similarity (SIM-windowed) metric.

Plain SIM embeds the *whole* generated clip once, so a clone that drifts off
the target voice halfway through can still score well on the strength of its
good half. SIM-windowed instead slides a fixed window over the generated audio,
embeds each window separately, and compares every window against the (whole)
reference embedding. Per sample it reports the windowed **mean** as the metric
value and logs mean / variance / min / max across windows; the min in
particular surfaces localized identity drift that whole-clip SIM averages away.

Aggregates across samples: the global metric value is the mean of per-sample
windowed means (with :attr:`variance` across samples, mirroring plain SIM), and
:attr:`window_variance_mean`, :attr:`window_min`, :attr:`window_max` expose the
mean within-sample window variance and the extreme window scores seen anywhere
in the run.

The default 8 s window matches the segment scale the WavLM-SV speaker
encoder is tuned for (short windows under ~4 s score noticeably lower and
noisier); halve/double via ``-mc`` when hunting shorter artifacts.

Configuration (via ``-mc``, using the metric alias as prefix)::

    -mc sim-windowed.window_seconds=8.0
    -mc sim-windowed.hop_seconds=4.0
    -mc sim-windowed.checkpoint=hf://.../wavlm_large_finetune.pth  # same as sim

Windows shorter than ``min_window_seconds`` are only evaluated when the clip is
too short to yield any full window (then the whole clip is one window).
"""

import librosa
import torch
import torch.nn.functional as F
from pyannote.metrics.types import Details, MetricComponents
from torchaudio.transforms import Resample

from argmaxtools.utils import get_logger

from ...types import PipelineType
from ..metric import MetricOptions
from ..registry import MetricRegistry
from .sim_metric import SpeechGenerationSpeakerSimilarity


logger = get_logger(__name__)


@MetricRegistry.register_metric(PipelineType.SPEECH_GENERATION, MetricOptions.SIM_WINDOWED)
class SpeechGenerationWindowedSpeakerSimilarity(SpeechGenerationSpeakerSimilarity):
    """Windowed SIM for speech-generation pipelines.

    Accepts the same constructor kwargs as plain SIM (``model_name``,
    ``checkpoint``, ``use_gpu``, ``device``) plus the windowing parameters
    ``window_seconds`` / ``hop_seconds`` / ``min_window_seconds``.
    """

    def __init__(
        self,
        window_seconds: float = 8.0,
        hop_seconds: float = 4.0,
        min_window_seconds: float = 2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.window_seconds = float(window_seconds)
        self.hop_seconds = float(hop_seconds)
        self.min_window_seconds = float(min_window_seconds)
        if self.window_seconds <= 0 or self.hop_seconds <= 0:
            raise ValueError("window_seconds and hop_seconds must be positive")
        # Window extremes across the whole run. Tracked outside the pyannote
        # component sums because min/max do not accumulate additively.
        self._run_window_min: float | None = None
        self._run_window_max: float | None = None

    @classmethod
    def metric_name(cls) -> str:
        return "windowed speaker similarity"

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return [
            "wsim_mean_sum",       # sum of per-sample windowed means
            "wsim_mean_sq_sum",    # sum of squared per-sample windowed means
            "wsim_var_sum",        # sum of per-sample within-window variances
            "window_count",        # total windows evaluated
            "count",               # samples evaluated
        ]

    # -- windowed scoring ------------------------------------------------------

    def _load_16k(self, audio_path: str) -> torch.Tensor:
        """Load an audio file as a mono 16 kHz waveform tensor ``(1, samples)``."""
        wav, sr = librosa.load(audio_path, sr=None, mono=False)
        if wav.ndim == 2:
            wav = wav[0, :]
        wav = torch.from_numpy(wav).unsqueeze(0).float()
        wav = Resample(orig_freq=sr, new_freq=self.SAMPLE_RATE)(wav)
        if self.use_gpu:
            wav = wav.cuda(self.device)
        return wav

    def _embed_waveform(self, wav_16k: torch.Tensor) -> torch.Tensor:
        model = self._get_model()
        with torch.no_grad():
            return model(wav_16k)

    def window_scores(self, generated_audio: str, reference_audio: str) -> list[float]:
        """Cosine similarity of each generated-audio window vs the whole reference."""
        return [score for _, score in self.window_scores_with_starts(generated_audio, reference_audio)]

    def window_scores_with_starts(self, generated_audio: str, reference_audio: str) -> list[tuple[float, float]]:
        """``(window start in seconds, similarity vs the whole reference)`` per window.

        The start times make a low-scoring window locatable in the generated clip,
        which is what turns a bad score into something you can listen to.
        """
        emb_reference = self._embed_waveform(self._load_16k(reference_audio))
        wav = self._load_16k(generated_audio)
        total = wav.shape[-1]
        win = int(self.window_seconds * self.SAMPLE_RATE)
        hop = int(self.hop_seconds * self.SAMPLE_RATE)
        min_win = int(self.min_window_seconds * self.SAMPLE_RATE)

        starts = list(range(0, max(total - win, 0) + 1, hop))
        windows = [(s, wav[:, s : s + win]) for s in starts]
        # Cover the tail beyond the last full window when it is long enough to
        # embed meaningfully; a clip shorter than one window is itself the only
        # window (degenerates to plain SIM).
        tail_start = starts[-1] + hop if windows else 0
        if total - tail_start >= min_win or not windows:
            windows.append((tail_start, wav[:, tail_start:]))

        scored = []
        for start, w in windows:
            emb = self._embed_waveform(w)
            scored.append((start / self.SAMPLE_RATE, F.cosine_similarity(emb, emb_reference).item()))
        return scored

    # -- BaseMetric interface --------------------------------------------------

    def compute_components(self, reference, hypothesis, **kwargs) -> Details:
        generated_audio = hypothesis.audio_path
        reference_audio = (
            getattr(hypothesis, "reference_audio_path", None)
            or kwargs.get("reference_audio")
            or kwargs.get("ref_audio")
        )
        if not reference_audio:
            raise ValueError(
                "SIM-windowed requires a target-speaker reference clip, but none was found. "
                "Voice-clone pipelines must set `reference_audio_path` on the prediction "
                "(or pass reference_audio=... to the metric call)."
            )

        scored = self.window_scores_with_starts(generated_audio, str(reference_audio))
        scores = [score for _, score in scored]
        n = len(scores)
        mean = sum(scores) / n
        var = sum((s - mean) ** 2 for s in scores) / n
        min_start, lo = min(scored, key=lambda window: window[1])
        hi = max(scores)

        self._run_window_min = lo if self._run_window_min is None else min(self._run_window_min, lo)
        self._run_window_max = hi if self._run_window_max is None else max(self._run_window_max, hi)
        logger.info(
            "SIM-windowed sample: mean=%.4f var=%.5f min=%.4f max=%.4f over %d windows "
            "(window=%.1fs hop=%.1fs)",
            mean, var, lo, hi, n, self.window_seconds, self.hop_seconds,
        )

        return {
            "wsim_mean_sum": mean,
            "wsim_mean_sq_sum": mean * mean,
            "wsim_var_sum": var,
            "window_count": float(n),
            "count": 1.0,
            # Not declared in metric_components(), so these are never accumulated
            # into run totals — they ride along in the per-sample detailed output
            # (W&B task table) to make single-sample drift analyzable.
            "wsim_min": lo,
            "wsim_max": hi,
            "wsim_min_start": min_start,
        }

    def compute_metric(self, detail: Details) -> float:
        count = detail["count"]
        if count == 0:
            return 0.0
        return detail["wsim_mean_sum"] / count

    @property
    def variance(self) -> float:
        """Population variance of per-sample windowed means across the run."""
        count = self.accumulated_["count"]
        if count == 0:
            return 0.0
        mean = self.accumulated_["wsim_mean_sum"] / count
        return self.accumulated_["wsim_mean_sq_sum"] / count - mean * mean

    @property
    def window_variance_mean(self) -> float:
        """Mean within-sample window variance across the run."""
        count = self.accumulated_["count"]
        if count == 0:
            return 0.0
        return self.accumulated_["wsim_var_sum"] / count

    @property
    def window_min(self) -> float | None:
        """Lowest single-window SIM observed anywhere in the run."""
        return self._run_window_min

    @property
    def window_max(self) -> float | None:
        """Highest single-window SIM observed anywhere in the run."""
        return self._run_window_max
