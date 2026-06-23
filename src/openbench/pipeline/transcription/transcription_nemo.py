"""
NeMo Context Biasing Pipeline for OpenBench

This pipeline implements CTC-based Word Spotter for context biasing
using NeMo ASR models (CTC and Hybrid Transducer-CTC).
"""

import json
import tempfile
from pathlib import Path
from typing import Callable

import librosa
import nemo.collections.asr as nemo_asr
import numpy as np
import torch
from argmaxtools.utils import get_logger
from nemo.collections.asr.models import (
    EncDecCTCModelBPE,
    EncDecHybridRNNTCTCModel,
)
from nemo.collections.asr.parts import context_biasing
from pydantic import Field

from ...pipeline import Pipeline, register_pipeline
from ...pipeline_prediction import Transcript
from ...types import PipelineType
from .common import TranscriptionConfig, TranscriptionOutput


logger = get_logger(__name__)

TEMP_AUDIO_DIR = Path("temp_audio_dir")


class NeMoTranscriptionPipelineConfig(TranscriptionConfig):
    """Configuration for NeMo Context Biasing Pipeline."""

    nemo_model_file: str = Field(description="Path to the .nemo file or name of pretrained model")
    decoder_type: str = Field(default="ctc", description="Decoder type: 'ctc' or 'rnnt'")
    device: str = Field(default="cpu", description="Device to load the model onto")
    acoustic_batch_size: int = Field(default=32, description="Batch size for acoustic model inference")
    beam_threshold: float = Field(default=5.0, description="Beam pruning threshold for CTC-WS decoding")
    context_score: float = Field(default=3.0, description="Per token weight for context biasing words")
    ctc_ali_token_weight: float = Field(
        default=0.6,
        description="Weight of CTC tokens to prevent false accept errors",
    )
    keyword_threshold: float = Field(
        default=-5.0,
        description="Threshold for keyword detection score",
    )
    blank_threshold: float = Field(
        default=0.8,
        description="Threshold for blank token probability",
    )
    non_blank_threshold: float = Field(
        default=0.001,
        description="Threshold for non-blank token probability",
    )
    intersection_threshold: float = Field(
        default=30.0,
        description="Threshold for intersection between spotted word and word from alignment (in percentage)",
    )
    spelling_separator: str = Field(
        default="_",
        description="Separator between word and its spellings",
    )


@register_pipeline
class NeMoTranscriptionPipeline(Pipeline):
    """NeMo Context Biasing Pipeline for keyword spotting."""

    _config_class = NeMoTranscriptionPipelineConfig
    pipeline_type = PipelineType.TRANSCRIPTION

    def __init__(self, config: NeMoTranscriptionPipelineConfig) -> None:
        super().__init__(config)

    def build_pipeline(self) -> Callable[[Path], TranscriptionOutput]:
        """Build the NeMo ASR pipeline with context biasing."""

        # Load NeMo ASR model
        if self.config.nemo_model_file.endswith(".nemo"):
            self.asr_model = nemo_asr.models.ASRModel.restore_from(
                self.config.nemo_model_file,
                map_location=torch.device(self.config.device),
            )
        else:
            logger.warning("nemo_model_file does not end with .nemo, trying to load pretrained model.")
            self.asr_model = nemo_asr.models.ASRModel.from_pretrained(
                self.config.nemo_model_file,
                map_location=torch.device(self.config.device),
            )

        if not isinstance(self.asr_model, (EncDecCTCModelBPE, EncDecHybridRNNTCTCModel)):
            raise ValueError("ASR model must be CTC BPE or Hybrid Transducer-CTC")

        # Set model to eval mode
        self.asr_model.eval()

        self.blank_idx = (
            self.asr_model.decoding.blank_id
            if isinstance(self.asr_model, EncDecCTCModelBPE)
            else self.asr_model.decoder.blank_idx
        )

        def transcribe(audio_path: Path) -> TranscriptionOutput:
            # Apply context biasing if keywords are available
            if hasattr(self, "context_graph") and self.context_graph is not None:
                pred_text = self._transcribe_with_context_biasing(audio_path)
            else:
                # No context biasing, use regular transcription
                pred_text = self._transcribe_regular(audio_path)

            # Remove temporary audio file
            audio_path.unlink(missing_ok=True)

            words = pred_text.split()

            return TranscriptionOutput(
                prediction=Transcript.from_words_info(
                    words=words,
                    speaker=None,
                    start=None,
                    end=None,
                )
            )

        return transcribe

    def _transcribe_regular(self, audio_path: Path) -> str:
        """Regular transcription without context biasing."""
        # Use NeMo's built-in transcribe method
        transcripts = self.asr_model.transcribe([str(audio_path)])
        if transcripts and len(transcripts) > 0:
            # Handle both string and Hypothesis object returns
            transcript = transcripts[0]
            if hasattr(transcript, "text"):
                return transcript.text
            else:
                return str(transcript)
        return ""

    def _transcribe_with_context_biasing(self, audio_path: Path) -> str:
        """Transcribe with context biasing using CTC Word Spotter."""

        # Get CTC logits first
        ctc_logprobs = self._get_ctc_logits(audio_path)

        # Apply context biasing
        ws_results = context_biasing.run_word_spotter(
            ctc_logprobs,
            self.context_graph,
            self.asr_model,
            blank_idx=self.blank_idx,
            beam_threshold=self.config.beam_threshold,
            cb_weight=self.config.context_score,
            ctc_ali_token_weight=self.config.ctc_ali_token_weight,
            keyword_threshold=self.config.keyword_threshold,
            blank_threshold=self.config.blank_threshold,
            non_blank_threshold=self.config.non_blank_threshold,
        )

        if not ws_results:
            # Fallback to regular transcription if word spotter fails
            return self._transcribe_regular(audio_path)

        # Get greedy predictions and merge with word spotter results
        preds = np.argmax(ctc_logprobs, axis=1)
        pred_text, _ = context_biasing.merge_alignment_with_ws_hyps(
            preds,
            self.asr_model,
            ws_results,
            decoder_type=self.config.decoder_type,
            intersection_threshold=self.config.intersection_threshold,
            blank_idx=self.blank_idx,
            print_stats=False,
        )
        return pred_text

    def _get_ctc_logits(self, audio_path: Path) -> np.ndarray:
        """Extract CTC logits from the model."""

        if isinstance(self.asr_model, EncDecCTCModelBPE):
            # For CTC models, get alignments
            hyp_results = self.asr_model.transcribe(
                [str(audio_path)],
                batch_size=self.config.acoustic_batch_size,
                return_hypotheses=True,
            )
            return hyp_results[0].alignments.cpu().numpy()
        else:
            # For Hybrid models, extract CTC logits manually
            with tempfile.TemporaryDirectory() as tmpdir:
                manifest_path = Path(tmpdir) / "manifest.json"

                audio_duration = librosa.get_duration(path=str(audio_path))
                with open(manifest_path, "w", encoding="utf-8") as fp:
                    entry = {
                        "audio_filepath": str(audio_path),
                        "duration": audio_duration,
                        "text": "",
                    }
                    fp.write(json.dumps(entry) + "\n")

                config = {
                    "paths2audio_files": [str(audio_path)],
                    "batch_size": self.config.acoustic_batch_size,
                    "temp_dir": tmpdir,
                    "num_workers": 0,
                    "channel_selector": None,
                    "augmentor": None,
                }
                datalayer = self.asr_model._setup_transcribe_dataloader(config)

                for test_batch in datalayer:
                    encoded, encoded_len = self.asr_model.forward(
                        input_signal=test_batch[0].to(self.asr_model.device),
                        input_signal_length=test_batch[1].to(self.asr_model.device),
                    )
                    ctc_dec_outputs = self.asr_model.ctc_decoder(encoder_output=encoded).cpu()
                    return ctc_dec_outputs[0, : encoded_len[0]].detach().cpu().numpy()

    def parse_input(self, input_sample) -> Path:
        """Override to extract keywords from sample before processing."""
        # Extract keywords from sample's extra_info if flag is enabled
        self.context_graph = None
        if self.config.use_keywords:
            keywords = input_sample.extra_info.get("dictionary", [])
            if keywords:
                context_transcripts = []
                for keyword in keywords:
                    word_tokenization = [self.asr_model.tokenizer.text_to_ids(keyword.lower())]
                    context_transcripts.append([keyword.lower(), word_tokenization])

                if context_transcripts:
                    self.context_graph = context_biasing.ContextGraphCTC(blank_id=self.blank_idx)
                    self.context_graph.add_to_graph(context_transcripts)

        # Warn if force_language is enabled (not currently supported)
        if self.config.force_language:
            logger.warning(
                f"{self.__class__.__name__} does not support language hinting. "
                "The force_language flag will be ignored."
            )

        return input_sample.save_audio(TEMP_AUDIO_DIR)

    def parse_output(self, output: TranscriptionOutput) -> TranscriptionOutput:
        return output
