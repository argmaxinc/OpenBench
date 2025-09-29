# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

import jiwer
from argmaxtools.utils import get_logger
from pyannote.metrics.base import BaseMetric
from pydantic import BaseModel, Field

from ...pipeline_prediction import Transcript
from .text_normalizer import EnglishTextNormalizer


logger = get_logger(__name__)


class AlignmentMetrics(BaseModel):
    wer: float = Field(..., description="Word Error Rate")
    mer: float = Field(..., description="Match Error Rate")
    wil: float = Field(..., description="Word Information Loss")
    wip: float = Field(..., description="Word Information Preservation")
    hits: int = Field(..., description="Number of correct words")
    substitutions: int = Field(..., description="Number of substitutions")
    deletions: int = Field(..., description="Number of deletions")
    insertions: int = Field(..., description="Number of insertions")
    ops: list[list[jiwer.AlignmentChunk]] = Field(..., description="Alignment operations")
    truth: list[list[str]] = Field(..., description="Reference words")
    hypothesis: list[list[str]] = Field(..., description="Hypothesis words")


def parse_diarzed_words(transcript: Transcript) -> tuple[list[str], list[str] | None]:
    """Parse a list of words into text and speaker strings.

    If the transcript has no speakers, return None.
    """
    word_list = transcript.get_words()
    speaker_list = transcript.get_speakers()
    return word_list, speaker_list


class BaseWordErrorMetric(BaseMetric):
    """Base class for word error metrics."""

    def __init__(
        self,
        use_text_normalizer: bool = True,
        english_spelling_mapping: dict[str, str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(instantaneous=True)
        self.use_text_normalizer = use_text_normalizer
        # NOTE: Currently only English text normalizer is supported
        self.text_normalizer = EnglishTextNormalizer(english_spelling_mapping)

    def _supports_paired_evaluation(self) -> bool:
        return True

    def _get_word_error_metrics(
        self, reference: Transcript, hypothesis: Transcript
    ) -> tuple[
        jiwer.AlignmentChunk,
        tuple[list[str], list[str]],
        tuple[list[str] | None, list[str] | None],
    ]:
        ref_words, ref_speakers = parse_diarzed_words(reference)
        hyp_words, hyp_speakers = parse_diarzed_words(hypothesis)

        if self.use_text_normalizer:
            ref_words, ref_speakers = self.text_normalizer(
                words=ref_words,
                speakers=ref_speakers,
            )
            hyp_words, hyp_speakers = self.text_normalizer(
                words=hyp_words,
                speakers=hyp_speakers,
            )

        result = jiwer.compute_measures(
            truth=" ".join(ref_words),
            hypothesis=" ".join(hyp_words),
        )
        result = AlignmentMetrics(**result)

        # Get alignments
        return result.ops[0], (ref_words, hyp_words), (ref_speakers, hyp_speakers)
