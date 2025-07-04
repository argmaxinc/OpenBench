# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

from typing import Any

import numpy as np
from pydantic import model_validator

from ..pipeline_prediction import Transcript
from .dataset_base import BaseDataset, BaseSample


class OrchestrationSample(BaseSample[Transcript, dict[str, Any]]):
    """Orchestration sample with speaker validation."""
    
    @model_validator(mode='after')
    def validate_speaker_labels(self) -> 'OrchestrationSample':
        """Ensure transcript has speaker labels."""
        if not self.reference.has_speakers:
            raise ValueError("Orchestration samples require transcript with speaker labels")
        return self


class OrchestrationDataset(BaseDataset[OrchestrationSample]):
    """Dataset for orchestration pipelines."""
    _expected_columns = ["audio", "transcript", "word_speakers"]
    _sample_class = OrchestrationSample
    
    def prepare_sample(self, row: dict) -> tuple[Transcript, dict[str, Any]]:
        """Prepare transcript with speaker labels and extra info from dataset row."""
        transcript_words = row["transcript"]
        word_speakers = row["word_speakers"]
        
        if len(word_speakers) != len(transcript_words):
            raise ValueError("word_speakers and transcript must have same length")
        
        reference = Transcript.from_words_info(
            words=transcript_words,
            start=row.get("word_timestamps_start"),
            end=row.get("word_timestamps_end"),
            speaker=word_speakers
        )
        extra_info: dict[str, Any] = {}
        return reference, extra_info 