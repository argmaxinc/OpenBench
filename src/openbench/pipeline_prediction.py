# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from argmaxtools.utils import get_logger
from pyannote.core import Annotation, Segment
from pyannote.database.util import load_rttm
from pydantic import BaseModel, Field


logger = get_logger(__name__)


# Diarization Prediction
class DiarizationAnnotation(Annotation):
    def to_annotation_file(self, output_dir: str, filename: str) -> str:
        path = os.path.join(output_dir, f"{filename}.rttm")
        with open(path, "w") as f:
            self.write_rttm(f)
        return path

    @classmethod
    def from_pyannote_annotation(cls, pyannote_annotation: Annotation) -> "DiarizationAnnotation":
        diarization_annotation = cls(pyannote_annotation.uri)
        for segment, _, speaker in pyannote_annotation.itertracks(yield_label=True):
            diarization_annotation[segment] = speaker
        return diarization_annotation

    @classmethod
    def load_annotation_file(cls, path: str) -> "DiarizationAnnotation":
        pyannote_annotation: dict[str, Annotation] = load_rttm(path)
        if len(pyannote_annotation) != 1:
            raise ValueError(f"Expected exactly one annotation in {path}, but got {len(pyannote_annotation)}")

        pyannote_annotation: Annotation = list(pyannote_annotation.values())[0]
        return cls.from_pyannote_annotation(pyannote_annotation)

    @property
    def timestamps_start(self) -> np.ndarray:
        return np.array([segment.start for segment, _, _ in self.itertracks(yield_label=True)])

    @property
    def timestamps_end(self) -> np.ndarray:
        return np.array([segment.end for segment, _, _ in self.itertracks(yield_label=True)])

    @property
    def speakers(self) -> np.ndarray:
        return np.array([speaker for _, _, speaker in self.itertracks(yield_label=True)])

    @property
    def num_speakers(self) -> int:
        return len(np.unique(self.speakers))

    def plot(self, ax: plt.Axes | None = None) -> plt.Axes:
        """Plot diarization annotation as a timeline with speaker segments.

        Args:
            ax: Optional matplotlib axes to plot on. If None, creates new figure/axes.

        Returns:
            The matplotlib axes object containing the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))

        # Get unique speakers and assign them y-positions and colors
        speakers = sorted(set(self.speakers))
        speaker_positions = {s: i for i, s in enumerate(speakers)}
        colors = plt.cm.rainbow(np.linspace(0, 1, len(speakers)))
        speaker_colors = dict(zip(speakers, colors))

        # Calculate total speaking time for each speaker
        speaker_durations = dict.fromkeys(speakers, 0)
        for segment, _, speaker in self.itertracks(yield_label=True):
            speaker_durations[speaker] += segment.duration

        # Plot segments for each speaker
        for segment, _, speaker in self.itertracks(yield_label=True):
            ax.barh(
                y=speaker_positions[speaker],
                width=segment.duration,
                left=segment.start,
                height=0.8,
                color=speaker_colors[speaker],
                label=f"{speaker} ({speaker_durations[speaker]:.2f}s)",
            )

        # Clean up duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        # Move legend outside the plot to the right and add title
        ax.legend(
            by_label.values(),
            by_label.keys(),
            title="Speakers",
            loc="center left",
            bbox_to_anchor=(1, 0.5),
        )

        # Customize appearance
        ax.set_yticks(list(speaker_positions.values()))
        ax.set_yticklabels(list(speaker_positions.keys()))
        ax.set_xlabel("Time (seconds)")
        ax.grid(True, axis="x", alpha=0.3)

        # Adjust layout to make room for legend
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)

        return ax

    def fill_small_gaps(self, min_active_offset: float = 0.0) -> "DiarizationAnnotation":
        """Fill small gaps between segments of the same speaker.

        Args:
            min_active_offset: Maximum gap duration (in seconds) to be filled.
                             If the gap between two segments is smaller than this,
                             they will be merged.

        Returns:
            A new DiarizationAnnotation with small gaps filled.
        """
        merged_annotation = DiarizationAnnotation(self.uri)

        for speaker_label in self.labels():
            speaker_annot = self.subset([speaker_label])
            segments = list(speaker_annot.itersegments())

            if not segments:
                continue

            current_segment = segments[0]
            for next_segment in segments[1:]:
                gap = next_segment.start - current_segment.end
                if gap <= min_active_offset:
                    # Merge segments
                    logger.debug(f"Speaker {speaker_label}: Merging segments {current_segment} and {next_segment}")
                    current_segment = Segment(current_segment.start, next_segment.end)
                else:
                    # Add current merged segment
                    merged_annotation[current_segment] = speaker_label
                    current_segment = next_segment

            # Add the last segment
            merged_annotation[current_segment] = speaker_label

        return merged_annotation


# ASR or Orchestration Prediction
class Word(BaseModel):
    word: str = Field(
        ...,
        description="The word in the transcription",
    )
    start: float | None = Field(
        None,
        description="The start time of the word in seconds",
    )
    end: float | None = Field(
        None,
        description="The end time of the word in seconds",
    )
    speaker: str | None = Field(
        None,
        description="The speaker of the word",
    )

    @classmethod
    def from_string(cls, word: str, speaker: str | None = None) -> "Word":
        return cls(word=word, speaker=speaker)


class Transcript(BaseModel):
    words: list[Word] = Field(
        ...,
        description="The words of the transcript",
    )

    @classmethod
    def from_words_info(
        cls,
        words: list[str],
        start: list[float] | None = None,
        end: list[float] | None = None,
        speaker: list[str] | None = None,
    ) -> "Transcript":
        if start is None:
            start = [None] * len(words)
        if end is None:
            end = [None] * len(words)
        if speaker is None:
            speaker = [None] * len(words)
        return cls(
            words=[
                Word(word=word, start=start, end=end, speaker=speaker)
                for word, start, end, speaker in zip(words, start, end, speaker)
            ]
        )

    def get_words(self) -> list[str]:
        return [word.word for word in self.words]

    def get_speakers(self) -> list[str] | None:
        return [word.speaker for word in self.words] if self.has_speakers else None

    def get_transcript_string(self) -> str:
        "Returns the transcript as a single string of words with spaces between them"
        return " ".join([word.word for word in self.words])

    def get_speakers_string(self) -> str:
        "Returns the speakers as a single string of speakers with spaces between them"
        return " ".join([word.speaker for word in self.words])

    @property
    def has_speakers(self) -> bool:
        return any(word.speaker is not None for word in self.words)

    def to_annotation_file(self, output_dir: str, filename: str) -> str:
        path = os.path.join(output_dir, f"{filename}.csv")
        data = defaultdict(list)
        for word in self.words:
            data["timestamp_start"].append(word.start)
            data["timestamp_end"].append(word.end)
            data["speaker"].append(word.speaker)
            data["word"].append(word.word)

        df = pd.DataFrame(data)
        df.to_csv(path, index=False)
        return path


# NOTE: StreamingTranscript is used only as output of pipelines. The reference for streaming transcript is of type Transcript.
class StreamingTranscript(BaseModel):
    transcript: str = Field(..., description="The final transcript")
    audio_cursor: list[float] | None = Field(None, description="The audio cursor in seconds")
    interim_results: list[str] | None = Field(None, description="The interim results")
    confirmed_audio_cursor: list[float] | None = Field(None, description="The confirmed audio cursor in seconds")
    confirmed_interim_results: list[str] | None = Field(None, description="The confirmed interim results")
    model_timestamps_hypothesis: list[list[dict[str, float]]] | None = Field(
        None,
        description="The model timestamps for the interim results as a list of lists of dictionaries with `start` and `end` keys",
    )
    model_timestamps_confirmed: list[list[dict[str, float]]] | None = Field(
        None,
        description="The model timestamps for the confirmed interim results as a list of lists of dictionaries with `start` and `end` keys",
    )

    def get_words(self) -> list[str]:
        return list(self.transcript.split())

    def get_speakers(self) -> list[str] | None:
        return None

    def to_annotation_file(self, output_dir: str, filename: str) -> str:
        path = os.path.join(output_dir, f"{filename}.json")
        data = {
            "transcript": self.transcript,
            "interim_results": self.interim_results,
            "audio_cursor": self.audio_cursor,
            "confirmed_audio_cursor": self.confirmed_audio_cursor,
            "confirmed_interim_results": self.confirmed_interim_results,
            "model_timestamps_hypothesis": self.model_timestamps_hypothesis,
            "model_timestamps_confirmed": self.model_timestamps_confirmed,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        return path
