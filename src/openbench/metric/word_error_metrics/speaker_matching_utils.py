# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

import numpy as np
from argmaxtools.utils import get_logger
from scipy import optimize


logger = get_logger(__name__)


def find_optimal_speaker_mapping(
    ref_speakers: list[str], hyp_speakers: list[str], matching_pairs: list[tuple[int, int]]
) -> dict[str, str]:
    """Find optimal speaker mapping using Hungarian algorithm.

    Args:
        ref_speakers: List of reference speaker labels
        hyp_speakers: List of hypothesis speaker labels
        matching_pairs: List of (ref_idx, hyp_idx) pairs that are aligned

    Returns:
        Dictionary mapping reference speakers to hypothesis speakers
    """
    # Get unique speakers
    unique_ref_speakers = sorted(set(ref_speakers))
    unique_hyp_speakers = sorted(set(hyp_speakers))

    # Handle edge cases
    if not unique_ref_speakers or not unique_hyp_speakers:
        return {}

    # Build cost matrix
    cost_matrix = np.zeros((len(unique_ref_speakers), len(unique_hyp_speakers)))

    # Convert to numpy arrays for faster operations
    matching_pairs = np.array(matching_pairs)
    ref_speakers_array = np.array(ref_speakers)
    hyp_speakers_array = np.array(hyp_speakers)

    # Calculate cost matrix efficiently
    for ref_idx, ref_spk in enumerate(unique_ref_speakers):
        for hyp_idx, hyp_spk in enumerate(unique_hyp_speakers):
            # Use boolean indexing to count matches
            ref_matches = ref_speakers_array[matching_pairs[:, 0]] == ref_spk
            hyp_matches = hyp_speakers_array[matching_pairs[:, 1]] == hyp_spk
            cost_matrix[ref_idx, hyp_idx] = np.sum(ref_matches & hyp_matches)

    # Find optimal speaker mapping using Hungarian algorithm
    row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix, maximize=True)
    speaker_mapping = {unique_ref_speakers[i]: unique_hyp_speakers[j] for i, j in zip(row_ind, col_ind)}

    logger.debug(f"Speaker mapping: {speaker_mapping}")
    return speaker_mapping


def concatenate_transcripts_by_speaker(
    ref_words: list[str],
    ref_speakers: list[str],
    hyp_words: list[str],
    hyp_speakers: list[str],
    speaker_mapping: dict[str, str],
) -> tuple[list[str], list[str]]:
    """Concatenate transcripts by speaker using optimal mapping.

    Args:
        ref_words: Reference words
        ref_speakers: Reference speaker labels
        hyp_words: Hypothesis words
        hyp_speakers: Hypothesis speaker labels
        speaker_mapping: Mapping from reference to hypothesis speakers

    Returns:
        Tuple of (concatenated_ref_words, concatenated_hyp_words)
    """
    # Group words by speaker for reference
    ref_by_speaker = {}
    for word, speaker in zip(ref_words, ref_speakers):
        if speaker not in ref_by_speaker:
            ref_by_speaker[speaker] = []
        ref_by_speaker[speaker].append(word)

    # Group words by speaker for hypothesis
    hyp_by_speaker = {}
    for word, speaker in zip(hyp_words, hyp_speakers):
        if speaker not in hyp_by_speaker:
            hyp_by_speaker[speaker] = []
        hyp_by_speaker[speaker].append(word)

    # Concatenate by mapped speakers
    concatenated_ref = []
    concatenated_hyp = []

    for ref_speaker in sorted(ref_by_speaker.keys()):
        # Get corresponding hypothesis speaker
        hyp_speaker = speaker_mapping.get(ref_speaker)
        if hyp_speaker is None:
            # If no mapping found, use the reference speaker as-is
            hyp_speaker = ref_speaker

        # Add reference words for this speaker
        concatenated_ref.extend(ref_by_speaker[ref_speaker])

        # Add hypothesis words for the mapped speaker (or empty if no mapping)
        if hyp_speaker in hyp_by_speaker:
            concatenated_hyp.extend(hyp_by_speaker[hyp_speaker])
        else:
            # If mapped speaker doesn't exist in hypothesis, add empty words
            concatenated_hyp.extend([""] * len(ref_by_speaker[ref_speaker]))

    return concatenated_ref, concatenated_hyp
