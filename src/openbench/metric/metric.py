# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

from enum import Enum


class MetricOptions(Enum):
    # Diarization Error Rate
    # Ref: https://pyannote.github.io/pyannote-metrics/reference.html#pyannote.metrics.diarization.DiarizationErrorRate
    DER = "der"

    # Jaccard Error Rate
    # This is computed as 1 - Avg(Jaccard Index)
    # where the Jaccard Index is the intersection over union for two speaker
    # segments, one for the reference and one for the hypothesis.
    # While similar to DER, JER weighs every speaker equally regardless of their speech duration.
    # Ref: https://pyannote.github.io/pyannote-metrics/reference.html#pyannote.metrics.diarization.JaccardErrorRate
    JER = "jer"

    # Diarization purity (cluster purity)
    # A hypothesized annotation has perfect purity if all of its labels overlap only segments
    # which are members of a single reference label.
    # Ref: https://pyannote.github.io/pyannote-metrics/reference.html#pyannote.metrics.diarization.DiarizationPurity
    DIARIZATION_PURITY = "diarization_purity"

    # Compute diarization purity, coverage and return their F-score.
    # Ref: https://pyannote.github.io/pyannote-metrics/reference.html#pyannote.metrics.diarization.DiarizationPurityCoverageFMeasure
    DIARIZATION_PURITY_COVERAGE_FMEASURE = "diarization_purity_coverage_fmeasure"

    # Diarization homogeneity
    # Measures cluster homogeneity.
    # Ref: https://pyannote.github.io/pyannote-metrics/reference.html#pyannote.metrics.diarization.DiarizationHomogeneity
    DIARIZATION_HOMOGENEITY = "diarization_homogeneity"

    # Detection Error Rate
    # Ref: https://pyannote.github.io/pyannote-metrics/reference.html#pyannote.metrics.detection.DetectionErrorRate
    DETECTION_ERROR_RATE = "detection_error_rate"

    # Detection Cost Function
    # Ref: https://pyannote.github.io/pyannote-metrics/reference.html#pyannote.metrics.detection.DetectionCostFunction
    DETECTION_COST_FUNCTION = "detection_cost_function"

    # Detection Precision-Recall F-Measure
    # Ref: https://pyannote.github.io/pyannote-metrics/reference.html#pyannote.metrics.detection.DetectionPrecisionRecallFMeasure
    DETECTION_PRECISION_RECALL_FMEASURE = "detection_precision_recall_fmeasure"

    # Speaker Counting Error Rate
    SCER = "scer"

    # Speaker Count Mean Absolute Error
    SCMAE = "scmae"

    # Speaker Count Accuracy
    SCA = "sca"

    # Word Diarization Error Rate
    # Evaluates both transcription and speaker assignment accuracy at word level
    # Ref: https://arxiv.org/pdf/1907.05337
    WDER = "wder"

    # Word Error Rate
    # Evaluates transcription accuracy at word level
    # Ref: https://en.wikipedia.org/wiki/Word_error_rate
    WER = "wer"

    # Concatenated minimum-Permutation Word Error Rate
    # Evaluates multi-speaker transcription by finding the optimal speaker permutation
    # Ref: https://arxiv.org/abs/2004.09249
    CPWER = "cpwer"

    # Streaming Transcription Latency Based on Hypothesis(Unconfirmed) Transcript
    # Evaluates the Latency of Realtime Transcription
    # Time passed after an audio chunk is sent until its corresponding transcription is received.
    STREAMING_LATENCY = "streaming_latency"

    # Streaming Transcription Latency Based on Confirmed Transcript
    # Evaluates the Latency of Realtime Transcription
    # Time passed after an audio chunk is sent until its corresponding transcription is received.
    CONFIRMED_STREAMING_LATENCY = "confirmed_streaming_latency"

    # Streaming Transcription Latency Based on Confirmed Transcript and Model Word Timestamps
    # Evaluates the Latency of Realtime Transcription
    # Time passed after an audio chunk is sent until its corresponding transcription is received.
    MODELTIMESTAMP_STREAMING_LATENCY = "model_timestamp_streaming_latency"

    # Streaming Transcription Latency Based on Confirmed Transcript and Model Word Timestamps
    # Evaluates the Latency of Realtime Transcription
    # Time passed after an audio chunk is sent until its corresponding transcription is received.
    MODELTIMESTAMP_CONFIRMED_STRM_LATENCY = "model_timestamp_confirmed_streaming_latency"

    # Number of Corrections Metrics for Streaming Transcription
    # Evaluates the Number of Deletions Based on Previous Interim Transcript
    NUM_DELETIONS = "number_deletions"

    # Number of Corrections Metrics for Streaming Transcription
    # Evaluates the Number of Substitutions Based on Previous Interim Transcript
    NUM_SUBSTITUTIONS = "number_substitutions"

    # Number of Corrections Metrics for Streaming Transcription
    # Evaluates the Number of Insertions Based on Previous Interim Transcript
    NUM_INSERTIONS = "number_insertions"

    # Keyword Boosting Metrics
    # Evaluates precision, recall and F-score for keyword recognition
    KEYWORD_FSCORE = "keyword_fscore"
    KEYWORD_PRECISION = "keyword_precision"
    KEYWORD_RECALL = "keyword_recall"
