# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""Pipeline alias registrations for common configurations."""

import os

from .diarization import (
    AWSTranscribePipeline,
    PicovoicePipeline,
    PyannoteApiPipeline,
    PyAnnotePipeline,
    SpeakerKitPipeline,
)
from .orchestration import (
    WhisperXPipeline,
)
from .pipeline_registry import PipelineRegistry
from .streaming_transcription import (
    DeepgramStreamingPipeline,
    FireworksStreamingPipeline,
    GladiaStreamingPipeline,
    OpenAIStreamingPipeline,
)
from .transcription import (
    SpeechAnalyzerPipeline,
    WhisperKitTranscriptionPipeline,
)


def register_pipeline_aliases() -> None:
    """Register all pipeline aliases with their configurations."""

    ################# DIARIZATION PIPELINES #################
    PipelineRegistry.register_alias(
        "aws-diarization",
        AWSTranscribePipeline,
        default_config={
            "out_dir": "./aws_diarization_results",
            "bucket_name": "diarization-benchmarks",
            "region_name": "us-east-2",
            "max_speakers": 30,
            "num_worker_processes": 8,
            "per_worker_chunk_size": 1,
        },
        description="AWS Transcribe with speaker diarization. Requires AWS credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) and S3 bucket setup.",
    )

    PipelineRegistry.register_alias(
        "pyannote",
        PyAnnotePipeline,
        default_config={
            "out_dir": "./pyannote_logs",
            "num_speakers": None,
            "min_speakers": None,
            "max_speakers": None,
            "use_oracle_clustering": False,
            "use_oracle_segmentation": False,
            "use_float16": True,
        },
        description="Pyannote open-source speaker diarization pipeline.",
    )

    PipelineRegistry.register_alias(
        "pyannote-api",
        PyannoteApiPipeline,
        default_config={
            "out_dir": "./pyannoteapi",
            "timeout": 3600,
            "request_buffer": 30,
        },
        description="Pyannote API speaker diarization pipeline. Requires API key from https://www.pyannote.ai/. Set `PYANNOTE_API_KEY` env var.",
    )

    PipelineRegistry.register_alias(
        "speakerkit",
        SpeakerKitPipeline,
        default_config={
            "out_dir": "./speakerkit-report",
            "cli_path": os.getenv("SPEAKERKIT_CLI_PATH"),
        },
        description="SpeakerKit speaker diarization pipeline. Requires CLI installation and API key. Set `SPEAKERKIT_CLI_PATH` and `SPEAKERKIT_API_KEY` env vars. For access to the CLI binary contact speakerkitpro@argmaxinc.com",
    )

    PipelineRegistry.register_alias(
        "picovoice-diarization",
        PicovoicePipeline,
        default_config={
            "out_dir": "./picovoice_logs",
        },
        description="Picovoice diarization pipeline. Requires API key from https://www.picovoice.ai/. Set `PICOVOICE_API_KEY` env var.",
    )

    ################# ORCHESTRATION PIPELINES #################

    PipelineRegistry.register_alias(
        "whisperx",
        WhisperXPipeline,
        default_config={
            "out_dir": "./whisperx_output",
            "model_name": "tiny",
            "device": "cpu",
            "compute_type": "int8",
            "batch_size": 16,
            "threads": 8,
        },
        description="WhisperX diarized transcription pipeline from https://github.com/m-bain/whisperX",
    )

    ################# TRANSCRIPTION PIPELINES #################

    PipelineRegistry.register_alias(
        "whisperkit-tiny",
        WhisperKitTranscriptionPipeline,
        default_config={
            "model_version": "tiny",
            "word_timestamps": True,
            "chunking_strategy": "vad",
        },
        description="WhisperKit transcription pipeline (open-source version) using the tiny version of the model. Requires Swift and Xcode installed.",
    )

    PipelineRegistry.register_alias(
        "whisperkit-large-v3",
        WhisperKitTranscriptionPipeline,
        default_config={
            "model_version": "large-v3",
            "word_timestamps": True,
            "chunking_strategy": "vad",
        },
        description="WhisperKit transcription pipeline (open-source version) using the large-v3 version of the model. Requires Swift and Xcode installed.",
    )

    PipelineRegistry.register_alias(
        "whisperkit-large-v3-turbo",
        WhisperKitTranscriptionPipeline,
        default_config={
            "model_version": "large-v3-v20240930",
            "word_timestamps": True,
            "chunking_strategy": "vad",
        },
        description="WhisperKit transcription pipeline (open-source version) using the large-v3-v20240930 version of the model (which is the same as large-v3-turbo from OpenAI). Requires Swift and Xcode installed.",
    )

    PipelineRegistry.register_alias(
        "speech-analyzer",
        SpeechAnalyzerPipeline,
        default_config={
            "clone_dir": "./speech_analyzer_repo",
        },
        description="Speech Analyzer transcription pipeline (open-source version). Requires Swift and Xcode installed.",
    )

    ################# STREAMING TRANSCRIPTION PIPELINES #################

    PipelineRegistry.register_alias(
        "deepgram-streaming",
        DeepgramStreamingPipeline,
        default_config={
            "sample_rate": 16000,
            "channels": 1,
            "sample_width": 2,
            "realtime_resolution": 0.02,
            "model_version": "nova-3",
            "endpoint_url": "wss://api.deepgram.com/v1/listen?model={model_version}&channels={channels}&sample_rate={sample_rate}&encoding=linear16&interim_results=true",
        },
        description="Deepgram streaming transcription pipeline. Requires API key from https://www.deepgram.com/. Set `DEEPGRAM_API_KEY` env var.",
    )

    PipelineRegistry.register_alias(
        "fireworks-streaming",
        FireworksStreamingPipeline,
        default_config={
            "sample_rate": 16000,
            "channels": 1,
            "sample_width": 2,
            "chunksize_ms": 50,
            "endpoint_url": "ws://audio-streaming.us-virginia-1.direct.fireworks.ai/v1/audio/transcriptions/streaming",
            "model": "whisper-v3-turbo",
        },
        description="Fireworks streaming transcription pipeline. Requires API key from https://www.fireworks.ai/. Set `FIREWORKS_API_KEY` env var.",
    )

    PipelineRegistry.register_alias(
        "gladia-streaming",
        GladiaStreamingPipeline,
        default_config={
            "sample_rate": 16000,
            "channels": 1,
            "sample_width": 2,
            "chunksize_ms": 50,
            "endpoint_url": "https://api.gladia.io/v2/live",
        },
        description="Gladia streaming transcription pipeline. Requires API key from https://www.gladia.io/. Set `GLADIA_API_KEY` env var.",
    )

    PipelineRegistry.register_alias(
        "openai-streaming",
        OpenAIStreamingPipeline,
        default_config={
            "sample_rate": 16000,
            "channels": 1,
            "sample_width": 2,
            "realtime_resolution": 0.02,
            "endpoint_url": "https://api.openai.com/v1/realtime/transcription_sessions",
            "model": "gpt-4o-transcribe",
        },
        description="OpenAI streaming transcription pipeline. Requires API key from https://www.openai.com/. Set `OPENAI_API_KEY` env var.",
    )


register_pipeline_aliases()
