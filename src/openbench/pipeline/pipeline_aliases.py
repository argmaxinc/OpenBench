# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""Pipeline alias registrations for common configurations."""

import os

from .diarization import (
    ArgmaxOpenSourceDiarizationPipeline,
    AWSTranscribePipeline,
    DeepgramDiarizationPipeline,
    ElevenLabsDiarizationPipeline,
    NeMoSortformerPipeline,
    PicovoicePipeline,
    PyannoteApiPipeline,
    PyAnnotePipeline,
    SpeakerKitPipeline,
)
from .orchestration import (
    ArgmaxOpenSourceOrchestrationPipeline,
    DeepgramOrchestrationPipeline,
    ElevenLabsOrchestrationPipeline,
    NeMoMTParakeetPipeline,
    OpenAIOrchestrationPipeline,
    PyannoteOrchestrationPipeline,
    WhisperKitProOrchestrationPipeline,
    WhisperXPipeline,
)
from .pipeline_registry import PipelineRegistry
from .speech_generation import (
    WhisperKitSpeechGenerationPipeline,
)
from .streaming_transcription import (
    AssemblyAIStreamingPipeline,
    DeepgramStreamingPipeline,
    FireworksStreamingPipeline,
    GladiaStreamingPipeline,
    OpenAIStreamingPipeline,
)
from .transcription import (
    ArgmaxOpenSourceTranscriptionPipeline,
    AssemblyAITranscriptionPipeline,
    DeepgramTranscriptionPipeline,
    ElevenLabsTranscriptionPipeline,
    GroqTranscriptionPipeline,
    NeMoTranscriptionPipeline,
    OpenAITranscriptionPipeline,
    PyannoteTranscriptionPipeline,
    SpeechAnalyzerPipeline,
    WhisperKitProTranscriptionPipeline,
    WhisperOSSTranscriptionPipeline,
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
        "nemo-sortformer",
        NeMoSortformerPipeline,
        default_config={
            "out_dir": "./nemo_sortformer_logs",
            "use_float16": True,
            "chunk_size": 340,
            "right_context": 40,
            "fifo_size": 40,
            "update_period": 300,
            "speaker_cache_size": 188,
        },
        description="NeMo Sortformer speaker diarization pipeline.",
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
            "engine": "pyannote",
        },
        description="SpeakerKit speaker diarization pipeline using community-1 model from pyannote. Requires CLI installation and API key. Set `SPEAKERKIT_CLI_PATH` and `SPEAKERKIT_API_KEY` env vars. For access to the CLI binary contact speakerkitpro@argmaxinc.com",
    )

    PipelineRegistry.register_alias(
        "speakerkit-sortformer-compressed",
        SpeakerKitPipeline,
        default_config={
            "out_dir": "./speakerkit-sortformer-report",
            "cli_path": os.getenv("SPEAKERKIT_CLI_PATH"),
            "engine": "sortformer",
        },
        description=(
            "SpeakerKit speaker diarization pipeline using Sortformer model compressed to 94MB. Requires CLI installation and API key. "
            "Set `SPEAKERKIT_CLI_PATH` and `SPEAKERKIT_API_KEY` env vars. For access to the CLI binary contact speakerkitpro@argmaxinc.com."
        ),
    )

    PipelineRegistry.register_alias(
        "argmax-oss-diarization",
        ArgmaxOpenSourceDiarizationPipeline,
        default_config={
            "out_dir": "./argmax_oss_diarization_reports",
        },
        description=(
            "Argmax SDK open-source diarization via `argmax-cli diarize`. "
            "Clone/build under ARGMAX_OSS_CACHE_DIR (default ~/.cache/openbench/argmax-oss) unless `cli_path` is set."
            "Uses pyannote's community-1 model for diarization."
        ),
    )

    PipelineRegistry.register_alias(
        "picovoice-diarization",
        PicovoicePipeline,
        default_config={
            "out_dir": "./picovoice_logs",
        },
        description="Picovoice diarization pipeline. Requires API key from https://www.picovoice.ai/. Set `PICOVOICE_API_KEY` env var.",
    )

    PipelineRegistry.register_alias(
        "deepgram-diarization",
        DeepgramDiarizationPipeline,
        default_config={
            "out_dir": "./deepgram_diarization_results",
            "model_version": "nova-3",
        },
        description="Deepgram diarization pipeline. Requires API key from https://www.deepgram.com/. Set `DEEPGRAM_API_KEY` env var.",
    )

    PipelineRegistry.register_alias(
        "elevenlabs-diarization",
        ElevenLabsDiarizationPipeline,
        default_config={
            "out_dir": "./elevenlabs_diarization_results",
            "model_id": "scribe_v2",
        },
        description="ElevenLabs diarization pipeline. Requires API key from https://elevenlabs.io/. Set `ELEVENLABS_API_KEY` env var.",
    )

    ################# ORCHESTRATION PIPELINES #################

    PipelineRegistry.register_alias(
        "whisperx-tiny",
        WhisperXPipeline,
        default_config={
            "out_dir": "./whisperx_output",
            "model_name": "tiny",
            "device": "cpu",
            "compute_type": "int8",
            "batch_size": 16,
            "threads": 8,
        },
        description="WhisperX diarized transcription pipeline from https://github.com/m-bain/whisperX running with the tiny model",
    )

    PipelineRegistry.register_alias(
        "whisperx-large-v3-turbo",
        WhisperXPipeline,
        default_config={
            "out_dir": "./whisperx_output",
            "model_name": "large-v3-turbo",
            "device": "cpu",
            "compute_type": "int8",
            "batch_size": 16,
            "threads": 8,
        },
        description="WhisperX diarzed transcription pipeline from https://github.com/m-bain/whisperX running with the large-v3-turbo model",
    )

    PipelineRegistry.register_alias(
        "deepgram-orchestration",
        DeepgramOrchestrationPipeline,
        default_config={
            "out_dir": "./deepgram_orchestration_results",
            "model_version": "nova-3",
        },
        description="Deepgram orchestration pipeline. Requires API key from https://www.deepgram.com/. Set `DEEPGRAM_API_KEY` env var.",
    )

    PipelineRegistry.register_alias(
        "elevenlabs-orchestration",
        ElevenLabsOrchestrationPipeline,
        default_config={
            "out_dir": "./elevenlabs_orchestration_results",
            "model_id": "scribe_v2",
        },
        description="ElevenLabs orchestration pipeline with diarization. Requires API key from https://elevenlabs.io/. Set `ELEVENLABS_API_KEY` env var.",
    )

    PipelineRegistry.register_alias(
        "whisperkitpro-orchestration-tiny",
        WhisperKitProOrchestrationPipeline,
        default_config={
            "repo_id": "argmaxinc/whisperkit-pro",
            "model_variant": "openai_whisper-tiny",
            "cli_path": os.getenv("WHISPERKITPRO_CLI_PATH"),
            "orchestration_strategy": "subsegment",
            "engine": "pyannote",
            "use_exclusive_reconciliation": True,
        },
        description="WhisperKitPro orchestration pipeline using the tiny version of the model. Requires `WHISPERKITPRO_CLI_PATH` env var and depending on your permissions also `WHISPERKITPRO_API_KEY` env var.",
    )

    PipelineRegistry.register_alias(
        "whisperkitpro-orchestration-large-v3",
        WhisperKitProOrchestrationPipeline,
        default_config={
            "repo_id": "argmaxinc/whisperkit-pro",
            "model_variant": "openai_whisper-large-v3",
            "cli_path": os.getenv("WHISPERKITPRO_CLI_PATH"),
            "orchestration_strategy": "subsegment",
            "engine": "pyannote",
            "use_exclusive_reconciliation": True,
        },
        description="WhisperKitPro orchestration pipeline using the large-v3 version of the model. Requires `WHISPERKITPRO_CLI_PATH` env var and depending on your permissions also `WHISPERKITPRO_API_KEY` env var.",
    )

    PipelineRegistry.register_alias(
        "whisperkitpro-orchestration-large-v3-turbo",
        WhisperKitProOrchestrationPipeline,
        default_config={
            "repo_id": "argmaxinc/whisperkit-pro",
            "model_variant": "openai_whisper-large-v3-v20240930",
            "cli_path": os.getenv("WHISPERKITPRO_CLI_PATH"),
            "orchestration_strategy": "subsegment",
            "engine": "pyannote",
            "use_exclusive_reconciliation": True,
        },
        description="WhisperKitPro orchestration pipeline using the large-v3-v20240930 version of the model (which is the same as large-v3-turbo from OpenAI). Requires `WHISPERKITPRO_CLI_PATH` env var and depending on your permissions also `WHISPERKITPRO_API_KEY` env var.",
    )

    PipelineRegistry.register_alias(
        "whisperkitpro-orchestration-large-v3-turbo-compressed",
        WhisperKitProOrchestrationPipeline,
        default_config={
            "repo_id": "argmaxinc/whisperkit-pro",
            "model_variant": "openai_whisper-large-v3-v20240930_626MB",
            "cli_path": os.getenv("WHISPERKITPRO_CLI_PATH"),
            "orchestration_strategy": "subsegment",
            "engine": "pyannote",
            "use_exclusive_reconciliation": True,
        },
        description="WhisperKitPro orchestration pipeline using the large-v3-v20240930 version of the model compressed to 626MB (which is the same as large-v3-turbo from OpenAI). Requires `WHISPERKITPRO_CLI_PATH` env var and depending on your permissions also `WHISPERKITPRO_API_KEY` env var.",
    )

    PipelineRegistry.register_alias(
        "whisperkitpro-orchestration-parakeet-v2",
        WhisperKitProOrchestrationPipeline,
        default_config={
            "repo_id": "argmaxinc/parakeetkit-pro",
            "model_variant": "nvidia_parakeet-v2",
            "cli_path": os.getenv("WHISPERKITPRO_CLI_PATH"),
            "orchestration_strategy": "subsegment",
            "engine": "pyannote",
            "use_exclusive_reconciliation": True,
        },
        description="WhisperKitPro orchestration pipeline using the parakeet-v2 version of the model. Requires `WHISPERKITPRO_CLI_PATH` env var and depending on your permissions also `WHISPERKITPRO_API_KEY` env var.",
    )

    PipelineRegistry.register_alias(
        "whisperkitpro-orchestration-parakeet-v2-compressed",
        WhisperKitProOrchestrationPipeline,
        default_config={
            "repo_id": "argmaxinc/parakeetkit-pro",
            "model_variant": "nvidia_parakeet-v2_476MB",
            "cli_path": os.getenv("WHISPERKITPRO_CLI_PATH"),
            "orchestration_strategy": "subsegment",
            "engine": "pyannote",
            "use_exclusive_reconciliation": True,
        },
        description="WhisperKitPro orchestration pipeline using the parakeet-v2 version of the model compressed to 476MB. Requires `WHISPERKITPRO_CLI_PATH` env var and depending on your permissions also `WHISPERKITPRO_API_KEY` env var.",
    )

    PipelineRegistry.register_alias(
        "whisperkitpro-orchestration-parakeet-v2-compressed-sortformer-compressed",
        WhisperKitProOrchestrationPipeline,
        default_config={
            "repo_id": "argmaxinc/parakeetkit-pro",
            "model_variant": "nvidia_parakeet-v2_476MB",
            "cli_path": os.getenv("WHISPERKITPRO_CLI_PATH"),
            "orchestration_strategy": "subsegment",
            "engine": "sortformer",
            "diarization_mode": "prerecorded",
        },
        description=(
            "WhisperKitPro orchestration pipeline using the parakeet-v2 version of the model compressed to 476MB and using Sortformer for diarization. "
            "Requires `WHISPERKITPRO_CLI_PATH` env var and depending on your permissions also `WHISPERKITPRO_API_KEY` env var."
        ),
    )

    PipelineRegistry.register_alias(
        "whisperkitpro-orchestration-parakeet-v3",
        WhisperKitProOrchestrationPipeline,
        default_config={
            "repo_id": "argmaxinc/parakeetkit-pro",
            "model_variant": "nvidia_parakeet-v3",
            "cli_path": os.getenv("WHISPERKITPRO_CLI_PATH"),
            "orchestration_strategy": "subsegment",
            "engine": "pyannote",
            "use_exclusive_reconciliation": True,
        },
        description="WhisperKitPro orchestration pipeline using the parakeet-v3 version of the model. Requires `WHISPERKITPRO_CLI_PATH` env var and depending on your permissions also `WHISPERKITPRO_API_KEY` env var.",
    )

    PipelineRegistry.register_alias(
        "whisperkitpro-orchestration-parakeet-v3-compressed",
        WhisperKitProOrchestrationPipeline,
        default_config={
            "repo_id": "argmaxinc/parakeetkit-pro",
            "model_variant": "nvidia_parakeet-v3_494MB",
            "cli_path": os.getenv("WHISPERKITPRO_CLI_PATH"),
            "orchestration_strategy": "subsegment",
            "engine": "pyannote",
            "use_exclusive_reconciliation": True,
        },
        description="WhisperKitPro orchestration pipeline using the parakeet-v3 version of the model compressed to 494MB. Requires `WHISPERKITPRO_CLI_PATH` env var and depending on your permissions also `WHISPERKITPRO_API_KEY` env var.",
    )

    PipelineRegistry.register_alias(
        "whisperkitpro-orchestration-parakeet-v3-compressed-sortformer-compressed",
        WhisperKitProOrchestrationPipeline,
        default_config={
            "repo_id": "argmaxinc/parakeetkit-pro",
            "model_variant": "nvidia_parakeet-v3_494MB",
            "cli_path": os.getenv("WHISPERKITPRO_CLI_PATH"),
            "orchestration_strategy": "subsegment",
            "engine": "sortformer",
            "diarization_mode": "prerecorded",
        },
        description=(
            "WhisperKitPro orchestration pipeline using the parakeet-v3 version of the model compressed to 494MB and using Sortformer for diarization. "
            "Requires `WHISPERKITPRO_CLI_PATH` env var and depending on your permissions also `WHISPERKITPRO_API_KEY` env var."
        ),
    )

    PipelineRegistry.register_alias(
        "openai-orchestration",
        OpenAIOrchestrationPipeline,
        default_config={
            "model_version": "gpt-4o-transcribe-diarize",
        },
        description="OpenAI orchestration pipeline using the `gpt-4o-transcribe-diarize` version of the model. Requires `OPENAI_API_KEY` env var.",
    )

    PipelineRegistry.register_alias(
        "nemo-mt-parakeet",
        NeMoMTParakeetPipeline,
        default_config={
            "out_dir": "./nemo_mt_parakeet_logs",
            "diar_model_id": "nvidia/diar_streaming_sortformer_4spk-v2.1",
            "asr_model_id": "nvidia/multitalker-parakeet-streaming-0.6b-v1",
        },
        description="NeMo Multi-Talker Parakeet orchestration pipeline (diarization + transcription).",
    )

    PipelineRegistry.register_alias(
        "pyannote-orchestration",
        PyannoteOrchestrationPipeline,
        default_config={
            "out_dir": "./pyannote_orchestration_results",
            "timeout": 3600,
            "request_buffer": 30,
        },
        description="PyannoteAI orchestration pipeline (diarization + transcription). Uses the precision-2 model with Nvidia Parakeet STT. Requires `PYANNOTE_TOKEN` env var from https://www.pyannote.ai/.",
    )

    PipelineRegistry.register_alias(
        "argmax-oss-orchestration-tiny",
        ArgmaxOpenSourceOrchestrationPipeline,
        default_config={
            "out_dir": "./argmax_oss_orchestration_reports",
            "model_version": "tiny",
            "word_timestamps": False,
            "chunking_strategy": "vad",
        },
        description="Argmax SDK (OSS): `argmax-cli transcribe --diarization` with verbose RTTM-like diarization log "
        "as the transcript (word + speaker per token). Cache: ARGMAX_OSS_CACHE_DIR or default; optional `cli_path`.",
    )

    ################# TRANSCRIPTION PIPELINES #################

    PipelineRegistry.register_alias(
        "whisperkit-tiny",
        ArgmaxOpenSourceTranscriptionPipeline,
        default_config={
            "model_version": "tiny",
            "word_timestamps": True,
            "chunking_strategy": "vad",
        },
        description="Argmax SDK (open source) transcription via `argmax-cli` (Swift release build, not debug). Model tiny. "
        "Cache: ARGMAX_OSS_CACHE_DIR or ~/.cache/openbench/argmax-oss unless `cli_path` is set. "
        "For `openbench-cli evaluate`, prefer `-d earnings22-3hours` over `librispeech-200`: longer clips amortize "
        "first-run model load so speed factor is less misleading than on very short utterances.",
    )

    PipelineRegistry.register_alias(
        "whisperkit-large-v3",
        ArgmaxOpenSourceTranscriptionPipeline,
        default_config={
            "model_version": "large-v3",
            "word_timestamps": True,
            "chunking_strategy": "vad",
        },
        description="Argmax SDK (open source) transcription via `argmax-cli` (release build). Model large-v3. "
        "Cache: ARGMAX_OSS_CACHE_DIR or default. For transcription benchmarks use `-d earnings22-3hours`; "
        "`librispeech-200` is better reserved for quick WER smoke tests.",
    )

    PipelineRegistry.register_alias(
        "whisperkit-large-v3-turbo",
        ArgmaxOpenSourceTranscriptionPipeline,
        default_config={
            "model_version": "large-v3-v20240930",
            "word_timestamps": True,
            "chunking_strategy": "vad",
        },
        description="Argmax SDK (open source) transcription via `argmax-cli` (release build). Model large-v3-v20240930. "
        "Cache: ARGMAX_OSS_CACHE_DIR or default. Prefer `-d earnings22-3hours` for evaluate; short LibriSpeech "
        "clips skew speed factor because of fixed startup cost per file.",
    )

    PipelineRegistry.register_alias(
        "speech-analyzer",
        SpeechAnalyzerPipeline,
        default_config={
            "clone_dir": "./speech_analyzer_repo",
        },
        description="Speech Analyzer transcription pipeline (open-source version). Requires Swift and Xcode installed.",
    )

    # Legacy mode without manual local model download
    # PipelineRegistry.register_alias(
    #     "whisperkitpro-tiny",
    #     WhisperKitProTranscriptionPipeline,
    #     default_config={
    #         "model_version": "tiny",
    #         "model_prefix": "openai",
    #         "model_repo_name": "argmaxinc/whisperkit-pro",
    #         "cli_path": os.getenv("WHISPERKITPRO_CLI_PATH"),
    #         "use_keywords": False
    #     },
    #     description="WhisperKitPro transcription pipeline using the tiny version of the model. Requires Swift and Xcode installed. Requires `WHISPERKITPRO_CLI_PATH` env var and depending on your permissions also `WHISPERKITPRO_API_KEY` env var.",
    # )

    # New mode with manual local model download
    # to avoid HF Download Rate Limiting
    PipelineRegistry.register_alias(
        "whisperkitpro-tiny",
        WhisperKitProTranscriptionPipeline,
        default_config={
            "repo_id": "argmaxinc/whisperkit-pro",
            "model_variant": "openai_whisper-tiny",
            "cli_path": os.getenv("WHISPERKITPRO_CLI_PATH"),
        },
        description="WhisperKitPro transcription pipeline using the tiny version of the model. Requires Swift and Xcode installed. Requires `WHISPERKITPRO_CLI_PATH` env var and depending on your permissions also `WHISPERKITPRO_API_KEY` env var.",
    )

    PipelineRegistry.register_alias(
        "whisperkitpro-large-v3",
        WhisperKitProTranscriptionPipeline,
        default_config={
            "repo_id": "argmaxinc/whisperkit-pro",
            "model_variant": "openai_whisper-large-v3",
            "cli_path": os.getenv("WHISPERKITPRO_CLI_PATH"),
        },
        description="WhisperKitPro transcription pipeline using the large-v3 version of the model. Requires Swift and Xcode installed. Requires `WHISPERKITPRO_CLI_PATH` env var and depending on your permissions also `WHISPERKITPRO_API_KEY` env var.",
    )

    PipelineRegistry.register_alias(
        "whisperkitpro-large-v3-turbo",
        WhisperKitProTranscriptionPipeline,
        default_config={
            "repo_id": "argmaxinc/whisperkit-pro",
            "model_variant": "openai_whisper-large-v3-v20240930",
            "cli_path": os.getenv("WHISPERKITPRO_CLI_PATH"),
        },
        description="WhisperKitPro transcription pipeline using the large-v3-v20240930 version of the model (which is the same as large-v3-turbo from OpenAI). Requires `WHISPERKITPRO_CLI_PATH` env var and depending on your permissions also `WHISPERKITPRO_API_KEY` env var.",
    )

    PipelineRegistry.register_alias(
        "whisperkitpro-large-v3-turbo-compressed",
        WhisperKitProTranscriptionPipeline,
        default_config={
            "repo_id": "argmaxinc/whisperkit-pro",
            "model_variant": "openai_whisper-large-v3-v20240930_626MB",
            "cli_path": os.getenv("WHISPERKITPRO_CLI_PATH"),
        },
        description="WhisperKitPro transcription pipeline using the large-v3-v20240930 version of the model compressed to 626MB (which is the same as large-v3-turbo from OpenAI). Requires `WHISPERKITPRO_CLI_PATH` env var and depending on your permissions also `WHISPERKITPRO_API_KEY` env var.",
    )

    PipelineRegistry.register_alias(
        "whisperkitpro-parakeet-v2",
        WhisperKitProTranscriptionPipeline,
        default_config={
            "repo_id": "argmaxinc/parakeetkit-pro",
            "model_variant": "nvidia_parakeet-v2",
            "cli_path": os.getenv("WHISPERKITPRO_CLI_PATH"),
        },
        description="WhisperKitPro transcription pipeline using the parakeet-v2 version of the model. Requires `WHISPERKITPRO_CLI_PATH` env var and depending on your permissions also `WHISPERKITPRO_API_KEY` env var.",
    )

    PipelineRegistry.register_alias(
        "whisperkitpro-parakeet-v2-compressed",
        WhisperKitProTranscriptionPipeline,
        default_config={
            "repo_id": "argmaxinc/parakeetkit-pro",
            "model_variant": "nvidia_parakeet-v2_476MB",
            "cli_path": os.getenv("WHISPERKITPRO_CLI_PATH"),
        },
        description="WhisperKitPro transcription pipeline using the parakeet-v2 version of the model compressed to 476MB. Requires `WHISPERKITPRO_CLI_PATH` env var and depending on your permissions also `WHISPERKITPRO_API_KEY` env var.",
    )

    PipelineRegistry.register_alias(
        "whisperkitpro-parakeet-v3",
        WhisperKitProTranscriptionPipeline,
        default_config={
            "repo_id": "argmaxinc/parakeetkit-pro",
            "model_variant": "nvidia_parakeet-v3",
            "cli_path": os.getenv("WHISPERKITPRO_CLI_PATH"),
        },
        description="WhisperKitPro transcription pipeline using the parakeet-v3 version of the model. Requires `WHISPERKITPRO_CLI_PATH` env var and depending on your permissions also `WHISPERKITPRO_API_KEY` env var.",
    )

    PipelineRegistry.register_alias(
        "whisperkitpro-parakeet-v3-compressed",
        WhisperKitProTranscriptionPipeline,
        default_config={
            "repo_id": "argmaxinc/parakeetkit-pro",
            "model_variant": "nvidia_parakeet-v3_494MB",
            "cli_path": os.getenv("WHISPERKITPRO_CLI_PATH"),
        },
        description="WhisperKitPro transcription pipeline using the parakeet-v3 version of the model compressed to 494MB. Requires `WHISPERKITPRO_CLI_PATH` env var and depending on your permissions also `WHISPERKITPRO_API_KEY` env var.",
    )

    PipelineRegistry.register_alias(
        "whisperkitpro-local-model",
        WhisperKitProTranscriptionPipeline,
        default_config={
            "model_dir": os.getenv("WHISPERKITPRO_LOCAL_MODEL_PATH"),
            "cli_path": os.getenv("WHISPERKITPRO_CLI_PATH"),
        },
        description=(
            "WhisperKitPro transcription using only a local model directory (no default Hugging Face repo). "
            "Set `WHISPERKITPRO_LOCAL_MODEL_PATH` to the folder passed as `--model-path` on the CLI; it must exist. "
            "Requires `WHISPERKITPRO_CLI_PATH` and may require `WHISPERKITPRO_API_KEY`."
        ),
    )

    PipelineRegistry.register_alias(
        "groq-whisper-large-v3-turbo",
        GroqTranscriptionPipeline,
        default_config={
            "model_id": "whisper-large-v3-turbo",
            "temperature": 0.0,
            "force_language": False,
        },
        description="Groq transcription pipeline using the whisper-large-v3-turbo model. Requires `GROQ_API_KEY` env var.",
    )

    PipelineRegistry.register_alias(
        "groq-whisper-large-v3",
        GroqTranscriptionPipeline,
        default_config={
            "model_id": "whisper-large-v3",
            "temperature": 0.0,
            "force_language": False,
        },
        description="Groq transcription pipeline using the whisper-large-v3 model. Requires `GROQ_API_KEY` env var.",
    )

    ################# KEYWORD-ENABLED TRANSCRIPTION PIPELINES #################

    PipelineRegistry.register_alias(
        "openai-transcription",
        OpenAITranscriptionPipeline,
        default_config={
            "out_dir": "./openai_keywords_results",
            "model_version": "whisper-1",
            "use_keywords": False,
        },
        description="OpenAI Whisper transcription with keyword boosting. Requires `OPENAI_API_KEY` env var.",
    )

    PipelineRegistry.register_alias(
        "whisper-tiny-oss-transcription",
        WhisperOSSTranscriptionPipeline,
        default_config={
            "out_dir": "./whisper_oss_results",
            "model_version": "tiny",
            "device": None,
            "use_keywords": False,
        },
        description="Open-source OpenAI Whisper transcription (local)",
    )

    PipelineRegistry.register_alias(
        "whisper-base-oss-transcription",
        WhisperOSSTranscriptionPipeline,
        default_config={
            "out_dir": "./whisper_oss_results",
            "model_version": "base",
            "device": None,
            "use_keywords": False,
        },
        description="Open-source OpenAI Whisper transcription (local)",
    )

    PipelineRegistry.register_alias(
        "whisper-large-v2-oss-transcription",
        WhisperOSSTranscriptionPipeline,
        default_config={
            "out_dir": "./whisper_oss_results",
            "model_version": "large-v2",
            "device": None,
            "use_keywords": False,
        },
        description="Open-source OpenAI Whisper transcription (local)",
    )

    PipelineRegistry.register_alias(
        "whisper-large-v3-oss-transcription",
        WhisperOSSTranscriptionPipeline,
        default_config={
            "out_dir": "./whisper_oss_results",
            "model_version": "large-v3",
            "device": None,
            "use_keywords": False,
        },
        description="Open-source OpenAI Whisper transcription (local)",
    )

    PipelineRegistry.register_alias(
        "whisper-turbo-oss-transcription",
        WhisperOSSTranscriptionPipeline,
        default_config={
            "out_dir": "./whisper_oss_results",
            "model_version": "large-v3-turbo",
            "device": None,
            "use_keywords": False,
        },
        description="Open-source OpenAI Whisper transcription (local)",
    )

    PipelineRegistry.register_alias(
        "deepgram-transcription",
        DeepgramTranscriptionPipeline,
        default_config={
            "out_dir": "./deepgram_keywords_results",
            "model_version": "nova-3",
            "use_keywords": False,
        },
        description="Deepgram transcription with keyword boosting. Requires `DEEPGRAM_API_KEY` env var.",
    )

    PipelineRegistry.register_alias(
        "nemo-transcription-ctc-large",
        NeMoTranscriptionPipeline,
        default_config={
            "out_dir": "./nemo_keywords_results",
            "nemo_model_file": "nvidia/stt_en_fastconformer_ctc_large",
            "decoder_type": "ctc",
            "device": "cpu",
            "acoustic_batch_size": 32,
            "beam_threshold": 7.0,
            "context_score": 3.0,
            "ctc_ali_token_weight": 0.5,
            "use_keywords": False,
            "spelling_separator": "_",
        },
        description="NeMo CTC transcription with context biasing for keyword spotting. Local model, no API key required.",
    )

    PipelineRegistry.register_alias(
        "assemblyai-transcription",
        AssemblyAITranscriptionPipeline,
        default_config={
            "model_version": "universal",
            "use_keywords": False,
        },
        description="AssemblyAI transcription pipeline with keyword boosting support. Requires API key from https://www.assemblyai.com/. Set `ASSEMBLYAI_API_KEY` env var.",
    )

    PipelineRegistry.register_alias(
        "elevenlabs-transcription",
        ElevenLabsTranscriptionPipeline,
        default_config={
            "model_id": "scribe_v2",
            "use_keywords": False,
        },
        description="ElevenLabs transcription pipeline with keyterm prompting support. Requires API key from https://elevenlabs.io/. Set `ELEVENLABS_API_KEY` env var.",
    )

    PipelineRegistry.register_alias(
        "pyannote-transcription",
        PyannoteTranscriptionPipeline,
        default_config={
            "out_dir": "./pyannote_transcription_results",
            "timeout": 3600,
            "request_buffer": 30,
        },
        description="PyannoteAI transcription pipeline (ignores speaker attribution). Uses the precision-2 model with Nvidia Parakeet STT. Requires `PYANNOTE_TOKEN` env var from https://www.pyannote.ai/.",
    )

    ################# SPEECH GENERATION PIPELINES #################

    PipelineRegistry.register_alias(
        "whisperkit-speech-generation",
        WhisperKitSpeechGenerationPipeline,
        default_config={
            "out_dir": "./speech_generation_results",
            "cli_path": os.getenv("WHISPERKIT_CLI_PATH"),
            "speaker": "aiden",
            "language": "english",
            "seed": 10,
            "temperature": 0.9,
            "top_k": 50,
            "max_new_tokens": 245,
        },
        description=(
            "WhisperKit speech-generation pipeline. Synthesizes audio from text prompts via whisperkit-cli TTS. "
            "WER is computed by `SpeechGenerationWordErrorRate`, which transcribes the generated audio with "
            "WhisperKitPro / parakeet-v2 by default and compares against the original prompt. "
            "Requires `WHISPERKIT_CLI_PATH` for TTS; the default WER metric also requires `WHISPERKITPRO_CLI_PATH`."
        ),
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

    PipelineRegistry.register_alias(
        "assemblyai-streaming",
        AssemblyAIStreamingPipeline,
        default_config={
            "sample_rate": 16000,
            "channels": 1,
            "sample_width": 2,
            "chunksize_ms": 50,
            "endpoint_url": "wss://streaming.assemblyai.com/v3/ws",
        },
        description="AssemblyAI streaming transcription pipeline. Requires API key from https://www.assemblyai.com/. Set `ASSEMBLYAI_API_KEY` env var.",
    )


register_pipeline_aliases()
