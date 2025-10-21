# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

from .apple_speech_analyzer import SpeechAnalyzerConfig, SpeechAnalyzerPipeline
from .common import TranscriptionOutput
from .transcription_assemblyai import AssemblyAITranscriptionPipeline, AssemblyAITranscriptionPipelineConfig
from .transcription_deepgram import DeepgramTranscriptionPipeline, DeepgramTranscriptionPipelineConfig
from .transcription_groq import GroqTranscriptionConfig, GroqTranscriptionPipeline
from .transcription_nemo import NeMoTranscriptionPipeline, NeMoTranscriptionPipelineConfig
from .transcription_openai import OpenAITranscriptionPipeline, OpenAITranscriptionPipelineConfig
from .transcription_whisperkitpro import WhisperKitProTranscriptionConfig, WhisperKitProTranscriptionPipeline
from .whisperkit import WhisperKitTranscriptionConfig, WhisperKitTranscriptionPipeline
from .transcription_oss_whisper import WhisperOSSTranscriptionPipeline, WhisperOSSTranscriptionPipelineConfig


__all__ = [
    "TranscriptionOutput",
    "SpeechAnalyzerPipeline",
    "SpeechAnalyzerConfig",
    "AssemblyAITranscriptionPipeline",
    "AssemblyAITranscriptionPipelineConfig",
    "WhisperKitTranscriptionPipeline",
    "WhisperKitTranscriptionConfig",
    "WhisperKitProTranscriptionPipeline",
    "WhisperKitProTranscriptionConfig",
    "OpenAITranscriptionPipeline",
    "OpenAITranscriptionPipelineConfig",
    "WhisperOSSTranscriptionPipeline",
    "WhisperOSSTranscriptionPipelineConfig",
    "DeepgramTranscriptionPipeline",
    "DeepgramTranscriptionPipelineConfig",
    "NeMoTranscriptionPipeline",
    "NeMoTranscriptionPipelineConfig",
]
