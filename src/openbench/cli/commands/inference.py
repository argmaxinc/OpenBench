# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""Inference command for openbench-cli."""

import sys
from pathlib import Path

import numpy as np
import typer
from pydub import AudioSegment

from openbench.dataset import DiarizationSample, OrchestrationSample, StreamingSample, TranscriptionSample
from openbench.pipeline import PipelineRegistry
from openbench.pipeline_prediction import DiarizationAnnotation, Transcript
from openbench.types import PipelineType

from ..command_utils import get_pipelines_help_text, validate_pipeline_name


BASE_OUTPUT_DIR = Path("inference_outputs")


def load_audio(audio_path: Path) -> tuple[np.ndarray, int]:
    """Load and preprocess audio file to mono 16kHz.

    Args:
        audio_path: Path to audio file

    Returns:
        Tuple of (waveform, sample_rate) where waveform is mono 16kHz
    """
    try:
        # Load audio with pydub
        typer.echo(f"🎵 Loading audio file: {audio_path}")
        audio = AudioSegment.from_file(audio_path)

        # Convert to mono if stereo
        if audio.channels > 1:
            typer.echo("🔄 Converting stereo to mono...")
            audio = audio.set_channels(1)

        # Convert to 16kHz if needed
        if audio.frame_rate != 16000:
            typer.echo(f"🔄 Converting sample rate from {audio.frame_rate}Hz to 16kHz...")
            audio = audio.set_frame_rate(16000)

        # Convert to numpy array
        samples = np.array(audio.get_array_of_samples())

        # Normalize to float32 between -1 and 1
        if audio.sample_width == 2:  # 16-bit
            samples = samples.astype(np.float32) / 32768.0
        elif audio.sample_width == 4:  # 32-bit
            samples = samples.astype(np.float32) / 2147483648.0

        typer.echo(f"✅ Audio loaded successfully: {len(samples)} samples, {audio.frame_rate}Hz")
        return samples, 16000

    except Exception as e:
        typer.echo(f"❌ Failed to load audio file {audio_path}: {e}", err=True)
        sys.exit(1)


def get_dummy_sample(
    pipeline_type: PipelineType, audio_name: str, waveform: np.ndarray, sample_rate: int
) -> StreamingSample | DiarizationSample | OrchestrationSample | TranscriptionSample:
    """Create a dummy sample for inference."""
    if pipeline_type == PipelineType.STREAMING_TRANSCRIPTION:
        return StreamingSample(
            audio_name=audio_name,
            waveform=waveform,
            sample_rate=sample_rate,
            extra_info={},
            reference=Transcript.from_words_info(words=["dummy"]),
        )
    elif pipeline_type == PipelineType.DIARIZATION:
        return DiarizationSample(
            audio_name=audio_name,
            waveform=waveform,
            sample_rate=sample_rate,
            extra_info={},
            reference=DiarizationAnnotation(),
        )
    elif pipeline_type == PipelineType.ORCHESTRATION:
        return OrchestrationSample(
            audio_name=audio_name,
            waveform=waveform,
            sample_rate=sample_rate,
            extra_info={},
            reference=Transcript.from_words_info(words=["dummy"], speaker=["SPEAKER_0"]),
        )
    elif pipeline_type == PipelineType.TRANSCRIPTION:
        return TranscriptionSample(
            audio_name=audio_name,
            waveform=waveform,
            sample_rate=sample_rate,
            extra_info={},
            reference=Transcript.from_words_info(words=["dummy"]),
        )


def inference(
    pipeline_name: str = typer.Option(
        ...,
        "--pipeline",
        "-p",
        help=f"Pipeline alias to run inference on\n\n{get_pipelines_help_text()}",
        callback=validate_pipeline_name,
    ),
    audio_path: Path = typer.Option(..., "--audio-path", "-a", help="Path to audio file to run inference on"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """Runs inference on an audio file using a pre-registered pipeline configuration through its alias."""
    try:
        # Validate audio file exists
        if not audio_path.exists():
            typer.echo(f"❌ Audio file not found: {audio_path}", err=True)
            sys.exit(1)

        # Create output directory
        output_dir = BASE_OUTPUT_DIR / pipeline_name / audio_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        typer.echo(f"📁 Output directory: {output_dir}")

        # Instantiate pipeline
        typer.echo(f"🔧 Creating pipeline: {pipeline_name}")
        pipeline = PipelineRegistry.create_pipeline(name=pipeline_name)

        # Loads audio file
        waveform, sample_rate = load_audio(audio_path)
        audio_name = audio_path.stem

        # Create dataset sample with a dummy reference since we don't have a reference for inference
        typer.echo("📝 Preparing sample for inference...")
        sample = get_dummy_sample(
            pipeline_type=pipeline.pipeline_type, audio_name=audio_name, waveform=waveform, sample_rate=sample_rate
        )

        # Run inference
        typer.echo("🚀 Running inference...")
        inference_result = pipeline(sample)
        prediction = inference_result.prediction

        # Save prediction
        typer.echo("💾 Saving prediction...")
        path_to_prediction = prediction.to_annotation_file(output_dir=output_dir, filename=audio_path.stem)
        typer.echo(f"✅ Prediction saved to {path_to_prediction}")

        if verbose:
            typer.echo(f"📊 Pipeline type: {pipeline.pipeline_type}")
            typer.echo(f"🎵 Audio duration: {len(waveform) / sample_rate:.2f} seconds")
            typer.echo(f"📁 Full output path: {path_to_prediction}")

    except Exception as e:
        typer.echo(f"❌ Inference failed: {e}", err=True)
        if verbose:
            import traceback

            typer.echo(f"📋 Full traceback:\n{traceback.format_exc()}", err=True)
        sys.exit(1)
