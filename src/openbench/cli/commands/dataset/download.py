# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from openbench.dataset import BaseDataset, BaseSample, DatasetRegistry
from openbench.types import PipelineType


console = Console()


def download(
    dataset_name: str | None = typer.Option(
        None, "--dataset-name", "-d", help="Name of the dataset alias to download (must be a registered dataset alias)"
    ),
    pipeline_type: PipelineType | None = typer.Option(
        None,
        "--pipeline-type",
        "-p",
        help="Pipeline type for the dataset. If the passed dataset alias has multiple supported pipeline types, this option is required.",
    ),
    sample_id: int | None = typer.Option(None, "--sample-id", "-s", min=0, help="Specific sample ID to download"),
    audio_name: str | None = typer.Option(None, "--audio-name", "-a", help="Specific audio name to download"),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for downloaded files. If not provided download will occur in `downloaded_datasets/<dataset_name>`",
    ),
) -> None:
    """Download a dataset or specific samples from a dataset.

    This command can download:

    - The entire dataset (default behavior)

    - A specific sample by sample ID

    - A specific sample by audio name

    The dataset_name must be one of the registered dataset aliases.

    Examples:

        # Download entire dataset


        openbench-cli dataset download --dataset-name my-dataset


        # Download with specific pipeline type


        openbench-cli dataset download --dataset-name my-dataset --pipeline-type transcription


        # Download specific sample by ID


        openbench-cli dataset download --dataset-name my-dataset --sample-id 123


        # Download specific sample by audio name


        openbench-cli dataset download --dataset-name my-dataset --audio-name sample_001.wav


        # Download to specific directory


        openbench-cli dataset download --dataset-name my-dataset --output-dir ./my_data

    """
    try:
        # Validate that dataset_name is provided
        if dataset_name is None:
            console.print("[red]Error: --dataset-name is required[/red]")
            raise typer.Exit(1)

        # Validate that only one of sample_id or audio_name is provided
        if sample_id and audio_name:
            console.print("[red]Error: Cannot specify both --sample-id and --audio-name[/red]")
            raise typer.Exit(1)

        # Validate that dataset_name is a registered alias
        available_aliases = DatasetRegistry.list_aliases()
        if dataset_name not in available_aliases:
            console.print(f"[red]Error: Dataset alias '{dataset_name}' not found.[/red]")
            console.print(f"[yellow]Available dataset aliases: {', '.join(available_aliases)}[/yellow]")
            raise typer.Exit(1)

        # Get the dataset
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading dataset...", total=None)

            try:
                dataset = DatasetRegistry.get_dataset_from_alias(dataset_name, pipeline_type)
                progress.update(task, description="Dataset loaded successfully")
            except Exception as e:
                console.print(f"[red]Error loading dataset: {e}[/red]")
                raise typer.Exit(1)

        # Set output directory
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = Path("downloaded_datasets") / dataset_name

        output_path.mkdir(parents=True, exist_ok=True)

        # Download based on the specified options
        if sample_id:
            _download_sample_by_id(dataset, sample_id, output_path)
        elif audio_name:
            _download_sample_by_audio_name(dataset, audio_name, output_path)
        else:
            _download_full_dataset(dataset, output_path)

        console.print(f"[green]Download completed successfully to: {output_path}[/green]")

    except Exception as e:
        console.print(f"[red]Error during download: {e}[/red]")
        raise typer.Exit(1)


def _download_full_dataset(dataset: type[BaseDataset], output_path: Path) -> None:
    """Download the entire dataset."""
    console.print(f"[blue]Downloading full dataset to {output_path}[/blue]")

    # Create subdirectories for different data types
    audio_dir = output_path / "audio"
    metadata_dir = output_path / "metadata"
    reference_dir = output_path / "reference"
    audio_dir.mkdir(exist_ok=True)
    metadata_dir.mkdir(exist_ok=True)
    reference_dir.mkdir(exist_ok=True)

    # Download all samples
    for i in range(len(dataset)):
        sample = dataset[i]
        _save_sample(sample, audio_dir, metadata_dir, reference_dir, i)


def _download_sample_by_id(dataset: type[BaseDataset], sample_id: int, output_path: Path) -> None:
    """Download a specific sample by sample ID."""
    if sample_id >= len(dataset):
        console.print(f"[red]Error: Sample ID {sample_id} is out of range (0-{len(dataset) - 1})[/red]")
        raise typer.Exit(1)

    console.print(f"[blue]Downloading sample {sample_id} to {output_path}[/blue]")

    # Create subdirectories
    audio_dir = output_path / "audio"
    metadata_dir = output_path / "metadata"
    reference_dir = output_path / "reference"
    audio_dir.mkdir(exist_ok=True)
    metadata_dir.mkdir(exist_ok=True)
    reference_dir.mkdir(exist_ok=True)

    sample = dataset[sample_id]
    _save_sample(sample, audio_dir, metadata_dir, reference_dir, sample_id)


def _download_sample_by_audio_name(dataset: type[BaseDataset], audio_name: str, output_path: Path) -> None:
    """Download a specific sample by audio name."""
    console.print(f"[blue]Searching for audio file '{audio_name}'...[/blue]")

    found_idx = None
    for i in range(len(dataset)):
        sample = dataset[i]
        if sample.audio_name == audio_name or sample.audio_name == Path(audio_name).stem:
            found_idx = i
            break

    if found_idx is None:
        console.print(f"[red]Error: Audio file '{audio_name}' not found in dataset[/red]")
        raise typer.Exit(1)

    console.print(f"[blue]Found audio file at index {found_idx}, downloading to {output_path}[/blue]")

    # Create subdirectories
    audio_dir = output_path / "audio"
    metadata_dir = output_path / "metadata"
    reference_dir = output_path / "reference"
    audio_dir.mkdir(exist_ok=True)
    metadata_dir.mkdir(exist_ok=True)
    reference_dir.mkdir(exist_ok=True)

    sample = dataset[found_idx]
    _save_sample(sample, audio_dir, metadata_dir, reference_dir, found_idx)


def _save_sample(sample: type[BaseSample], audio_dir: Path, metadata_dir: Path, reference_dir: Path, idx: int) -> None:
    """Save a sample to disk."""

    # Save audio file
    _ = sample.save_audio(audio_dir)

    # Save reference file
    sample.reference.to_annotation_file(str(reference_dir), sample.audio_name)

    # Save metadata
    metadata = {
        "sample_index": idx,
        "audio_name": sample.audio_name,
        "sample_rate": sample.sample_rate,
        "duration": sample.get_audio_duration(),
        "extra_info": sample.extra_info,
    }

    metadata_filename = f"{sample.audio_name}_metadata.json"
    metadata_path = metadata_dir / metadata_filename
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4, default=str)
