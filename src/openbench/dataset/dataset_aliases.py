# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""Dataset alias registrations for the CLI."""

import os

from ..types import PipelineType
from .dataset_base import DatasetConfig
from .dataset_registry import DatasetRegistry


def register_dataset_aliases() -> None:
    """Register all dataset aliases with their configurations."""

    ########## DIARIZATION ##########
    DatasetRegistry.register_alias(
        "voxconverse",
        DatasetConfig(dataset_id="diarizers-community/voxconverse", split="test"),
        supported_pipeline_types={
            PipelineType.DIARIZATION,
        },
        description="Speaker diarization dataset with ground truth annotations",
    )

    DatasetRegistry.register_alias(
        "callhome_hf",
        DatasetConfig(dataset_id="talkbank/callhome", split="data", subset="eng", num_samples=1),
        supported_pipeline_types={
            PipelineType.DIARIZATION,
        },
        description="Talkbank provided Callhome dataset english subset with 1 sample for debugging purposes. Note that this dataset is gated on HF Hub.",
    )

    DatasetRegistry.register_alias(
        "msdwild",
        DatasetConfig(dataset_id="argmaxinc/msdwild", split="validation"),
        supported_pipeline_types={
            PipelineType.DIARIZATION,
        },
        description="Multi-speaker dataset with diverse scenarios",
    )

    DatasetRegistry.register_alias(
        "earnings21",
        DatasetConfig(dataset_id="argmaxinc/earnings21", split="test"),
        supported_pipeline_types={
            PipelineType.DIARIZATION,
            PipelineType.TRANSCRIPTION,
            PipelineType.ORCHESTRATION,
        },
        description="Earnings call dataset with transcription ground truth",
    )

    DatasetRegistry.register_alias(
        "ami-ihm",
        DatasetConfig(dataset_id="diarizers-community/ami", split="test", subset="ihm"),
        supported_pipeline_types={
            PipelineType.DIARIZATION,
        },
        description="AMI meeting dataset with IHM microphone setup",
    )

    DatasetRegistry.register_alias(
        "ami-sdm",
        DatasetConfig(dataset_id="diarizers-community/ami", split="test", subset="sdm"),
        supported_pipeline_types={
            PipelineType.DIARIZATION,
        },
        description="AMI meeting dataset with SDM microphone setup",
    )

    DatasetRegistry.register_alias(
        "american-life-podcast",
        DatasetConfig(
            dataset_id=os.getenv("AMERICAN_LIFE_PODCAST_DATASET_REPO_ID", "argmaxinc/american-life"), split="test"
        ),
        supported_pipeline_types={
            PipelineType.DIARIZATION,
        },
        description=(
            "This American Life podcast dataset. "
            "To use this dataset recreate it using the `download_dataset.py` script in the common/ directory in OpenBench repository. "
            "We are not allowe to distribute it since the audios from https://www.thisamericanlife.org/ don't mention any license. "
            "Once you do that you can set the `AMERICAN_LIFE_PODCAST_DATASET_REPO_ID` environment variable to the repo id of the dataset you created, otherwise it will default to the private Argmax Inc. repo."
        ),
    )

    DatasetRegistry.register_alias(
        "ava-avd",
        DatasetConfig(dataset_id="argmaxinc/ava-avd", split="test"),
        supported_pipeline_types={
            PipelineType.DIARIZATION,
        },
        description="AVA Audio-Visual Diarization dataset",
    )

    DatasetRegistry.register_alias(
        "icsi",
        DatasetConfig(dataset_id="argmaxinc/icsi", split="test"),
        supported_pipeline_types={
            PipelineType.DIARIZATION,
        },
        description="ICSI meeting corpus dataset",
    )

    DatasetRegistry.register_alias(
        "aishell-4",
        DatasetConfig(dataset_id="argmaxinc/aishell-4", split="test"),
        supported_pipeline_types={
            PipelineType.DIARIZATION,
        },
        description="AISHELL-4 dataset for speaker diarization",
    )

    DatasetRegistry.register_alias(
        "ali-meetings",
        DatasetConfig(dataset_id="argmaxinc/ali-meetings", split="test"),
        supported_pipeline_types={
            PipelineType.DIARIZATION,
        },
        description="AliMeetings dataset for speaker diarization",
    )

    DatasetRegistry.register_alias(
        "callhome",
        DatasetConfig(dataset_id=os.getenv("CALLHOME_DATASET_REPO_ID", "argmaxinc/callhome"), split="part2"),
        supported_pipeline_types={
            PipelineType.DIARIZATION,
        },
        description=(
            "CALLHOME dataset distributed by LDC also known as NIST SRE 2000. "
            "To use this dataset you need to buy the license at https://catalog.ldc.upenn.edu/LDC2001S97 and use the `download_dataset.py` script in the common/ directory in OpenBench repository to download create the dataset."
            "Once you do that you can set the `CALLHOME_DATASET_REPO_ID` environment variable to the repo id of the dataset you created, otherwise it will default to the private Argmax Inc. repo."
        ),
    )

    DatasetRegistry.register_alias(
        "ego4d",
        DatasetConfig(dataset_id=os.getenv("EGO4D_DATASET_REPO_ID", "argmaxinc/ego4d"), split="validation"),
        supported_pipeline_types={
            PipelineType.DIARIZATION,
        },
        description=(
            "Ego4D dataset for speaker diarization. "
            "To use this dataset you need to request access to the dataset at https://ego4d-data.org/docs/start-here/ and use the `download_dataset.py` script in the common/ directory in OpenBench repository to download create the dataset."
            "Once you do that you can set the `EGO4D_DATASET_REPO_ID` environment variable to the repo id of the dataset you created, otherwise it will default to the private Argmax Inc. repo. "
            "NOTE: We use the validation split for evaluation because the test split reference values are not available."
        ),
    )

    DatasetRegistry.register_alias(
        "dihard-3",
        DatasetConfig(dataset_id=os.getenv("DIHARD_3_DATASET_REPO_ID", "argmaxinc/dihard-3"), split="full"),
        supported_pipeline_types={
            PipelineType.DIARIZATION,
        },
        description=(
            "DIHARD-3 dataset for speaker diarization. "
            "To use this dataset you need to buy the license at https://catalog.ldc.upenn.edu/LDC2022S14 and use the `download_dataset.py` script in the common/ directory in OpenBench repository to download create the dataset."
            "Once you do that you can set the `DIHARD_3_DATASET_REPO_ID` environment variable to the repo id of the dataset you created, otherwise it will default to the private Argmax Inc. repo."
        ),
    )

    ########## TRANSCRIPTION ##########

    DatasetRegistry.register_alias(
        "callhome-english",
        DatasetConfig(
            dataset_id=os.getenv("CALLHOME_ENGLISH_DATASET_REPO_ID", "argmaxinc/callhome-english"), split="test"
        ),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
            PipelineType.ORCHESTRATION,
        },
        description=(
            "Callhome English dataset for transcription and orchestration evaluation. "
            "To use this dataset you need to buy the license for the audio files at https://catalog.ldc.upenn.edu/LDC97S42 and the license for the transcript files at https://catalog.ldc.upenn.edu/LDC97T14"
            "and use the `download_dataset.py` script in the common/ directory in OpenBench repository to download create the dataset."
            "Once you do that you can set the `CALLHOME_ENGLISH_DATASET_REPO_ID` environment variable to the repo id of the dataset you created, otherwise it will default to the private Argmax Inc. repo."
        ),
    )

    DatasetRegistry.register_alias(
        "librispeech",
        DatasetConfig(dataset_id="argmaxinc/librispeech-openbench", split="test", subset="full"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="LibriSpeech dataset for transcription evaluation",
    )

    DatasetRegistry.register_alias(
        "librispeech-200",
        DatasetConfig(dataset_id="argmaxinc/librispeech-openbench", split="test", subset="200"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="LibriSpeech dataset for transcription evaluation with only 200 files. Commonly used for debugging or get an estimate of the performance of the model.",
    )

    DatasetRegistry.register_alias(
        "earnings22",
        DatasetConfig(dataset_id="argmaxinc/earnings22-openbench", split="test", subset="full"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Earnings call dataset for transcription evaluation.",
    )

    DatasetRegistry.register_alias(
        "earnings22-12hours",
        DatasetConfig(dataset_id="argmaxinc/earnings22-openbench", split="test", subset="12hours"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Earnings call dataset for transcription evaluation with only 12 hours of audio.",
    )

    DatasetRegistry.register_alias(
        "earnings22-3hours",
        DatasetConfig(dataset_id="argmaxinc/earnings22-openbench", split="test", subset="3hours"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Earnings call dataset for transcription evaluation with only 3 hours of audio.",
    )

    DatasetRegistry.register_alias(
        "earnings22-keywords",
        DatasetConfig(dataset_id="argmaxinc/earnings22-kws-golden", split="test"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Earnings22 keyword spotting golden dataset specifically for keyword boosting transcription evaluation.",
    )

    DatasetRegistry.register_alias(
        "earnings22-keywords-debug",
        DatasetConfig(dataset_id="argmaxinc/earnings22-kws-golden", split="test", num_samples=5),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Debug version of keyword spotting dataset with only 5 samples for quick testing.",
    )

    DatasetRegistry.register_alias(
        "common-voice",
        DatasetConfig(
            dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="full"
        ),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains all languages",
    )

    DatasetRegistry.register_alias(
        "common-voice",
        DatasetConfig(
            dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="full"
        ),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains all languages",
    )

    # Common Voice specific languages subsets

    # English
    DatasetRegistry.register_alias(
        "common-voice-en",
        DatasetConfig(dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="en"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains only english",
    )

    # Spanish
    DatasetRegistry.register_alias(
        "common-voice-es",
        DatasetConfig(dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="es"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains only spanish",
    )

    # German
    DatasetRegistry.register_alias(
        "common-voice-de",
        DatasetConfig(dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="de"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains only german",
    )

    # French
    DatasetRegistry.register_alias(
        "common-voice-fr",
        DatasetConfig(dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="fr"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains only french",
    )

    # Portuguese
    DatasetRegistry.register_alias(
        "common-voice-pt",
        DatasetConfig(dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="pt"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains only portuguese",
    )

    # Japanese
    DatasetRegistry.register_alias(
        "common-voice-ja",
        DatasetConfig(dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="ja"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains only japanese",
    )

    # Italian
    DatasetRegistry.register_alias(
        "common-voice-it",
        DatasetConfig(dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="it"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains only italian",
    )

    # Chinese
    DatasetRegistry.register_alias(
        "common-voice-zh",
        DatasetConfig(dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="zh"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains only chinese",
    )

    # Dutch
    DatasetRegistry.register_alias(
        "common-voice-nl",
        DatasetConfig(dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="nl"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains only dutch",
    )

    # Polish
    DatasetRegistry.register_alias(
        "common-voice-pl",
        DatasetConfig(dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="pl"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains only polish",
    )

    # Indonesian
    DatasetRegistry.register_alias(
        "common-voice-id",
        DatasetConfig(dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="id"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains only indonesian",
    )

    # Galician
    DatasetRegistry.register_alias(
        "common-voice-gl",
        DatasetConfig(dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="gl"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains only galician",
    )

    # Romanian
    DatasetRegistry.register_alias(
        "common-voice-ro",
        DatasetConfig(dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="ro"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains only romanian",
    )

    # Czech
    DatasetRegistry.register_alias(
        "common-voice-cs",
        DatasetConfig(dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="cs"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains only czech",
    )

    # Swedish
    DatasetRegistry.register_alias(
        "common-voice-sv",
        DatasetConfig(dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="sv"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains only swedish",
    )

    # Hungarian
    DatasetRegistry.register_alias(
        "common-voice-hu",
        DatasetConfig(dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="hu"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains only hungarian",
    )

    # Greek
    DatasetRegistry.register_alias(
        "common-voice-el",
        DatasetConfig(dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="el"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains only greek",
    )

    # Finnish
    DatasetRegistry.register_alias(
        "common-voice-fi",
        DatasetConfig(dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="fi"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains only finnish",
    )

    # Vietnamese
    DatasetRegistry.register_alias(
        "common-voice-vi",
        DatasetConfig(dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="vi"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains only vietnamese",
    )

    # Danish
    DatasetRegistry.register_alias(
        "common-voice-da",
        DatasetConfig(dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="da"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains only danish",
    )

    # Catalan
    DatasetRegistry.register_alias(
        "common-voice-ca",
        DatasetConfig(dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="ca"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains only catalan",
    )

    # Russian
    DatasetRegistry.register_alias(
        "common-voice-ru",
        DatasetConfig(dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench", split="test", subset="ru"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Common Voice dataset for transcription evaluation with up to 400 samples per language this subset contains only russian",
    )

    ########## STREAMING TRANSCRIPTION ##########

    DatasetRegistry.register_alias(
        "timit",
        DatasetConfig(dataset_id="kylelovesllms/timit_asr", split="test", num_samples=300),
        supported_pipeline_types={PipelineType.STREAMING_TRANSCRIPTION},
        description="TIMIT dataset for streaming transcription evaluation",
    )

    DatasetRegistry.register_alias(
        "timit-debug",
        DatasetConfig(dataset_id="kylelovesllms/timit_asr", split="test", num_samples=5),
        supported_pipeline_types={PipelineType.STREAMING_TRANSCRIPTION},
        description="TIMIT dataset for streaming transcription evaluation for debugging purposes only",
    )

    DatasetRegistry.register_alias(
        "timit-stitched",
        DatasetConfig(dataset_id="argmaxinc/timit_stitched", split="test"),
        supported_pipeline_types={PipelineType.STREAMING_TRANSCRIPTION},
        description="TIMIT stitched dataset for streaming transcription evaluation",
    )

    ########## END POINTING TEST DATASETS ##########

    DatasetRegistry.register_alias(
        "timit-stitched-short-silences-debug",
        DatasetConfig(dataset_id="argmaxinc/timit_stitched_silenced-v1", split="train", num_samples=1),
        supported_pipeline_types={PipelineType.STREAMING_TRANSCRIPTION},
        description="TIMIT stitched Debug dataset with short silences for endpointing evals",
    )

    DatasetRegistry.register_alias(
        "timit-stitched-medium-silences-debug",
        DatasetConfig(dataset_id="argmaxinc/timit_stitched_silenced-v2", split="train", num_samples=1),
        supported_pipeline_types={PipelineType.STREAMING_TRANSCRIPTION},
        description="TIMIT stitched Debug dataset with medium silences for endpointing evals",
    )

    DatasetRegistry.register_alias(
        "timit-stitched-long-silences-debug",
        DatasetConfig(dataset_id="argmaxinc/timit_stitched_silenced-v3", split="train", num_samples=1),
        supported_pipeline_types={PipelineType.STREAMING_TRANSCRIPTION},
        description="TIMIT stitched Debug dataset with long silences for endpointing evals",
    )

    DatasetRegistry.register_alias(
        "timit-stitched-very-long-silences-debug",
        DatasetConfig(dataset_id="argmaxinc/timit_stitched_silenced-v4", split="train", num_samples=1),
        supported_pipeline_types={PipelineType.STREAMING_TRANSCRIPTION},
        description="TIMIT stitched Debug dataset with very long silences for endpointing evals",
    )

    DatasetRegistry.register_alias(
        "timit-stitched-short-silences",
        DatasetConfig(dataset_id="argmaxinc/timit_stitched_silenced-v1", split="train"),
        supported_pipeline_types={PipelineType.STREAMING_TRANSCRIPTION},
        description="TIMIT stitched dataset with short silences for endpointing evals",
    )

    DatasetRegistry.register_alias(
        "timit-stitched-medium-silences",
        DatasetConfig(dataset_id="argmaxinc/timit_stitched_silenced-v2", split="train"),
        supported_pipeline_types={PipelineType.STREAMING_TRANSCRIPTION},
        description="TIMIT stitched dataset with medium silences for endpointing evals",
    )

    DatasetRegistry.register_alias(
        "timit-stitched-long-silences",
        DatasetConfig(dataset_id="argmaxinc/timit_stitched_silenced-v3", split="train"),
        supported_pipeline_types={PipelineType.STREAMING_TRANSCRIPTION},
        description="TIMIT stitched dataset with long silences for endpointing evals",
    )

    DatasetRegistry.register_alias(
        "timit-stitched-very-long-silences",
        DatasetConfig(dataset_id="argmaxinc/timit_stitched_silenced-v4", split="train"),
        supported_pipeline_types={PipelineType.STREAMING_TRANSCRIPTION},
        description="TIMIT stitched dataset with very long silences for endpointing evals",
    )


register_dataset_aliases()
