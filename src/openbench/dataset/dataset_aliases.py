# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""Dataset alias registrations for the CLI."""

import os
from pathlib import Path

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
        "ami-ihm-openbench",
        DatasetConfig(dataset_id="argmaxinc/ami-openbench", split="test", subset="ihm-mix"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
            PipelineType.ORCHESTRATION,
            PipelineType.DIARIZATION,
        },
        description=(
            "AMI-IHM dataset for transcription, orchestration and diarization evaluation. "
            "The audio files and the MT-ASR annotations were taken from https://github.com/BUTSpeechFIT/mt-asr-data-prep and processed to create the dataset. "
            "The diarization annotations were taken from https://github.com/nttcslab-sp/diar-forced-alignment which should contain tighter annotations "
            "when compared to using the ASR segments as the ground truth. See `Can We Really Repurpose Multi-Speaker ASR Corpus for Speaker Diarization?` for more details."
        ),
    )

    DatasetRegistry.register_alias(
        "ami-sdm-openbench",
        DatasetConfig(dataset_id="argmaxinc/ami-openbench", split="test", subset="sdm"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
            PipelineType.ORCHESTRATION,
            PipelineType.DIARIZATION,
        },
        description=(
            "AMI-SDM dataset for transcription, orchestration and diarization evaluation. "
            "The audio files and the MT-ASR annotations were taken from https://github.com/BUTSpeechFIT/mt-asr-data-prep and processed to create the dataset. "
            "The diarization annotations were taken from https://github.com/nttcslab-sp/diar-forced-alignment which should contain tighter annotations "
            "when compared to using the ASR segments as the ground truth. See `Can We Really Repurpose Multi-Speaker ASR Corpus for Speaker Diarization?` for more details."
        ),
    )

    DatasetRegistry.register_alias(
        "callhome-english",
        DatasetConfig(
            dataset_id=os.getenv("CALLHOME_ENGLISH_DATASET_REPO_ID", "argmaxinc/callhome-english"), split="test"
        ),
        supported_pipeline_types={PipelineType.TRANSCRIPTION, PipelineType.ORCHESTRATION, PipelineType.DIARIZATION},
        description=(
            "Callhome English dataset for transcription and orchestration evaluation. "
            "To use this dataset you need to buy the license for the audio files at https://catalog.ldc.upenn.edu/LDC97S42 and the license for the transcript files at https://catalog.ldc.upenn.edu/LDC97T14"
            "and use the `download_dataset.py` script in the common/ directory in OpenBench repository to download create the dataset."
            "Once you do that you can set the `CALLHOME_ENGLISH_DATASET_REPO_ID` environment variable to the repo id of the dataset you created, otherwise it will default to the private Argmax Inc. repo."
        ),
    )

    DatasetRegistry.register_alias(
        "chime-6",
        DatasetConfig(dataset_id="argmaxinc/chime-6", split="test"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
            PipelineType.ORCHESTRATION,
            PipelineType.DIARIZATION,
        },
        description="CHiME-6 dataset for transcription and orchestration evaluation. The audio files are from microphone 2 first channel. For more information see https://www.chimechallenge.org/datasets/chime6",
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
        "earnings22-kws-chunkwise",
        DatasetConfig(
            dataset_id="argmaxinc/earnings22-custom-vocab",
            split="test",
            column_mapping={"dictionary": "keywords-file", "keywords": "dictionary"},
        ),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Earnings22 keyword spotting golden dataset specifically for keyword boosting transcription evaluation.",
    )

    DatasetRegistry.register_alias(
        "earnings22-kws-filewise",
        DatasetConfig(dataset_id="argmaxinc/earnings22-custom-vocab", split="test"),
        supported_pipeline_types={
            PipelineType.TRANSCRIPTION,
        },
        description="Earnings22 keyword spotting golden dataset specifically for keyword boosting transcription evaluation.",
    )

    DatasetRegistry.register_alias(
        "earnings22-kws-golden-filewise",
        DatasetConfig(dataset_id="argmaxinc/earnings22-kws-golden-filewise", split="test"),
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
        DatasetConfig(
            dataset_id="argmaxinc/common_voice_17_0-argmax_subset-400-openbench",
            split="test",
            subset="pt",
        ),
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

    ########## SPEECH GENERATION ##########

    DatasetRegistry.register_alias(
        "customer-service-tts-prompts-vocalized",
        DatasetConfig(
            dataset_id="argmaxinc/customer-service-tts-prompts-vocalized",
            split="validation",
        ),
        supported_pipeline_types={
            PipelineType.SPEECH_GENERATION,
        },
        description="Customer service TTS prompts with vocalized audio for speech generation evaluation.",
    )

    # Seed-TTS eval set (voice cloning): each row ships a target-speaker reference
    # clip (`audio` @ 16kHz) with its transcript (`target_text`, used as --ref-text),
    # the distinct text to synthesize (`prompt_text`, used as --text), `language`,
    # and a stable id (`sample_idx`, the source file name). Consumed by
    # argmax-speech-generation-prototype's voice-clone mode: SIM compares the
    # generated clip against `audio`; WER transcribes it vs `prompt_text`.
    DatasetRegistry.register_alias(
        "seedtts-eval",
        DatasetConfig(
            dataset_id="argmaxinc/seedTTS-eval",
            split="train",
        ),
        supported_pipeline_types={
            PipelineType.SPEECH_GENERATION,
        },
        description="Seed-TTS evaluation set (1088 samples) — reference clip + transcript + language for voice-clone speech generation evaluation (WER + SIM).",
    )

    # Seed-TTS eval v2: same schema as seedtts-eval (audio + target_text/prompt_text
    # + language + sample_idx), an updated 1088-sample voice-clone eval set.
    DatasetRegistry.register_alias(
        "seedtts-eval-v2",
        DatasetConfig(
            dataset_id="argmaxinc/seedTTS-eval-v2",
            split="train",
        ),
        supported_pipeline_types={
            PipelineType.SPEECH_GENERATION,
        },
        description="Seed-TTS evaluation set v2 (1088 samples) — reference clip + transcript + language for voice-clone speech generation evaluation (WER + SIM).",
    )

    # Seed-TTS smoke-test subset of seedtts-eval: same dataset, capped to the first 3
    # samples for quick pipeline validation before a full 1088-sample run.
    DatasetRegistry.register_alias(
        "seedtts-eval-mini",
        DatasetConfig(
            dataset_id="argmaxinc/seedTTS-eval",
            split="train",
            num_samples=3,
        ),
        supported_pipeline_types={
            PipelineType.SPEECH_GENERATION,
        },
        description="Seed-TTS smoke-test subset (first 3 samples) for quick pipeline validation.",
    )

    # Voice-clone eval set curated from argmaxinc/force_aligner_speech_regions
    # (call-center recordings, 30 speakers) in the seedTTS-eval schema plus a SIM
    # yardstick: `prompt_text` is the text to synthesize (WER ground truth),
    # `audio` is a same-speaker reference clip with transcript `target_text`
    # (-> ref_text), and `target_audio` is the held-out REAL recording of
    # `prompt_text` (-> sim_audio) so SIM is NOT computed against the clip the
    # model conditioned on.
    DatasetRegistry.register_alias(
        "voiceclone-eval",
        DatasetConfig(
            dataset_id="argmaxinc/voiceclone-eval",
            split="train",
        ),
        supported_pipeline_types={
            PipelineType.SPEECH_GENERATION,
        },
        description=(
            "Voice-clone evaluation set (374 samples, 30 call-center speakers) built from "
            "force_aligner_speech_regions in seedTTS-eval format — reference clip + transcript "
            "+ same-speaker synthesis text + held-out real target audio as the SIM yardstick "
            "for voice-clone speech generation evaluation (WER + SIM)."
        ),
    )

    # Smoke-test subset of voiceclone-eval: same dataset, capped to the first 3
    # samples for quick pipeline validation before a full 374-sample run.
    DatasetRegistry.register_alias(
        "voiceclone-eval-mini",
        DatasetConfig(
            dataset_id="argmaxinc/voiceclone-eval",
            split="train",
            num_samples=3,
        ),
        supported_pipeline_types={
            PipelineType.SPEECH_GENERATION,
        },
        description="Voice-clone smoke-test subset (first 3 samples) for quick pipeline validation.",
    )

    # Reference-length SIM study, hosted variant: argmaxinc/reflen-sim-eval.
    # 50 call-center speakers, each with one fixed synthesis text (`prompt_text`,
    # from the `targets` config) and a sweep of cumulative-length reference clips
    # (5 s up to several minutes, from the `references` config). The `default`
    # config is the flat per-reference join in the voiceclone-eval schema:
    # `audio` + `target_text` are the reference clip/transcript (-> ref_audio /
    # ref_text), `target_audio` is the held-out REAL recording of `prompt_text`
    # (-> sim_audio), and `sample_idx` is `{source_id}-refNN`; `segment_count` /
    # `reference_length` carry the sweep position for analysis.
    DatasetRegistry.register_alias(
        "reflen-sim-eval",
        DatasetConfig(
            dataset_id="argmaxinc/reflen-sim-eval",
            subset="default",
            split="train",
        ),
        supported_pipeline_types={
            PipelineType.SPEECH_GENERATION,
        },
        description=(
            "Reference-length SIM study (50 call-center speakers x cumulative reference-length "
            "sweep) — per row: reference clip + transcript at one length, fixed same-speaker "
            "synthesis text, and held-out real target audio as the SIM yardstick (WER + SIM)."
        ),
    )

    # Smoke-test subset of reflen-sim-eval: same dataset, capped to the first 3
    # samples (the three shortest references of the first speaker).
    DatasetRegistry.register_alias(
        "reflen-sim-eval-mini",
        DatasetConfig(
            dataset_id="argmaxinc/reflen-sim-eval",
            subset="default",
            split="train",
            num_samples=3,
        ),
        supported_pipeline_types={
            PipelineType.SPEECH_GENERATION,
        },
        description="Reference-length smoke-test subset (first 3 samples) for quick pipeline validation.",
    )

    # Worst Exp1 Python 1.7B WER failures for RD-691 guardrails A/B.
    DatasetRegistry.register_alias(
        "rd691-worst3-reflen",
        DatasetConfig(
            dataset_id="argmaxinc/reflen-sim-eval",
            subset="default",
            split="train",
            include_sample_ids=frozenset(
                {
                    "en_US_Southern_Banking_1589014_channel1-ref26",
                    "en_US_Southern_Banking_1589014_channel1-ref27",
                    "en_US_Southern_Aviation_1589010_channel1-ref30",
                }
            ),
        ),
        supported_pipeline_types={
            PipelineType.SPEECH_GENERATION,
        },
        description=(
            "Three worst Exp1 prototype-1.7b reflen WER samples for RD-691 "
            "guardrails comparison (long Southern banking/aviation refs)."
        ),
    )

    # The exact 100 reflen samples scored by the local Swift throughput run
    # (voice-clone-benchmark rows as of 2026-07-22): cross-implementation
    # comparisons re-run precisely this subset.
    DatasetRegistry.register_alias(
        "reflen-sim-eval-100",
        DatasetConfig(
            dataset_id="argmaxinc/reflen-sim-eval-100",
            split="train",
        ),
        supported_pipeline_types={
            PipelineType.SPEECH_GENERATION,
        },
        description=(
            "reflen-sim-eval subset (100 samples, refs <= 240 s) matching the local Swift "
            "throughput-vocoder run for implementation-neutral A/B comparisons."
        ),
    )

    # Prompt-length SIM study, hosted variant: argmaxinc/promptlen-sim-eval.
    # Fixed-length same-speaker references (30-96 s), synthesis texts swept by
    # length (`prompt_length` 22-1013 chars). Same flat voiceclone-eval schema
    # except the reference clip ships as `reference_audio` (mapped to `audio`
    # here); `target_audio` is the held-out REAL recording (-> sim_audio) and
    # `target_text` the reference transcript (-> ref_text).
    DatasetRegistry.register_alias(
        "promptlen-sim-eval",
        DatasetConfig(
            dataset_id="argmaxinc/promptlen-sim-eval",
            split="train",
            column_mapping={"reference_audio": "audio"},
        ),
        supported_pipeline_types={
            PipelineType.SPEECH_GENERATION,
        },
        description=(
            "Prompt-length SIM study (126 samples, call-center speakers) — fixed reference "
            "clip + transcript per speaker with synthesis-text length swept 22-1013 chars, "
            "held-out real target audio as the SIM yardstick (WER + SIM)."
        ),
    )

    # Smoke-test subset of promptlen-sim-eval (first 3 samples).
    DatasetRegistry.register_alias(
        "promptlen-sim-eval-mini",
        DatasetConfig(
            dataset_id="argmaxinc/promptlen-sim-eval",
            split="train",
            num_samples=3,
            column_mapping={"reference_audio": "audio"},
        ),
        supported_pipeline_types={
            PipelineType.SPEECH_GENERATION,
        },
        description="Prompt-length smoke-test subset (first 3 samples) for quick pipeline validation.",
    )

    # Refclone / reference-length study: built locally from
    # argmaxinc/force_aligner_speech_regions via scripts/refclone/build_dataset.py.
    # Each row: target text + real target audio (SIM) + variable-length ref_audio/ref_text (ICL).
    _refclone_path = os.getenv(
        "REFCLONE_DATASET_PATH",
        str(Path(__file__).resolve().parents[3] / "outputs" / "refclone_openbench" / "hf_dataset"),
    )
    DatasetRegistry.register_alias(
        "refclone-speech-regions",
        DatasetConfig(dataset_id=_refclone_path, split="train"),
        supported_pipeline_types={PipelineType.SPEECH_GENERATION},
        description=(
            "Reference-length voice-clone study set built from force_aligner_speech_regions. "
            "Run scripts/refclone/build_dataset.py first (or set REFCLONE_DATASET_PATH)."
        ),
    )
    DatasetRegistry.register_alias(
        "refclone-speech-regions-mini",
        DatasetConfig(dataset_id=_refclone_path, split="train", num_samples=2),
        supported_pipeline_types={PipelineType.SPEECH_GENERATION},
        description="Smoke-test subset (first 2 reference lengths) of refclone-speech-regions.",
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

    # Local dataset to use with env variables to override with env vars
    # and allow easy testing of local datasets
    if os.getenv("LOCAL_DATASET_PATH") and os.getenv("LOCAL_DATASET_SPLIT"):
        DatasetRegistry.register_alias(
            "local-dataset",
            DatasetConfig(dataset_id=os.getenv("LOCAL_DATASET_PATH"), split=os.getenv("LOCAL_DATASET_SPLIT")),
            supported_pipeline_types={
                PipelineType.TRANSCRIPTION,
                PipelineType.DIARIZATION,
                PipelineType.STREAMING_TRANSCRIPTION,
                PipelineType.ORCHESTRATION,
            },
            description="Local dataset for testing. To use this dataset you need to set the `LOCAL_DATASET_PATH` and `LOCAL_DATASET_SPLIT` environment variables.",
        )


register_dataset_aliases()
