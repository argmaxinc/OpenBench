# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

import argparse
import json
import os
import re
import shutil
import subprocess
import tarfile
import time
import warnings
import zipfile
from abc import ABC, abstractmethod
from functools import wraps
from pathlib import Path
from typing import Any

import datasets
import gdown  # Used for MSDWild dataset
import pandas as pd
import requests
import soundfile as sf
from argmaxtools.utils import get_logger
from bs4 import BeautifulSoup
from huggingface_hub import snapshot_download
from pyannote.core import Annotation, Segment, Timeline
from pyannote.database.util import load_rttm, load_uem
from pydantic import BaseModel, Field, model_validator
from pydub import AudioSegment
from scipy.io import wavfile
from textgrid import TextGrid
from tqdm import tqdm


logger = get_logger(__name__)

# Suppress FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

RAW_DATASETS_DIR = "raw_datasets"

# Add at the top of the file, after imports
DATASETS_URLS = {
    "msdwild": {
        "audio_url": "https://drive.google.com/uc?id=1I5qfuPPGBM9keJKz0VN-OYEeRMJ7dgpl",
        "rttm_url": {
            "few_train": "https://raw.githubusercontent.com/X-LANCE/MSDWILD/refs/heads/master/rttms/few.train.rttm",
            "few_validation": "https://raw.githubusercontent.com/X-LANCE/MSDWILD/refs/heads/master/rttms/few.val.rttm",
            "many_validation": "https://raw.githubusercontent.com/X-LANCE/MSDWILD/refs/heads/master/rttms/many.val.rttm",
        },
    },
    "icsi-meetings": {
        "shell_script_url": "https://groups.inf.ed.ac.uk/ami//download/temp/icsiBuild-12511-Wed-Dec-18-2024.wget.sh",
        "annotations_url": "https://groups.inf.ed.ac.uk/ami/ICSICorpusAnnotations/ICSI_core_NXT.zip",
    },
    "earnings21": {"repo_url": "https://github.com/revdotcom/speech-datasets.git"},
    "ali-meetings": {
        "audio_annot_url": "https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Test_Ali.tar.gz"
    },
    "aishell-4": {"hf_repo_id": "AISHELL/AISHELL-4"},
    "american-life": {
        "base_episode_url": "https://tal.fm/",
        "max_episode": 702,
        "kaggle_dataset_id": "shuyangli94/this-american-life-podcast-transcriptsalignments",
        "annotation_splits": {
            "train": "train-transcripts-aligned.json",
            "test": "test-transcripts-aligned.json",
            "validation": "valid-transcripts-aligned.json",
        },
        "speaker_map": "full-speaker-map.json",
    },
    "ava-avd": {
        "offsets_url": "https://raw.githubusercontent.com/showlab/AVA-AVD/refs/heads/main/dataset/split/offsets.txt",
        "video_list_url": "https://raw.githubusercontent.com/showlab/AVA-AVD/refs/heads/main/dataset/split/video.list",
        "train_list_url": "https://raw.githubusercontent.com/showlab/AVA-AVD/refs/heads/main/dataset/split/train.list",
        "val_list_url": "https://raw.githubusercontent.com/showlab/AVA-AVD/refs/heads/main/dataset/split/val.list",
        "test_list_url": "https://raw.githubusercontent.com/showlab/AVA-AVD/refs/heads/main/dataset/split/test.list",
    },
    "callhome": {
        "annotations_url": "https://us.openslr.org/resources/10/sre2000-key.tar.gz",
        "sph2pipe_repo_url": "https://github.com/EduardoPach/sph2pipe.git",
        "part1_url": "https://raw.githubusercontent.com/BUTSpeechFIT/CALLHOME_sublists/refs/heads/main/part1/part1.txt",
        "part2_url": "https://raw.githubusercontent.com/BUTSpeechFIT/CALLHOME_sublists/refs/heads/main/part2/part2.txt",
    },
}


def retry(max_retries: int = 5, backoff_factor: int = 2, exceptions: tuple = (Exception,)):
    """
    A decorator to retry a function upon encountering specified exceptions.

    Args:
        max_retries (int): Maximum number of retry attempts.
        backoff_factor (int): Exponential backoff factor (seconds).
        exceptions (tuple): Tuple of exception types to retry on.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)  # Call the original function
                except exceptions as e:
                    retries += 1
                    if retries > max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries.")
                        # Re-raise the exception after exhausting retries
                        raise e
                    wait_time = backoff_factor**retries
                    logger.warning(
                        f"Error in {func.__name__}: {e}. Retrying ({retries}/{max_retries}) in {wait_time}s..."
                    )
                    time.sleep(wait_time)

        return wrapper

    return decorator


@retry()
def download_file(url: str, output_filename: str) -> None:
    # First check if file already exists
    if Path(output_filename).exists():
        logger.info(f"File {output_filename} already exists, skipping download...")
        return

    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_mb = int(response.headers.get("content-length", 0)) / 1024 / 1024
    chunk_size = 8192
    chunk_size_mb = chunk_size / 1024 / 1024
    logger.info(f"Downloading {output_filename} from {url} ({total_mb:.2f} MB)")
    with tqdm(
        total=total_mb,
        unit="MB",
        desc=f"Downloading {output_filename}",
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        with open(output_filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(chunk_size_mb)


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def load_list(path: str) -> list[str]:
    with open(path, "r") as f:
        values = f.readlines()
    return values


class AwsProfileInfo(BaseModel):
    profile: str = Field(..., description="The AWS profile")
    access_id: str = Field(..., description="The AWS access ID")
    access_key: str = Field(..., description="The AWS access key")
    region: str | None = Field(None, description="The AWS region")
    output: str = Field("json", description="The AWS output format")

    @property
    def aws_dir(self) -> Path:
        dir = Path.home() / ".aws"
        dir.mkdir(parents=True, exist_ok=True)
        return dir

    @property
    def credentials_path(self) -> Path:
        return self.aws_dir / "credentials"

    @property
    def config_path(self) -> Path:
        return self.aws_dir / "config"

    def exists(self) -> bool:
        if not self.credentials_path.exists():
            return False
        profiles = [
            line.split("[")[1].split("]")[0]
            for line in self.credentials_path.read_text().splitlines()
            if line.startswith("[")
        ]
        return any(profile == self.profile for profile in profiles)

    def write_credentials(self) -> None:
        if self.exists():
            logger.info(f"AWS profile '{self.profile}' already exists, updating access and secret key...")
            lines = self.credentials_path.read_text().splitlines()
            for line in lines:
                if line.startswith(f"[{self.profile}]"):
                    lines[lines.index(line)] = f"[{self.profile}]\n"
                    lines[lines.index(line) + 1] = f"aws_access_key_id = {self.access_id}\n"
                    lines[lines.index(line) + 2] = f"aws_secret_access_key = {self.access_key}\n"
                    logger.info(f"AWS profile '{self.profile}' updated successfully.")

            # Merge all lines into a single file again
            with self.credentials_path.open("w") as f:
                f.writelines(lines)
            return

        logger.info(f"Creating AWS credentials file for profile '{self.profile}'...")
        with self.credentials_path.open("a") as f:
            f.write(f"[{self.profile}]\n")
            f.write(f"aws_access_key_id = {self.access_id}\n")
            f.write(f"aws_secret_access_key = {self.access_key}\n")

    def write_config(self) -> None:
        if self.exists():
            logger.info(f"AWS profile '{self.profile}' already exists, skipping config file creation...")
            return

        logger.info(f"Creating AWS config file for profile '{self.profile}'...")
        with self.config_path.open("a") as f:
            f.write(f"[{self.profile}]\n")
            if self.region:
                f.write(f"region = {self.region}\n")
            f.write(f"output = {self.output}\n")

    def create_profile(self) -> None:
        self.write_credentials()
        self.write_config()


class SpeakerDiarizationData(BaseModel):
    split: str = Field(..., description="The split of the dataset")
    split: str = Field(..., description="The split of the dataset")
    audio_paths: list[str] = Field(..., description="The path to the audio files")
    annotation_paths: list[str] = Field(..., description="The path to the annotation files (.rttm)")
    uem_paths: list[str] | None = Field(
        None,
        description="The path to the UEM files",
    )
    transcript: list[list[str]] | None = Field(
        None,
        description="The transcript of the audio as a list of strings (words)",
    )
    word_speakers: list[list[str]] | None = Field(
        None,
        description="The speakers of the words in the transcript as a list of strings representing the speaker for each word",
    )
    word_timestamps: list[list[tuple[float, float]]] | None = Field(
        None,
        description="The timestamps of the words in the transcript as a list of tuples",
    )
    metadata: dict[str, list[Any]] | None = Field(
        None,
        description="A dictionary containing metadata for each sample of the dataset",
    )

    @model_validator(mode="after")
    def check_audio_annotation_match(self) -> "SpeakerDiarizationData":
        if len(self.audio_paths) != len(self.annotation_paths):
            raise ValueError("The number of audio files and annotation files must be the same")

        for audio_path, rttm_path in zip(self.audio_paths, self.annotation_paths):
            if Path(audio_path).stem != Path(rttm_path).stem:
                raise ValueError(f"Audio file {audio_path} and RTTM file {rttm_path} do not match")

        return self

    @model_validator(mode="after")
    def check_uem_annotation_match(self) -> "SpeakerDiarizationData":
        if self.uem_paths is None:
            return self

        if len(self.uem_paths) != len(self.annotation_paths):
            raise ValueError("The number of UEM files and annotation files must be the same")

        return self

    @model_validator(mode="after")
    def check_transcript_match_audios(self) -> "SpeakerDiarizationData":
        if self.transcript is None:
            return self

        if len(self.transcript) != len(self.audio_paths):
            raise ValueError("The number of transcripts and audio files must be the same")

        return self

    @model_validator(mode="after")
    def check_transcript_word_speakers_match(self) -> "SpeakerDiarizationData":
        if self.transcript is None or self.word_speakers is None:
            return self

        if len(self.transcript) != len(self.word_speakers):
            raise ValueError(
                "The number of transcripts and word speakers must be the same "
                f"{len(self.transcript)=} and {len(self.word_speakers)=}"
            )

        for transcript, word_speakers in zip(self.transcript, self.word_speakers):
            if len(transcript) != len(word_speakers):
                raise ValueError(
                    "The number of words in the transcript and the number of word speakers must be the same "
                    f"{len(transcript)=} and {len(word_speakers)=}"
                )

        return self

    @model_validator(mode="after")
    def check_word_speakers_timestamps_match(self) -> "SpeakerDiarizationData":
        if self.word_speakers is None or self.word_timestamps is None:
            return self

        if len(self.word_speakers) != len(self.word_timestamps):
            raise ValueError(
                "The number of word speakers and word timestamps must be the same "
                f"{len(self.word_speakers)=} and {len(self.word_timestamps)=}"
            )

        for word_speakers, word_timestamps in zip(self.word_speakers, self.word_timestamps):
            if len(word_speakers) != len(word_timestamps):
                raise ValueError(
                    "The number of word speakers and word timestamps must be the same "
                    f"{len(word_speakers)=} and {len(word_timestamps)=}"
                )

        return self


class SpeakerDiarizationDataset(ABC):
    sample_rate: int = 16_000

    def __init__(self, org_name: str = "argmaxinc") -> None:
        self.org_name = org_name

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        pass

    @property
    def output_dir(self) -> Path:
        return Path(RAW_DATASETS_DIR) / self.dataset_name

    @abstractmethod
    def download(self) -> None:
        """This method should download the dataset and will save it in the `output_dir` path.
        The download should generate the audio files and annotation files in .rttm format.
        """
        pass

    @abstractmethod
    def create_dataset(self) -> dict[str, SpeakerDiarizationData]:
        """This method should return a dictionary of `SpeakerDiarizationData` objects for each audio file in the dataset
        The keys should represent the dataset splits (train, test, validation) and the values should be a list of `SpeakerDiarizationData` objects
        assume that the dataset is already downloaded to the `output_dir` path.
        """
        pass

    def process_rttm_file(self, rttm_file: str) -> tuple[list[float], list[float], list[str | int]]:
        logger.info(f"Processing RTTM file {rttm_file}")
        rttm_data: dict[str, Annotation] = load_rttm(rttm_file)
        if len(rttm_data) > 1 or len(rttm_data) == 0:
            raise ValueError(f"RTTM file {rttm_file} has more than one or no annotations")
        annotation_uri = list(rttm_data.keys())[0]
        annotation: Annotation = rttm_data[annotation_uri]
        timestamps_start = []
        timestamps_end = []
        speakers = []
        for segment, _, label in annotation.itertracks(yield_label=True):
            timestamps_start.append(segment.start)
            timestamps_end.append(segment.end)
            speakers.append(label)
        return timestamps_start, timestamps_end, speakers

    def process_uem_file(self, uem_file: str) -> list[tuple[float, float]]:
        logger.info(f"Processing UEM file {uem_file}")
        uem_data: dict[str, Timeline] = load_uem(uem_file)
        uem_timestamps: list[tuple[float, float]] = []
        if len(uem_data) > 1 or len(uem_data) == 0:
            raise ValueError(f"UMEM file {uem_file} has more than one or no annotations")
        uem_uri = list(uem_data.keys())[0]
        uem: Timeline = uem_data[uem_uri]
        for segment in uem:
            uem_timestamps.append((segment.start, segment.end))

        return uem_timestamps

    def build_dataset(self, data: dict[str, SpeakerDiarizationData]) -> datasets.DatasetDict:
        dataset_dict = {}
        for split, split_data in data.items():
            audio_paths = split_data.audio_paths
            timestamps_start, timestamps_end, speakers = [], [], []
            for rttm_file in split_data.annotation_paths:
                _timestamps_start, _timestamps_end, _speakers = self.process_rttm_file(rttm_file)
                timestamps_start.append(_timestamps_start)
                timestamps_end.append(_timestamps_end)
                speakers.append(_speakers)

            metadata = split_data.metadata or {}
            dataset_entry = dict(
                audio=audio_paths,
                timestamps_start=timestamps_start,
                timestamps_end=timestamps_end,
                speakers=speakers,
                **metadata,
            )

            if split_data.uem_paths is not None:
                uem_timestamps = [self.process_uem_file(uem_path) for uem_path in split_data.uem_paths]
                dataset_entry["uem_timestamps"] = uem_timestamps

            if split_data.transcript is not None:
                dataset_entry["transcript"] = split_data.transcript

            if split_data.word_speakers is not None:
                dataset_entry["word_speakers"] = split_data.word_speakers

            if split_data.word_timestamps is not None:
                dataset_entry["word_timestamps"] = split_data.word_timestamps

            dataset_dict[split] = datasets.Dataset.from_dict(dataset_entry).cast_column(
                "audio", datasets.Audio(sampling_rate=self.sample_rate)
            )

        return datasets.DatasetDict(dataset_dict)

    def generate(self) -> datasets.DatasetDict:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(self.output_dir)
        self.download()
        data = self.create_dataset()
        return self.build_dataset(data)

    def push_to_hub(self) -> None:
        ds = self.generate()
        ds.push_to_hub(f"{self.org_name}/{self.dataset_name}", private=True)


class Earnings21Dataset(SpeakerDiarizationDataset):
    @property
    def dataset_name(self) -> str:
        return "earnings21"

    def download(self) -> None:
        repo_url = DATASETS_URLS[self.dataset_name]["repo_url"]
        logger.info(f"Downloading {self.dataset_name} dataset to {os.getcwd()}")
        # Clone repo to self.output_dir
        logger.info(f"Cloning repo {repo_url} to {os.getcwd()}")
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                repo_url,
            ]
        )
        # Fetch earnings21 directory
        logger.info(f"Fetching earnings21 directory on {os.getcwd()}")
        subprocess.run(
            ["git", "sparse-checkout", "init", "--cone"],
            cwd="speech-datasets",
        )
        subprocess.run(
            ["git", "sparse-checkout", "set", "earnings21"],
            cwd="speech-datasets",
        )

    def _parse_nlp_references(self, nlp_file_path: Path) -> tuple[list[str], list[str]]:
        df = pd.read_csv(nlp_file_path, delimiter="|")
        words = df["token"].astype(str).tolist()
        speakers = df["speaker"].astype(str).tolist()

        return words, speakers

    def create_dataset(self) -> dict[str, SpeakerDiarizationData]:
        rttm_dir = Path("speech-datasets", "earnings21", "rttms")
        audio_dir = Path("speech-datasets", "earnings21", "media")
        transcript_dir = Path("speech-datasets", "earnings21", "transcripts", "nlp_references")

        rttm_files = sorted((rttm_dir.glob("*.rttm")))
        audio_files = sorted((audio_dir.glob("*.mp3")))
        audio_files_wav = [audio_file.with_suffix(".wav") for audio_file in audio_files]

        transcripts: list[list[str]] = []
        word_speakers: list[list[str]] = []

        # Convert mp3 and populate transcripts and word_speakers
        for audio_file, audio_file_wav in tqdm(zip(audio_files, audio_files_wav), total=len(audio_files)):
            if audio_file_wav.exists():
                tqdm.write(f"Audio file {audio_file_wav} already exists, skipping...")
            else:
                tqdm.write(f"Converting {audio_file} to {audio_file_wav}")
                AudioSegment.from_mp3(str(audio_file)).export(str(audio_file_wav), format="wav")

            nlp_file_path = transcript_dir / audio_file.with_suffix(".nlp").name
            words, speakers = self._parse_nlp_references(nlp_file_path)
            transcripts.append(words)
            word_speakers.append(speakers)

        # Only one partition so I'll consider as test
        return {
            "test": SpeakerDiarizationData(
                split="test",
                audio_paths=[str(audio_file_wav) for audio_file_wav in audio_files_wav],
                annotation_paths=[str(rttm_file) for rttm_file in rttm_files],
                transcript=transcripts,
                word_speakers=word_speakers,
            )
        }


class MSDWildDataset(SpeakerDiarizationDataset):
    @property
    def dataset_name(self) -> str:
        return "msdwild"

    def _download_wav(self) -> None:
        url_for_wav = DATASETS_URLS[self.dataset_name]["audio_url"]
        output_file = Path("wav.zip")

        if output_file.exists():
            logger.info("Audio files already exist, skipping download...")
            return

        logger.info(f"Downloading audio files from {url_for_wav} to {output_file}")
        gdown.download(url_for_wav, str(output_file))
        logger.info(f"Extracting audio files from {output_file} to {self.output_dir}")
        with zipfile.ZipFile(str(output_file), "r") as zip_ref:
            zip_ref.extractall()

    def _download_rttm(self) -> None:
        url_for_rttm: dict[str, str] = DATASETS_URLS[self.dataset_name]["rttm_url"]
        output_rttm_dir = Path("rttms")
        output_rttm_dir.mkdir(parents=True, exist_ok=True)

        for split, url in url_for_rttm.items():
            # `subset` is either few or many and `split_kind` is either val or train
            subset, split_kind = split.split("_")
            response = requests.get(url)
            if not response.ok:
                raise ValueError(
                    f"Failed to download RTTM file from {url_for_rttm} with status code {response.status_code}"
                )

            with open(f"{split}.rttm", "wb") as f:
                f.write(response.content)

            rttm_data: dict[str, Annotation] = load_rttm(f"{split}.rttm")

            for name, annotation in rttm_data.items():
                split_dir = output_rttm_dir / split_kind
                split_dir.mkdir(parents=True, exist_ok=True)
                output_rttm_file = split_dir / f"{name}.rttm"
                logger.info(f"Writing RTTM file {output_rttm_file} for {name}")
                with open(str(output_rttm_file), "w") as f:
                    annotation.write_rttm(f)

    def download(self) -> None:
        self._download_wav()
        self._download_rttm()

    def create_dataset(self) -> dict[str, SpeakerDiarizationData]:
        wav_dir = Path("wav")
        rttm_dir = Path("rttms")
        data: dict[str, SpeakerDiarizationData] = {}

        split_dirs = list(rttm_dir.glob("*"))

        for split_dir in split_dirs:
            rttm_files = sorted((split_dir.glob("*.rttm")))
            audio_files = []
            for rttm_file in rttm_files:
                audio_file = wav_dir / f"{rttm_file.stem}.wav"
                if not audio_file.exists():
                    raise ValueError(f"Audio file {audio_file} does not exist for RTTM file {rttm_file}")
                audio_files.append(str(audio_file))

            data[split_dir.name] = SpeakerDiarizationData(
                split=split_dir.name,
                audio_paths=[str(audio_file) for audio_file in audio_files],
                annotation_paths=[str(rttm_file) for rttm_file in rttm_files],
            )

        return data


class ICSIMeetingsDataset(SpeakerDiarizationDataset):
    @property
    def dataset_name(self) -> str:
        return "icsi-meetings"

    def _download_wav(self) -> None:
        if Path("Signals").exists():
            logger.info("Audio files already exist, skipping download...")
            return

        url_to_sh = DATASETS_URLS[self.dataset_name]["shell_script_url"]
        logger.info(f"Downloading shell script from {url_to_sh} to {self.output_dir}")
        response = requests.get(url_to_sh)

        if not response.ok:
            raise ValueError(
                f"Failed to download shell script from {url_to_sh} with status code {response.status_code}"
            )

        shell_script_filename = "icsi_download.sh"
        with open(shell_script_filename, "w") as f:
            f.write(response.text)

        logger.info(f"Executing shell script from {self.output_dir / shell_script_filename}")
        subprocess.run(["chmod", "+x", shell_script_filename])
        subprocess.run([f"./{shell_script_filename}"])

    def _download_annotations(self) -> None:
        if Path("annotations.zip").exists():
            logger.info("Annotations already exist, skipping download...")
            return

        url_to_annotations = DATASETS_URLS[self.dataset_name]["annotations_url"]
        logger.info(f"Downloading annotations from {url_to_annotations} to {self.output_dir}")
        download_file(url_to_annotations, "annotations.zip")

        logger.info(f"Extracting annotations from {self.output_dir / 'annotations.zip'} to {self.output_dir}")
        with zipfile.ZipFile("annotations.zip", "r") as zip_ref:
            zip_ref.extractall()

    def _resolve_subsegments(
        self, segment: BeautifulSoup, audio_id: str
    ) -> tuple[list[float], list[float], list[str | int]]:
        timestamp_start = segment.get("starttime")
        timestamp_end = segment.get("endtime")
        speaker = segment.get("participant")

        segment_id = segment.get("nite:id")
        segment_type = segment.get("type")
        if segment_type == "supersegment":
            logger.info(f"{audio_id} -> Supersegment found at {segment_id}, trying to use subsegmenets")
            super_timestamp_start = timestamp_start
            super_timestamp_end = timestamp_end
            super_speaker = speaker

            subsegments = segment.find_all("segment", {"type": "subsegment"})
            subsegments_timestamp_start = []
            subsegments_timestamp_end = []
            subsegments_speakers = []
            subsegmenets_all_correct = True
            for subsegment in subsegments:
                sub_starttime = subsegment.get("starttime")
                sub_endtime = subsegment.get("endtime")
                sub_speaker = subsegment.get("participant")
                sub_id = subsegment.get("nite:id")

                if sub_starttime is None or sub_endtime is None or sub_speaker is None:
                    logger.warning(
                        f"{audio_id} -> Subsegment {sub_id} has one or more invalid values with starttime {sub_starttime}, endtime {sub_endtime} and speaker {sub_speaker} "
                        "Falling back to supersegment values"
                    )
                    subsegmenets_all_correct = False
                    break
            timestamp_start = [super_timestamp_start] if not subsegmenets_all_correct else subsegments_timestamp_start
            timestamp_end = [super_timestamp_end] if not subsegmenets_all_correct else subsegments_timestamp_end
            speaker = [super_speaker] if not subsegmenets_all_correct else subsegments_speakers
        else:
            timestamp_start = [timestamp_start]
            timestamp_end = [timestamp_end]
            speaker = [speaker]

        return timestamp_start, timestamp_end, speaker

    def _parse_to_rttm(self) -> None:
        # Annotations are in .xml format, we need to parse them to .rttm
        rttm_dir = Path("rttms")
        rttm_dir.mkdir(parents=True, exist_ok=True)

        xml_dir = Path("ICSI", "Segments")
        xml_files = sorted((xml_dir.glob("*.xml")))

        annotation_dict: dict[str, Annotation] = {}
        for xml_file in tqdm(
            xml_files,
            desc="Parsing XML files to RTTM",
            total=len(xml_files),
            unit="annotation",
        ):
            audio_id = xml_file.stem.split(".")[0]
            annotation = Annotation(uri=audio_id) if audio_id not in annotation_dict else annotation_dict[audio_id]

            with open(str(xml_file), "r") as f:
                soup = BeautifulSoup(f, "xml")

            # Filter out subsegments as we'll handle them using their parent segmenet i.e. type == "supersegment"
            all_segments = [s for s in soup.find_all("segment") if s.get("type") != "subsegment"]
            for segment in all_segments:
                timestamps_start, timestamps_end, speakers = self._resolve_subsegments(segment, audio_id)

                for timestamp_start, timestamp_end, speaker in zip(timestamps_start, timestamps_end, speakers):
                    logger.info(
                        f"{audio_id} -> Adding segment {timestamp_start} - {timestamp_end} for speaker {speaker}"
                    )
                    annotation[Segment(start=float(timestamp_start), end=float(timestamp_end))] = speaker

            if audio_id not in annotation_dict:
                annotation_dict[audio_id] = annotation

        for audio_id, annotation in annotation_dict.items():
            output_rttm_file = rttm_dir / f"{audio_id}.rttm"
            with open(str(output_rttm_file), "w") as f:
                annotation.write_rttm(f)

    def download(self) -> None:
        self._download_wav()
        self._download_annotations()
        self._parse_to_rttm()

    def create_dataset(self) -> dict[str, SpeakerDiarizationData]:
        rttm_dir = Path("rttms")
        wav_dir = Path("Signals")

        rttm_files = sorted((rttm_dir.glob("*.rttm")))
        wav_files = sorted((wav_dir.glob("*/*.wav")))
        wav_ids = [wav_file.parent.name for wav_file in wav_files]

        for rttm_file, wav_id in zip(rttm_files, wav_ids):
            if rttm_file.stem != wav_id:
                raise ValueError(f"RTTM file {rttm_file} and WAV file {wav_id} do not match")

        return {
            "test": SpeakerDiarizationData(
                split="test",
                audio_paths=[str(wav_file) for wav_file in wav_files],
                annotation_paths=[str(rttm_file) for rttm_file in rttm_files],
            )
        }


class AliMeetingsDataset(SpeakerDiarizationDataset):
    @property
    def dataset_name(self) -> str:
        return "ali-meetings"

    def _download_and_extract(self) -> None:
        url_to_audio_annot = DATASETS_URLS[self.dataset_name]["audio_annot_url"]
        logger.info(f"Downloading audio and annotations from {url_to_audio_annot} to {self.output_dir}")
        download_file(url_to_audio_annot, "audio_annot.tar.gz")
        logger.info(
            f"Extracting audio and annotations from {self.output_dir / 'audio_annot.tar.gz'} to {self.output_dir}"
        )
        with tarfile.open("audio_annot.tar.gz", "r") as tar_ref:
            tar_ref.extractall()

    def _convert_textgrid_to_rttm(self) -> None:
        textgrid_dir = Path("Test_Ali", "Test_Ali_far", "textgrid_dir")
        textgrid_files = sorted((textgrid_dir.glob("*.TextGrid")))

        rttm_dir = Path("rttms")
        rttm_dir.mkdir(parents=True, exist_ok=True)

        for textgrid_file in textgrid_files:
            logger.info(f"Converting {textgrid_file} to RTTM")
            textgrid = TextGrid.fromFile(str(textgrid_file))
            rttm_name = f"{textgrid_file.stem}.rttm"
            rttm_annotation = Annotation(uri=textgrid_file.stem)
            for tier in textgrid.tiers:
                speaker = tier.name
                for interval in tier.intervals:
                    timestamp_start = interval.minTime
                    timestamp_end = interval.maxTime
                    segment = Segment(start=timestamp_start, end=timestamp_end)
                    rttm_annotation[segment] = speaker

            logger.info(f"Writing RTTM file {rttm_name} to {rttm_dir}")
            with open(str(rttm_dir / rttm_name), "w") as f:
                rttm_annotation.write_rttm(f)

    def download(self) -> None:
        self._download_and_extract()
        self._convert_textgrid_to_rttm()

    def create_dataset(self) -> dict[str, SpeakerDiarizationData]:
        audio_dir = Path("Test_Ali", "Test_Ali_far", "audio_dir")
        rttm_dir = Path("rttms")

        audio_files = sorted((audio_dir.glob("*.wav")))
        rttm_files = sorted((rttm_dir.glob("*.rttm")))

        for audio_file, rttm_file in zip(audio_files, rttm_files):
            # For some reason the naming convention is slightly different for audio and annotations
            audio_stem = audio_file.stem  # e.g. R8001_M8004_MS801
            audio_id = "_".join(audio_stem.split("_")[:-1])
            rttm_id = rttm_file.stem  # e.g. R8001_M8004
            if audio_id != rttm_id:
                raise ValueError(f"Audio file {audio_file} and annotation file {rttm_file} do not match")

        return {
            "test": SpeakerDiarizationData(
                split="test",
                audio_paths=[str(audio_file) for audio_file in audio_files],
                annotation_paths=[str(rttm_file) for rttm_file in rttm_files],
            )
        }


class AIShell4Dataset(SpeakerDiarizationDataset):
    @property
    def dataset_name(self) -> str:
        return "aishell-4"

    def download(self) -> None:
        hf_repo_id = DATASETS_URLS[self.dataset_name]["hf_repo_id"]
        logger.info(f"Downloading audio and annotation for {self.dataset_name} dataset to {self.output_dir}")
        snapshot_download(
            repo_id=hf_repo_id,
            repo_type="dataset",
            local_dir=".",
            allow_patterns="test/*",
        )

    def create_dataset(self) -> dict[str, SpeakerDiarizationData]:
        audio_dir = Path("test", "wav")
        annot_dir = Path("test", "TextGrid")

        audio_files = sorted((audio_dir.glob("*.flac")))
        rttm_files = sorted((annot_dir.glob("*.rttm")))

        logger.info(f"Found {len(audio_files)} audio files and {len(rttm_files)} annotation files")
        logger.info("Veryfing that audio and annotation files match...")
        for audio_file, rttm_file in zip(audio_files, rttm_files):
            if audio_file.stem != rttm_file.stem:
                raise ValueError(f"Audio file {audio_file} and annotation file {rttm_file} do not match")
        logger.info("Audio and annotation files match")

        return {
            "test": SpeakerDiarizationData(
                split="test",
                audio_paths=[str(audio_file) for audio_file in audio_files],
                annotation_paths=[str(rttm_file) for rttm_file in rttm_files],
            )
        }


class AmericanLifeDataset(SpeakerDiarizationDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Importing here as if you don't have kaggle credentials setup it will fail
        from kaggle.api.kaggle_api_extended import KaggleApi

        self.kaggle_api = KaggleApi()
        logger.info("Authenticating to Kaggle...")
        try:
            self.kaggle_api.authenticate()
        except Exception as e:
            logger.error(
                "Failed to authenticate to Kaggle make sure you have a Kaggle API key in your ~/.kaggle/kaggle.json file."
            )
            raise e

    @property
    def dataset_name(self) -> str:
        return "american-life"

    def _get_audio_url(self, episode: int) -> str | None:
        base_episode_url = DATASETS_URLS[self.dataset_name]["base_episode_url"]
        episode_url = base_episode_url + str(episode)

        response = requests.get(episode_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        download_link = soup.find("li", class_="download")

        if download_link is None:
            return None

        return download_link.find("a")["href"]

    def _get_episode_to_download(self) -> set[int]:
        annot_dir = Path("kaggle_annotations")
        annot_files = sorted((annot_dir.glob("*.json")))
        # Get only the transcripts
        annot_files = [annot_file for annot_file in annot_files if "transcripts" in annot_file.stem]
        # Get episode number
        eps_to_download = []
        for annot_file in annot_files:
            data = load_json(annot_file)
            episodes_in_split = [int(key.split("-")[1]) for key in data.keys()]
            eps_to_download.extend(episodes_in_split)
        return set(eps_to_download)

    def _download_audio(self) -> None:
        audio_dir = Path("audio")
        audio_dir.mkdir(parents=True, exist_ok=True)
        episodes = self._get_episode_to_download()
        for episode in episodes:
            audio_url = self._get_audio_url(episode)
            if audio_url is None:
                logger.info(f"No download link found for episode {episode}...")
                continue
            audio_file_path = audio_dir / f"episode_{episode}.mp3"
            download_file(audio_url, str(audio_file_path))

    def _download_annotations(self) -> None:
        annot_dir = Path("kaggle_annotations")
        annot_dir.mkdir(parents=True, exist_ok=True)

        kaggle_dataset = DATASETS_URLS[self.dataset_name]["kaggle_dataset_id"]
        logger.info(f"Downloading dataset {kaggle_dataset} to {self.output_dir}")
        self.kaggle_api.dataset_download_files(kaggle_dataset, path=str(annot_dir), unzip=False)

        filename = "this-american-life-podcast-transcriptsalignments.zip"
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(str(annot_dir))

    def download(self) -> None:
        self._download_annotations()
        self._download_audio()

    def _create_annotation_from_json(self, segments: list[dict], speaker_map: dict[str, str]) -> Annotation:
        annot = Annotation()
        for segment in segments:
            speaker_id = speaker_map[segment["speaker"]]
            segment_start = segment["utterance_start"]
            segment_end = segment["utterance_end"]

            annot[Segment(start=segment_start, end=segment_end)] = speaker_id
        return annot

    def _parse_annot_json(self, json_path: str) -> dict[str, Annotation]:
        speaker_map_file = Path("kaggle_annotations") / DATASETS_URLS[self.dataset_name]["speaker_map"]
        speaker_map = load_json(speaker_map_file)
        annotations: dict[str, list[dict]] = load_json(Path("kaggle_annotations") / json_path)
        parsed_annotations: dict[str, Annotation] = {}
        for episode, segments in annotations.items():
            episode_number = int(episode.split("-")[1])
            uri = f"episode_{episode_number}"
            parsed_annotation = self._create_annotation_from_json(segments, speaker_map)
            parsed_annotation.uri = uri
            parsed_annotations[uri] = parsed_annotation
        return parsed_annotations

    def create_dataset(self) -> dict[str, SpeakerDiarizationData]:
        annotation_splits = DATASETS_URLS[self.dataset_name]["annotation_splits"]
        splits: list[str] = list(annotation_splits.keys())

        for split, json_path in annotation_splits.items():
            parsed_annotations = self._parse_annot_json(json_path)
            split_dir = Path(f"{split}_annotations")
            split_dir.mkdir(parents=True, exist_ok=True)
            for episode, annotation in parsed_annotations.items():
                output_file = split_dir / f"{episode}.rttm"
                with open(str(output_file), "w") as f:
                    annotation.write_rttm(f)

        audio_dir = Path("audio")
        data = {}
        for split in splits:
            annot_dir = Path(f"{split}_annotations")
            rttm_files = sorted((annot_dir.glob("*.rttm")))
            audio_files = [audio_dir / f"{rttm_file.stem}.mp3" for rttm_file in rttm_files]
            data[split] = SpeakerDiarizationData(
                split=split,
                audio_paths=[str(audio_file) for audio_file in audio_files],
                annotation_paths=[str(rttm_file) for rttm_file in rttm_files],
            )
        return data


class AVAAvdDataset(SpeakerDiarizationDataset):
    videos_dir = Path("videos")  # Directory to download videos
    splits_dir = Path("splits")  # Directory to download the .list with the split infos
    audios_dir = Path("audios")  # Directory to store the audio files
    rttms_dir = Path("rttms")  # Directory to store the rttm files
    temp_audios_dir = Path("temp_audios")  # Directory to store the audio files

    @property
    def dataset_name(self) -> str:
        return "ava-avd"

    def _download_lists(self) -> None:
        videos_url = DATASETS_URLS[self.dataset_name]["video_list_url"]
        train_url = DATASETS_URLS[self.dataset_name]["train_list_url"]
        val_url = DATASETS_URLS[self.dataset_name]["val_list_url"]
        test_url = DATASETS_URLS[self.dataset_name]["test_list_url"]
        offsets_url = DATASETS_URLS[self.dataset_name]["offsets_url"]

        self.splits_dir.mkdir(parents=True, exist_ok=True)

        download_file(videos_url, str(self.splits_dir / "video.list"))
        download_file(train_url, str(self.splits_dir / "train.list"))
        download_file(val_url, str(self.splits_dir / "validation.list"))
        download_file(test_url, str(self.splits_dir / "test.list"))
        download_file(offsets_url, str(self.splits_dir / "offsets.txt"))

    # Adapted from https://github.com/showlab/AVA-AVD/blob/main/dataset/scripts/download.py#L3
    def _download_videos(self) -> None:
        videos_dir = Path("videos")
        videos_dir.mkdir(parents=True, exist_ok=True)
        video_ids = load_list(str(self.splits_dir / "video.list"))

        for i, video in enumerate(video_ids):
            logger.info(f"Downloading {video}[{i + 1}]/[{len(video_ids)}]")
            cmd = f"wget -P {str(videos_dir)} https://s3.amazonaws.com/ava-dataset/trainval/{video.strip()}"
            subprocess.call(cmd, shell=True)

    # Adapted from https://github.com/showlab/AVA-AVD/blob/main/dataset/scripts/download.py#L13
    def _download_annotations(self) -> None:
        logger.info("Downloading annotations")
        cmd = "gdown --id 18kjJJbebBg7e8umI6HoGE4_tI3OWufzA"
        subprocess.call(cmd, shell=True)
        logger.info("Extracting annotations")
        cmd = "tar -xvf annotations.tar.gz"
        # This will add a `rttms` directory and others (which we don't need)
        subprocess.call(cmd, shell=True)
        os.remove("annotations.tar.gz")

    def download(self) -> None:
        self._download_lists()
        self._download_videos()
        self._download_annotations()

    # Adapted from https://github.com/showlab/AVA-AVD/blob/main/dataset/scripts/preprocessing.py#L76
    def _split_waves(self) -> None:
        self.audios_dir.mkdir(parents=True, exist_ok=True)
        self.temp_audios_dir.mkdir(parents=True, exist_ok=True)
        if not self.rttms_dir.exists():
            raise ValueError("RTTM directory does not exist")

        for video in self.videos_dir.iterdir():
            video_uid = video.stem
            logger.info(f"Extracting audio from {video} to {self.temp_audios_dir / f'{video_uid}.wav'}")
            cmd = f"ffmpeg -y -i {video} -qscale:a 0 -ac 1 -vn -threads 6 -ar 16000 {self.temp_audios_dir / f'{video_uid}.wav'}"
            subprocess.call(cmd, shell=True)

            rttms = self.rttms_dir.glob(f"{video_uid}*.rttm")
            for rttm in sorted(rttms):
                logger.info(f"Processing {rttm}")
                rttm_uid = rttm.stem
                rttm_annotation = load_rttm(rttm)[rttm_uid]
                # Pyannote returns the segments in order, so we can just take the first and last
                segments = list(rttm_annotation.itersegments())
                min_timestamp = segments[0].start
                max_timestamp = segments[-1].end
                logger.info(f"Found for {rttm_uid}: Min timestamp: {min_timestamp}, max timestamp: {max_timestamp}")
                # Using wavfile to keep consistency with original implementation
                sample_rate, wave = wavfile.read(f"{self.temp_audios_dir}/{video_uid}.wav")
                assert sample_rate == 16000

                wave = wave[int(min_timestamp * sample_rate) : int(max_timestamp * sample_rate)]
                logger.info(f"Writing {rttm_uid} to {self.audios_dir}/{rttm_uid}.wav")
                wavfile.write(f"{self.audios_dir}/{rttm_uid}.wav", sample_rate, wave)
        shutil.rmtree(f"{self.temp_audios_dir}")

    def offset_annotation(self, annotation: Annotation) -> Annotation:
        new_annotation = Annotation(uri=annotation.uri)
        offset = list(annotation.itersegments())[0].start
        logger.info(f"Applying offset {offset}s to all segments of {annotation.uri}")
        for segment, _, label in annotation.itertracks(yield_label=True):
            new_annotation[Segment(segment.start - offset, segment.end - offset)] = label
        return new_annotation

    def fix_annotations(self) -> None:
        for rttm in self.rttms_dir.glob("*.rttm"):
            annotation = load_rttm(rttm)[rttm.stem]
            offset_annotation = self.offset_annotation(annotation)
            with open(rttm, "w") as f:
                offset_annotation.write_rttm(f)

    def create_dataset(self) -> dict[str, SpeakerDiarizationData]:
        data: dict[str, SpeakerDiarizationData] = {}
        self._split_waves()
        self.fix_annotations()
        split_lists = [f for f in self.splits_dir.glob("*.list") if f.stem != "video"]

        for split_list in split_lists:
            split_name = split_list.stem

            rttm_split_dir = self.rttms_dir / split_name
            rttm_split_dir.mkdir(exist_ok=True)

            audio_split_dir = self.audios_dir / split_name
            audio_split_dir.mkdir(exist_ok=True)

            files_in_split = load_list(str(split_list))
            for file_in_split in files_in_split:
                filename = file_in_split.strip()
                path_from = self.rttms_dir / f"{filename}.rttm"
                path_to = rttm_split_dir / f"{filename}.rttm"
                logger.info(f"Copying {path_from} to {path_to}")
                shutil.copy(path_from, path_to)

                path_from = self.audios_dir / f"{filename}.wav"
                path_to = audio_split_dir / f"{filename}.wav"
                logger.info(f"Copying {path_from} to {path_to}")
                shutil.copy(path_from, path_to)

            audio_paths = sorted((audio_split_dir.glob("*.wav")))
            annotation_paths = sorted((rttm_split_dir.glob("*.rttm")))

            data[split_name] = SpeakerDiarizationData(
                split=split_name,
                audio_paths=[str(audio_path) for audio_path in audio_paths],
                annotation_paths=[str(annotation_path) for annotation_path in annotation_paths],
            )

        return data


class DIHAR3DDataset(SpeakerDiarizationDataset):
    # DIHARD-3 dataset is not available for download and needs to be acquired from LDC
    # https://catalog.ldc.upenn.edu/LDC2022S14
    MAIN_DATASET_DIR = Path(
        os.getenv("DIHARD_DATASET_DIR") or Path("~", "third_dihard_challenge_eval", "data").expanduser()
    )

    @property
    def audio_dir(self) -> Path:
        return self.MAIN_DATASET_DIR / "flac"

    @property
    def rttm_dir(self) -> Path:
        return self.MAIN_DATASET_DIR / "rttm"

    @property
    def uem_dir(self) -> Path:
        return self.MAIN_DATASET_DIR / "uem_scoring"

    @property
    def dataset_name(self) -> str:
        return "dihard-3"

    def download(self) -> None:
        # There's no downloaded needed for this dataset
        return

    def _get_modalities(self, split: str) -> dict[str, list[str]]:
        uem_files = [i for i in (self.uem_dir / split).glob("*.uem") if i.stem != "all"]
        dfs = []
        for uem_file in uem_files:
            uem = load_uem(uem_file)
            df = pd.DataFrame(uem.keys(), columns=["uri"])
            df["modality"] = uem_file.stem
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True).sort_values("uri")
        return {"modality": df["modality"].tolist()}

    def create_dataset(self) -> dict[str, SpeakerDiarizationData]:
        data = {}
        splits = ["core", "full"]
        for split in splits:
            uem_file = self.uem_dir / split / "all.uem"
            uems: dict[str, Timeline] = load_uem(uem_file)

            audio_paths = []
            uem_paths = []
            rttm_paths = []

            metadata = self._get_modalities(split)
            uems_split_dir = Path(split) / "uem"
            uems_split_dir.mkdir(parents=True, exist_ok=True)

            for uri, uem in uems.items():
                audio_file = self.audio_dir / f"{uri}.flac"
                rttm_file = self.rttm_dir / f"{uri}.rttm"

                logger.info(f"Writing {uri} to {uems_split_dir / f'{uri}.uem'}")
                uem_path = uems_split_dir / f"{uri}.uem"
                with open(uem_path, "w") as f:
                    uem.write_uem(f)

                audio_paths.append(str(audio_file))
                rttm_paths.append(str(rttm_file))
                uem_paths.append(str(uem_path))

            data[split] = SpeakerDiarizationData(
                split=split,
                audio_paths=audio_paths,
                annotation_paths=rttm_paths,
                uem_paths=uem_paths,
                metadata=metadata,
            )

        return data


class CallHomeDataset(SpeakerDiarizationDataset):
    # Path to the main direcotry with the dataset from CallHome
    # You have to acquire the dataset by buying it from https://catalog.ldc.upenn.edu/LDC2001S97
    AUDIO_ROOT = (
        os.getenv("CALLHOME_AUDIO_ROOT")
        or Path(
            "~",
            "callhome",
            "nist_recognition_evaluation",
            "r65_8_1",
            "sid00sg1",
            "data",
        ).expanduser()
    )
    RTTM_DIR = Path("rttms")
    AUDIO_DIR = Path("audios")
    ANNOTATION_DIR = Path("sre2000-key")
    SPH2PIPE_DIR = Path("sph2pipe")

    @property
    def dataset_name(self) -> str:
        return "callhome"

    def _download_split_lists(self) -> None:
        """Download the part1 and part2 split files."""
        logger.info("Downloading split list files...")
        # Download part1 and part2
        for split in ["part1", "part2"]:
            download_file(
                DATASETS_URLS[self.dataset_name][f"{split}_url"],
                f"{split}.txt",
            )

    def _download_annotations(self) -> None:
        """Download and extract the annotation files."""
        logger.info("Downloading and extracting annotations...")
        # Download annotations
        annotations_file = "sre2000-key.tar.gz"
        download_file(
            DATASETS_URLS[self.dataset_name]["annotations_url"],
            annotations_file,
        )
        with tarfile.open(annotations_file, "r:gz") as tar:
            tar.extractall()
        os.remove(annotations_file)  # Clean up the tar file after extraction

    def download(self) -> None:
        """Download all necessary dataset files."""
        self._download_split_lists()
        self._download_annotations()

    def _docker_run(self, sph_path: Path, wav_path: Path) -> None:
        """Run make docker-run from sph2pipe directory."""
        # Step 1. Copy the .sph file to the sph2pipe directory - because of the docker mounted volume
        logger.info(f"Copying {sph_path} to {self.SPH2PIPE_DIR / sph_path.name}")
        shutil.copy(sph_path, self.SPH2PIPE_DIR / sph_path.name)
        # Step 2. Run the docker-run command
        logger.info(f"Running docker-run command: make docker-run INPUT={sph_path.name} OUTPUT={wav_path.name}")
        cmd = f"make docker-run INPUT={sph_path.name} OUTPUT={wav_path.name}"
        subprocess.run(cmd, shell=True, check=True, cwd=self.SPH2PIPE_DIR)
        # Step 3. Move the .wav from the sph2pipe directory to the output directory
        logger.info(f"Moving {self.SPH2PIPE_DIR / wav_path.name} to {wav_path}")
        shutil.move(self.SPH2PIPE_DIR / wav_path.name, wav_path)
        # Step 4. Clean up the .sph copy in the sph2pipe directory
        logger.info(f"Cleaning up {self.SPH2PIPE_DIR / sph_path.name}")
        (self.SPH2PIPE_DIR / sph_path.name).unlink()

    def _convert_sph_to_wav(self, sph_path: Path, wav_path: Path) -> None:
        """Convert a SPH audio file to WAV format using sph2pipe in docker and resample to 16kHz."""
        logger.info(f"Converting {sph_path} to {wav_path}")
        try:
            # Step 1: Convert SPH to WAV at original 8kHz using docker
            temp_wav = wav_path.with_name(f"temp_{wav_path.name}")
            self._docker_run(sph_path, temp_wav)
            # Step 2: Resample from 8kHz to 16kHz
            logger.info(f"Resampling {temp_wav} from 8kHz to 16kHz")
            cmd_resample = f"ffmpeg -y -i {temp_wav} -ar 16000 {wav_path}"
            subprocess.run(
                cmd_resample,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Clean up temporary file
            temp_wav.unlink()

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to convert {sph_path} to WAV: {e}")
            if temp_wav.exists():
                temp_wav.unlink()
            raise

    def _setup_sph2pipe(self) -> None:
        """Clone the sph2pipe repository and build the executable."""
        logger.info("Cloning sph2pipe repository...")
        if not self.SPH2PIPE_DIR.exists():
            subprocess.run(
                f"git clone {DATASETS_URLS[self.dataset_name]['sph2pipe_repo_url']}",
                shell=True,
                check=True,
            )
        # Build the docker image based on the instructions in sph2pipe README
        # See https://github.com/EduardoPach/sph2pipe?tab=readme-ov-file#docker-usage
        logger.info("Building docker image...")
        subprocess.run("make docker-build", shell=True, check=True, cwd="sph2pipe")
        logger.info("Docker image built successfully")

    def _setup_directories(self) -> None:
        """Create necessary directories for the dataset."""
        for directory in [self.RTTM_DIR, self.AUDIO_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

    def _process_audio_file(
        self,
        uri: str,
        split: str,
        audio_split_dir: Path,
        rttm_split_dir: Path,
        annotations: dict,
    ) -> tuple[str | None, str | None]:
        """Process a single audio file and its annotation."""
        audio_path = self.AUDIO_ROOT / f"{uri.strip()}.sph"
        final_audio_path = audio_split_dir / f"{uri.strip()}.wav"
        final_rttm_path = rttm_split_dir / f"{uri.strip()}.rttm"

        if not audio_path.exists():
            logger.warning(f"{split.capitalize()} audio file {audio_path} does not exist")
            return None, None

        if uri.strip() not in annotations:
            logger.warning(f"{split.capitalize()} annotation {uri} does not exist")
            return None, None

        # Write RTTM file
        with open(final_rttm_path, "w") as f:
            annotations[uri.strip()].write_rttm(f)

        # Convert audio to WAV
        self._convert_sph_to_wav(audio_path, final_audio_path)

        return str(final_audio_path), str(final_rttm_path)

    def create_dataset(self) -> dict[str, SpeakerDiarizationData]:
        """Create the dataset by processing audio files and annotations."""
        # Load annotations and split lists
        annotations = load_rttm(str(self.ANNOTATION_DIR / "fullref.rttm"))
        splits = {split: load_list(f"{split}.txt") for split in ["part1", "part2"]}

        self._setup_directories()
        self._setup_sph2pipe()

        # Process each split
        data = {}
        total_num_files = sum(len(split_list) for split_list in splits.values())
        with tqdm(total=total_num_files, desc="Processing files") as pbar:
            for split, split_list in splits.items():
                rttm_split_dir = self.RTTM_DIR / split
                audio_split_dir = self.AUDIO_DIR / split
                rttm_split_dir.mkdir(parents=True, exist_ok=True)
                audio_split_dir.mkdir(parents=True, exist_ok=True)

                audio_paths = []
                rttm_paths = []

                for uri in split_list:
                    pbar.set_description(f"Processing {uri}")
                    audio_path, rttm_path = self._process_audio_file(
                        uri, split, audio_split_dir, rttm_split_dir, annotations
                    )
                    if audio_path and rttm_path:
                        audio_paths.append(audio_path)
                        rttm_paths.append(rttm_path)
                    pbar.update(1)
                    pbar.set_description(f"Finished {uri}")
                if not audio_paths:
                    logger.warning(f"No valid files found for split {split}")
                    continue

                data[split] = SpeakerDiarizationData(
                    split=split,
                    audio_paths=audio_paths,
                    annotation_paths=rttm_paths,
                )

        if not data:
            raise ValueError("No valid data found in any split")

        return data


class Ego4dDataset(SpeakerDiarizationDataset):
    ego4d_access_id: str = os.getenv("EGO4D_ACCESS_ID")
    ego4d_access_key: str = os.getenv("EGO4D_ACCESS_KEY")
    ego4d_profile: str = "ego4d"
    download_dir: Path = Path("ego4d_data")
    validation_clips_url: str = "https://raw.githubusercontent.com/EGO4D/audio-visual/refs/heads/main/diarization/audio-only/data_preparation/clips_val.txt"
    rttm_generator_url: str = "https://raw.githubusercontent.com/EGO4D/audio-visual/refs/heads/main/diarization/audio-only/data_preparation/generate_voice_rttms.py"
    rttm_dir: Path = Path("rttms")
    audio_dir: Path = Path("audio")

    @property
    def dataset_name(self) -> str:
        return "ego4d"

    def _download_video_files(self, split_path: Path) -> None:
        split = split_path.stem
        # Create the download directory if it doesn't exist
        (self.download_dir / split).mkdir(parents=True, exist_ok=True)
        cmd = [
            "ego4d",
            "--output_directory",
            str(self.download_dir / split),
            "--datasets",
            "clips",
            "annotations",
            "--video_uid_file",
            str(split_path),
            "--aws_profile_name",
            self.ego4d_profile,
            "-y",
        ]
        # Run command and check for errors
        try:
            logger.info(f"Downloading video files for split {split}...")
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download video files for split {split} while running: {cmd}\nError: {e}")
            raise

    def download(self) -> None:
        aws_profile = AwsProfileInfo(
            profile=self.ego4d_profile,
            access_id=self.ego4d_access_id,
            access_key=self.ego4d_access_key,
        )
        aws_profile.create_profile()
        download_file(self.validation_clips_url, "validation.txt")
        self._download_video_files(Path("validation.txt"))

    def _generate_rttms(self, split: str) -> None:
        download_file(self.rttm_generator_url, "generate_voice_rttms.py")
        data_dir = self.download_dir / split
        rttms_output_dir = self.rttm_dir / split
        lab_output_dir = Path("labs") / split
        rttms_output_dir.mkdir(parents=True, exist_ok=True)
        lab_output_dir.mkdir(parents=True, exist_ok=True)
        json_file = data_dir / "v2" / "annotations" / "av_val.json"

        cmd = [
            "poetry",
            "run",
            "python",
            "generate_voice_rttms.py",
            "--json-filename",
            str(json_file),
            "--rttm-output-dir",
            str(rttms_output_dir),
            "--lab-output-dir",
            str(lab_output_dir),
        ]
        # Run the RTTM generator
        try:
            logger.info("Generating RTTMs...")
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to generate RTTMs while running: {cmd}\nError: {e}")
            raise

    def _extract_audio(self, video_path: Path, audio_path: Path) -> None:
        cmd = f"ffmpeg -i {video_path} -acodec pcm_s16le -ac 1 -ar 16000 {audio_path}"
        try:
            logger.info(f"Extracting audio from {video_path} to {audio_path}")
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to extract audio while running: {cmd}\nError: {e}")
            raise

    def _convert_videos_to_audio(self, split: str) -> None:
        data_dir = self.download_dir / split
        output_audio_dir = self.audio_dir / split
        output_audio_dir.mkdir(parents=True, exist_ok=True)
        video_dir = data_dir / "v2" / "clips"

        for video_path in video_dir.glob("*.mp4"):
            basename = video_path.stem
            audio_path = output_audio_dir / f"{basename}.wav"
            self._extract_audio(video_path, audio_path)

    def create_dataset(self) -> dict[str, SpeakerDiarizationData]:
        splits = os.listdir(self.download_dir)
        for split in splits:
            logger.info(f"Processing split {split}...")
            self._generate_rttms(split)
            self._convert_videos_to_audio(split)

        data = {}
        for split in splits:
            audio_split_dir = self.audio_dir / split
            rttm_split_dir = self.rttm_dir / split
            audio_paths = sorted((audio_split_dir.glob("*.wav")))
            rttm_paths = sorted((rttm_split_dir.glob("*.rttm")))

            if any(audio_path.stem != rttm_path.stem for audio_path, rttm_path in zip(audio_paths, rttm_paths)):
                raise ValueError(f"Mismatch in audio and RTTM files for split {split}")

            data[split] = SpeakerDiarizationData(
                split=split,
                audio_paths=[str(audio_path) for audio_path in audio_paths],
                annotation_paths=[str(rttm_path) for rttm_path in rttm_paths],
            )
        return data


# ForCallHome English Transcript which contains transcript with word-level speaker labels.
# To get access to the data on has to license it from LDC.
# Transcripts can be found at: https://catalog.ldc.upenn.edu/LDC97T14
# Audios can be found at: https://catalog.ldc.upenn.edu/LDC97S42
class CallHomeEnglishTranscript(SpeakerDiarizationDataset):
    audio_data_dir_path: Path = (
        os.getenv("CALLHOME_ENGLISH_AUDIO_DIR")
        or Path(
            "~",
            "callhome_eng",
            "data",
        ).expanduser()
    )
    transcript_data_dir_path: Path = (
        os.getenv("CALLHOME_ENGLISH_ANNOT_DIR")
        or Path(
            "~",
            "callhome_english_trans_970711",
            "transcrpt",
        ).expanduser()
    )
    split_name_mapping: dict[str, str] = {
        "devtest": "validation",
        "evaltest": "test",
        "train": "train",
    }
    dst_audio_dir: Path = Path("audios")
    dst_rttm_dir: Path = Path("rttms")
    dst_transcript_dir: Path = Path("transcripts")
    sph2pipe_url: str = "https://github.com/EduardoPach/sph2pipe.git"
    sph2pipe_dir = Path("sph2pipe")

    @property
    def dataset_name(self) -> str:
        return "callhome-english"

    def _clean_content(self, content: str) -> str:
        """Clean the content text according to specific rules.

        Args:
            content: The raw content text

        Returns:
            Cleaned content text
        """
        # Remove sound made by the talker {text}
        content = re.sub(r"\{[^}]*\}", "", content)

        # Remove sound not made by the talker [text]
        content = re.sub(r"\[[^\]]*\]", "", content)

        # Remove annotator comments [ [ text ] ]
        content = re.sub(r"\[\s*\[\s*[^\]]*\s*\]\s*\]", "", content)

        # Remove end of continuous or intermittent sound [/text]
        content = re.sub(r"\[/[^\]]*\]", "", content)

        # Keep aside talking //text//
        content = re.sub(r"//([^/]*)//", r"\1", content)

        # Keep mispronounced word +text+
        content = re.sub(r"\+([^+]*)\+", r"\1", content)

        # Remove ampersands to mark names and places &text
        content = re.sub(r"&([^\s]+)", r"\1", content)

        # Remove partial words -text or text-
        content = re.sub(r"(?<![A-Za-z-])-[A-Za-z]+\b|\b[A-Za-z]+-(?![A-Za-z-])", "", content)
        # Remove multiple spaces
        content = re.sub(r"\s{2,}", " ", content).strip()

        # Remove % for non-lexemes %text
        content = re.sub(r"%([^\s]+)", r"\1", content)

        # Remove ** from idiosyncratic word **text**
        content = re.sub(r"\*\*([^*]+)\*\*", r"\1", content)

        # Remove unintelligible symbols (( ))
        content = re.sub(r"\(\(\s*\)\)", "[UNINTELLIGIBLE]", content)

        # Remove unrecognized language symbols <? (( ))?
        content = re.sub(r"<\?\s*\(\(\s*\)\)\s*>", "[UNINTELLIGIBLE]", content)

        # Keep text in different language, remove the <>
        content = re.sub(r"<([^>]+)>", r"\1", content)

        # Keep unintelligible text, remove (()) from it
        content = re.sub(r"\(\(([^)]+)\)\)", r"\1", content)

        return content

    def parse_transcript_to_df(self, file: Path) -> pd.DataFrame:
        """
        Parse a transcript file with lines of the form:
            <start_timestamp> <end_timestamp> <speaker_tag>: <text>
        where:
            - timestamps are ints or floats (e.g., 451.37)
            - speaker_tag is a single uppercase letter
            - text may be wrapped across multiple lines

        Returns a DataFrame with columns: start, end, content, speaker.
        """
        # Matches: start, end, speaker, text
        line_re = re.compile(
            r"^\s*"  # optional leading spaces
            r"(\d+(?:\.\d+)?)\s+"  # start timestamp
            r"(\d+(?:\.\d+)?)\s+"  # end timestamp
            r"([A-Z]):\s*"  # speaker tag + colon
            r"(.*)$"  # text (may be empty)
        )

        lines = file.read_text().splitlines()

        rows = []
        current = None  # holds the current utterance we’re building

        for raw_line in lines:
            line = raw_line.rstrip("\n")

            # Skip empty lines entirely
            if not line.strip():
                continue

            # Skip comment / metadata lines starting with '#'
            if line.lstrip().startswith("#"):
                continue

            m = line_re.match(line)
            if m:
                # Start of a new utterance: flush the previous one
                if current is not None:
                    # Normalize whitespace in accumulated content
                    current["content"] = " ".join(current["content"].split())
                    rows.append(current)

                start_str, end_str, speaker, text = m.groups()
                current = {
                    "start": float(start_str),
                    "end": float(end_str),
                    "speaker": speaker,
                    "content": text.strip(),
                }
            else:
                # Continuation of the previous line’s content
                if current is None:
                    # Malformed file line before any utterance; ignore
                    continue
                # Append continuation with a space, then normalize later
                current["content"] += " " + line.strip()

        # Don’t forget the last utterance
        if current is not None:
            current["content"] = " ".join(current["content"].split())
            rows.append(current)

        # Ensure column order
        return pd.DataFrame(rows, columns=["start", "end", "content", "speaker"])

    def _preprocess_transcript(self, transcript: Path) -> tuple[list[str], list[str]]:
        """Process a CallHome English transcript file.

        Args:
            transcript: The path to the transcript file

        Returns:
            A tuple containing:
            - List of words
            - List of speaker labels (A or B) corresponding to each word
        """
        df = self.parse_transcript_to_df(transcript)

        df = (
            df
            # Clean the content column
            .assign(content=lambda df: df["content"].apply(self._clean_content))
            # Kepp only lines with content
            .pipe(lambda df: df.loc[df["content"].str.strip().str.len() > 0])
        )

        # Get speech start and end
        start_speech_segments = df["start"].tolist()
        end_speech_segments = df["end"].tolist()
        speaker_speech_segments = df["speaker"].tolist()

        # Get words and speakers
        word_speakers, words = (
            df.assign(words=df["content"].str.split())
            .explode("words")[["speaker", "words"]]
            .transpose()
            .values.tolist()
        )

        return (
            words,
            word_speakers,
            speaker_speech_segments,
            start_speech_segments,
            end_speech_segments,
        )

    def _setup_sph2pipe(self) -> None:
        """Clone the sph2pipe repository and build the executable."""
        logger.info("Cloning sph2pipe repository...")
        if not self.sph2pipe_dir.exists():
            subprocess.run(
                f"git clone {self.sph2pipe_url}",
                shell=True,
                check=True,
            )
        # Build the docker image based on the instructions in sph2pipe README
        # See https://github.com/EduardoPach/sph2pipe?tab=readme-ov-file#docker-usage
        logger.info("Building docker image...")
        subprocess.run("make docker-build", shell=True, check=True, cwd="sph2pipe")
        logger.info("Docker image built successfully")

    def _docker_run(self, sph_path: Path, wav_path: Path) -> None:
        """Run make docker-run from sph2pipe directory."""
        # Step 1. Copy the .sph file to the sph2pipe directory - because of the docker mounted volume
        logger.info(f"Copying {sph_path} to {self.sph2pipe_dir / sph_path.name}")
        shutil.copy(sph_path, self.sph2pipe_dir / sph_path.name)
        # Step 2. Run the docker-run command
        logger.info(f"Running docker-run command: make docker-run INPUT={sph_path.name} OUTPUT={wav_path.name}")
        cmd = f"make docker-run INPUT={sph_path.name} OUTPUT={wav_path.name}"
        subprocess.run(cmd, shell=True, check=True, cwd=self.sph2pipe_dir)
        # Step 3. Move the .wav from the sph2pipe directory to the output directory
        logger.info(f"Moving {self.sph2pipe_dir / wav_path.name} to {wav_path}")
        shutil.move(self.sph2pipe_dir / wav_path.name, wav_path)
        # Step 4. Clean up the .sph copy in the sph2pipe directory
        logger.info(f"Cleaning up {self.sph2pipe_dir / sph_path.name}")
        (self.sph2pipe_dir / sph_path.name).unlink()

    def _convert_sph_to_wav(self, sph_path: Path, wav_path: Path, between: tuple[float, float]) -> None:
        """Convert a SPH audio file to WAV format using sph2pipe in docker and resample to 16kHz."""
        logger.info(f"Converting {sph_path} to {wav_path}")
        try:
            # Step 1: Convert SPH to WAV at original 8kHz using docker
            temp_wav = wav_path.with_name(f"temp_{wav_path.name}")
            self._docker_run(sph_path, temp_wav)
            # Step 2: Resample from 8kHz to 16kHz
            logger.info(f"Resampling {temp_wav} from 8kHz to 16kHz")
            cmd_resample = f"ffmpeg -y -i {temp_wav} -ar 16000 {wav_path}"
            subprocess.run(
                cmd_resample,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            # Step 3: Trim the audio to the between region
            logger.info(f"Trimming {wav_path} to {between}")
            start, end = between
            wav, sr = sf.read(wav_path)
            wav = wav[int(start * sr) : int(end * sr)]
            sf.write(wav_path, wav, sr)

            # Clean up temporary file
            temp_wav.unlink()

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to convert {sph_path} to WAV: {e}")
            if temp_wav.exists():
                temp_wav.unlink()
            raise

    def _save_word_speaker_pairs(self, dst_transcript_file: Path, word_speakers: list[str], words: list[str]) -> None:
        df = pd.DataFrame({"word": words, "speaker": word_speakers})
        df.to_csv(dst_transcript_file, index=False)

    def _save_rttm(
        self,
        rttm_file: Path,
        speakers: list[str],
        start_times: list[float],
        end_times: list[float],
    ) -> None:
        """Save speaker segments to RTTM format file using pyannote.core.

        Args:
            rttm_file: Path to output RTTM file
            speakers: List of speaker IDs for each segment
            start_times: List of segment start times in seconds
            end_times: List of segment end times in seconds
        """
        annotation = Annotation(uri=rttm_file.stem)
        offset = min(start_times)
        for spk, start, end in zip(speakers, start_times, end_times):
            segment = Segment(start - offset, end - offset)
            annotation[segment] = spk

        with rttm_file.open("w") as f:
            annotation.write_rttm(f)

    def _process_split(self, split_key: str, split_name: str) -> SpeakerDiarizationData:
        transcript_dir = self.transcript_data_dir_path / split_key
        source_audio_dir = self.audio_data_dir_path / split_key

        transcript_files: list[Path] = sorted((transcript_dir.glob("*.txt")))
        audio_files: list[Path] = sorted((source_audio_dir.glob("*.sph")))

        split_audio_dir = self.dst_audio_dir / split_name
        split_rttm_dir = self.dst_rttm_dir / split_name
        split_transcript_dir = self.dst_transcript_dir / split_name

        split_audio_dir.mkdir(parents=True, exist_ok=True)
        split_rttm_dir.mkdir(parents=True, exist_ok=True)
        split_transcript_dir.mkdir(parents=True, exist_ok=True)

        dst_audio_files: list[str] = []
        dst_words: list[list[str]] = []
        dst_speakers: list[list[str]] = []
        dst_rttm_files: list[list[str]] = []

        for audio_file, transcript_file in tqdm(zip(audio_files, transcript_files)):
            (
                words,
                word_speakers,
                speaker_speech_segments,
                start_speech_segments,
                end_speech_segments,
            ) = self._preprocess_transcript(transcript_file)
            dst_words.append(words)
            dst_speakers.append(word_speakers)

            # Get paths for the output files
            dst_audio_file = split_audio_dir / audio_file.with_suffix(".wav").name
            dst_rttm_file = split_rttm_dir / audio_file.with_suffix(".rttm").name
            dst_transcript_file = split_transcript_dir / audio_file.with_suffix(".txt").name

            # Save audio
            between = (min(start_speech_segments), max(end_speech_segments))
            self._convert_sph_to_wav(audio_file, dst_audio_file, between)

            # Save rttm
            self._save_rttm(
                dst_rttm_file,
                speaker_speech_segments,
                start_speech_segments,
                end_speech_segments,
            )

            # Save word-speaker pairs
            self._save_word_speaker_pairs(dst_transcript_file, word_speakers, words)

            dst_audio_files.append(str(dst_audio_file))
            dst_rttm_files.append(str(dst_rttm_file))

        return SpeakerDiarizationData(
            split=split_name,
            audio_paths=dst_audio_files,
            annotation_paths=dst_rttm_files,
            uem_paths=None,
            transcript=dst_words,
            word_speakers=dst_speakers,
            word_timestamps=None,
            metadata=None,
        )

    # We don't need to download
    def download(self) -> None:
        pass

    def create_dataset(self) -> dict[str, SpeakerDiarizationData]:
        self._setup_sph2pipe()
        data_dict = {
            split_name: self._process_split(split_key, split_name)
            for split_key, split_name in self.split_name_mapping.items()
        }
        return data_dict


class Chime6Dataset(SpeakerDiarizationDataset):
    audio_eval_url = "https://openslr.trmal.net/resources/150/CHiME6_eval.tar.gz"
    annotation_url = "https://openslr.trmal.net/resources/150/CHiME6_transcriptions.tar.gz"

    @property
    def dataset_name(self) -> str:
        return "chime-6"

    def download(self) -> None:
        download_file(self.audio_eval_url, "CHiME6_eval.tar.gz")
        download_file(self.annotation_url, "CHiME6_transcriptions.tar.gz")

    def _extract_audio(self) -> None:
        with tarfile.open("CHiME6_eval.tar.gz", "r") as tar:
            tar.extractall()
        with tarfile.open("CHiME6_transcriptions.tar.gz", "r") as tar:
            tar.extractall()

    def _save_rttm(
        self, rttm_file: Path, speakers: list[str], start_times: list[float], end_times: list[float]
    ) -> str:
        annotation = Annotation(uri=rttm_file.stem)
        offset = min(start_times)
        for spk, start, end in zip(speakers, start_times, end_times):
            segment = Segment(start - offset, end - offset)
            annotation[segment] = spk
        with rttm_file.open("w") as f:
            annotation.write_rttm(f)
        return str(rttm_file)

    def _to_seconds(self, time: str) -> float:
        # Format is HH:MM:SS.SSS
        hours, minutes, seconds = time.split(":")
        return float(hours) * 3600 + float(minutes) * 60 + float(seconds)

    def create_dataset(self) -> dict[str, SpeakerDiarizationData]:
        self._extract_audio()

        annotation_dir = Path("transcriptions") / "transcriptions" / "eval"
        audio_dir = Path("CHiME6_eval") / "CHiME6" / "audio" / "eval"

        annotation_files = sorted((annotation_dir.glob("*.json")))

        all_words: list[list[str]] = []
        all_speakers: list[list[str]] = []
        all_audio_files: list[str] = []
        all_annotations: list[str] = []
        for annotation_file in annotation_files:
            annotation_data: list[dict[str, Any]] = load_json(annotation_file)

            name = annotation_file.stem
            audio_file = audio_dir / f"{name}_U02.CH1.wav"
            if not audio_file.exists():
                raise FileNotFoundError(f"Audio file {audio_file} not found")
            all_audio_files.append(str(audio_file))

            words: list[str] = []
            speakers: list[str] = []
            word_speakers: list[str] = []
            start_times: list[float] = []
            end_times: list[float] = []
            for utterance in annotation_data:
                _words = utterance["words"].split()
                speaker = utterance["speaker"]

                # For the transcript
                words.extend(_words)
                word_speakers.extend([speaker] * len(_words))

                # For the rttm
                speakers.append(speaker)
                start_times.append(self._to_seconds(utterance["start_time"]))
                end_times.append(self._to_seconds(utterance["end_time"]))

            rttm_file = audio_file.with_suffix(".rttm")
            self._save_rttm(rttm_file, speakers, start_times, end_times)
            all_annotations.append(str(rttm_file))

            all_words.append(words)
            all_speakers.append(word_speakers)

        return {
            "test": SpeakerDiarizationData(
                split="test",
                audio_paths=all_audio_files,
                annotation_paths=all_annotations,
                transcript=all_words,
                word_speakers=all_speakers,
                uem_paths=None,
                word_timestamps=None,
                metadata=None,
            )
        }


def main(dataset_name: str, generate_only: bool, hf_repo_owner: str) -> None:
    if dataset_name == "earnings21":
        ds = Earnings21Dataset(hf_repo_owner)
    elif dataset_name == "msdwild":
        ds = MSDWildDataset(hf_repo_owner)
    elif dataset_name == "icsi-meetings":
        ds = ICSIMeetingsDataset(hf_repo_owner)
    elif dataset_name == "ali-meetings":
        ds = AliMeetingsDataset(hf_repo_owner)
    elif dataset_name == "aishell-4":
        ds = AIShell4Dataset(hf_repo_owner)
    elif dataset_name == "american-life":
        ds = AmericanLifeDataset(hf_repo_owner)
    elif dataset_name == "ava-avd":
        ds = AVAAvdDataset(hf_repo_owner)
    elif dataset_name == "dihard-3":
        ds = DIHAR3DDataset(hf_repo_owner)
    elif dataset_name == "callhome":
        ds = CallHomeDataset(hf_repo_owner)
    elif dataset_name == "ego4d":
        ds = Ego4dDataset(hf_repo_owner)
    elif dataset_name == "callhome-english":
        ds = CallHomeEnglishTranscript(hf_repo_owner)
    elif dataset_name == "chime-6":
        ds = Chime6Dataset(hf_repo_owner)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    if generate_only:
        ds.generate()
    else:
        ds.push_to_hub()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        choices=[
            "earnings21",
            "msdwild",
            "icsi-meetings",
            "ali-meetings",
            "aishell-4",
            "american-life",
            "ava-avd",
            "dihard-3",
            "callhome",
            "ego4d",
            "callhome-english",
            "chime-6",
        ],
    )
    parser.add_argument("--generate-only", action="store_true", help="Generate the dataset only,")
    parser.add_argument(
        "--hf-repo-owner",
        type=str,
        required=False,
        default="argmaxinc",
        help="The organization to push the dataset to",
    )
    args = parser.parse_args()
    main(args.dataset_name, args.generate_only, args.hf_repo_owner)
