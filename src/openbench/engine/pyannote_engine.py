# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""PyannoteAI API engine for diarization and transcription."""

import os
import time
from datetime import datetime
from pathlib import Path

import requests
from argmaxtools.utils import get_logger
from pyannote.core import Segment
from pydantic import BaseModel, Field, model_validator

from ..pipeline_prediction import DiarizationAnnotation


__all__ = [
    "PyannoteAIApi",
    "PyannoteApiDiarizationOutput",
    "PyannoteApiOrchestrationOutput",
    "PyannoteApiSegment",
    "PyannoteApiWord",
    "PyannoteApiTurn",
]

logger = get_logger(__name__)


def to_camel(string: str) -> str:
    """Convert snake_case to camelCase."""
    components = string.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


# Response models for diarization segments
class PyannoteApiSegment(BaseModel):
    """A single diarization segment from PyannoteAI API."""

    speaker: str
    start: float
    end: float


class PyannoteApiDiarization(BaseModel):
    """Diarization output containing a list of segments."""

    diarization: list[PyannoteApiSegment]

    def to_pyannote_annotation(self) -> DiarizationAnnotation:
        """Convert to pyannote DiarizationAnnotation format."""
        annotation = DiarizationAnnotation()
        for segment in self.diarization:
            annotation[Segment(segment.start, segment.end)] = segment.speaker
        return annotation


# Response models for transcription (word-level and turn-level)
class PyannoteApiWord(BaseModel):
    """A single word from PyannoteAI transcription output."""

    start: float
    end: float
    text: str
    speaker: str

    class Config:
        populate_by_name = True


class PyannoteApiTurn(BaseModel):
    """A speaker turn from PyannoteAI transcription output."""

    start: float
    end: float
    text: str
    speaker: str

    class Config:
        populate_by_name = True


class PyannoteApiOrchestrationData(BaseModel):
    """Output data containing diarization and transcription results."""

    diarization: list[PyannoteApiSegment]
    word_level_transcription: list[PyannoteApiWord] = Field(alias="wordLevelTranscription")
    turn_level_transcription: list[PyannoteApiTurn] = Field(alias="turnLevelTranscription")

    class Config:
        populate_by_name = True

    def to_pyannote_annotation(self) -> DiarizationAnnotation:
        """Convert diarization to pyannote DiarizationAnnotation format."""
        annotation = DiarizationAnnotation()
        for segment in self.diarization:
            annotation[Segment(segment.start, segment.end)] = segment.speaker
        return annotation


# Base output class with common fields
class PyannoteApiBaseOutput(BaseModel):
    """Base output model with common job metadata."""

    job_id: str = Field(
        description="The id of the job that was submitted to pyannote-ai",
    )
    status: str = Field(
        description="The status of the job that was submitted to pyannote-ai",
    )
    created_at: datetime = Field(
        description="The time the job was created",
    )
    updated_at: datetime | None = Field(
        description="The time the job was updated. For some reason it can be None.",
    )
    job_polling_elapsed_time: float = Field(
        description="The time it took to poll the job results",
    )

    class Config:
        alias_generator = to_camel
        populate_by_name = True

    @model_validator(mode="before")
    @classmethod
    def parse_shape(cls, data: dict) -> dict:
        # Handle 'Z' suffix (Zulu/UTC timezone) which fromisoformat doesn't support directly
        if isinstance(data.get("createdAt"), str):
            created_at = data["createdAt"].replace("Z", "+00:00")
            data["createdAt"] = datetime.fromisoformat(created_at)
        if isinstance(data.get("updatedAt"), str) and data["updatedAt"] is not None:
            updated_at = data["updatedAt"].replace("Z", "+00:00")
            data["updatedAt"] = datetime.fromisoformat(updated_at)
        return data

    def get_elapsed_time(self) -> float:
        """Get the elapsed time for the job."""
        if self.updated_at is not None:
            return (self.updated_at - self.created_at).total_seconds()
        return self.job_polling_elapsed_time


class PyannoteApiDiarizationOutput(PyannoteApiBaseOutput):
    """Output model for diarization-only jobs."""

    output: PyannoteApiDiarization = Field(
        description="The diarization output of the job",
    )


class PyannoteApiOrchestrationOutput(PyannoteApiBaseOutput):
    """Output model for jobs with transcription enabled."""

    output: PyannoteApiOrchestrationData = Field(
        description="The diarization and transcription output of the job",
    )


class PyannoteAIApi:
    """
    PyannoteAI API client for diarization and transcription.

    Expects the environment variable `PYANNOTE_TOKEN` to be set with a valid pyannote-ai token.

    Args:
        timeout: Timeout for job polling in seconds
        request_buffer: Buffer for request rate limiting
        transcription: Whether to enable transcription (STT) in addition to diarization
    """

    diarization_url = "https://api.pyannote.ai/v1/diarize"
    media_url = "https://api.pyannote.ai/v1/media/input"
    jobs_url = "https://api.pyannote.ai/v1/jobs"

    def __init__(
        self,
        timeout: int = 1800,
        request_buffer: int = 30,
        transcription: bool = False,
    ) -> None:
        self.timeout = timeout
        self.request_buffer = request_buffer
        self.transcription = transcription

        # Check that the API key is set
        if not os.getenv("PYANNOTE_TOKEN"):
            raise ValueError("`PYANNOTE_TOKEN` environment variable is not set")

    def get_presigned_url(self, audio_path: str) -> str:
        """
        Get a presigned URL for uploading audio to PyannoteAI temporary storage.

        Args:
            audio_path: Path to the local audio file

        Returns:
            The media URL for the uploaded audio
        """
        logger.debug(f"Getting presigned url for {audio_path}")
        name = Path(audio_path).with_suffix(".wav").name
        # For some reason if the name has underscores, it will fail
        name = "".join([n.capitalize() for n in name.split("_")])
        # Pushing audio file to temporary storage from pyannote-ai
        audio_url = f"media://example/{name}"
        logger.debug(f"Audio url: {audio_url}")
        body = {"url": audio_url}
        # Post request to get the presigned url associated with the `audio_url`
        response = requests.post(
            url=self.media_url,
            headers={"Authorization": f"Bearer {os.environ['PYANNOTE_TOKEN']}"},
            json=body,
        )
        response.raise_for_status()
        data = response.json()
        presigned_url = data["url"]
        logger.debug(f"Presigned url: {presigned_url}")

        # Upload the audio file to the presigned url
        # Audio should be < 24hrs and < 1GB
        with open(audio_path, "rb") as audio_file:
            requests.put(
                url=presigned_url,
                data=audio_file,
            )
        logger.debug(f"Audio file uploaded to {presigned_url}")
        return audio_url

    def diarize(
        self,
        audio_url: str,
        num_speakers: int | None = None,
        transcription: bool | None = None,
    ) -> requests.Response:
        """
        Submit a diarization job to PyannoteAI.

        Args:
            audio_url: The media URL of the uploaded audio
            num_speakers: Optional number of speakers hint
            transcription: Whether to enable transcription (overrides instance setting)

        Returns:
            The response from the diarization endpoint
        """
        data = {"url": audio_url}

        if num_speakers is not None:
            data["numSpeakers"] = num_speakers

        # Use instance transcription setting if not overridden
        enable_transcription = transcription if transcription is not None else self.transcription
        if enable_transcription:
            data["transcription"] = True

        response = requests.post(
            self.diarization_url,
            headers={"Authorization": f"Bearer {os.environ['PYANNOTE_TOKEN']}"},
            json=data,
        )
        response.raise_for_status()
        return response

    def get_job_results(
        self,
        diarization_response: requests.Response,
        transcription: bool | None = None,
    ) -> PyannoteApiDiarizationOutput | PyannoteApiOrchestrationOutput:
        """
        Poll for job results until completion.

        Args:
            diarization_response: The response from the diarization endpoint
            transcription: Whether transcription was enabled (determines output type)

        Returns:
            Either PyannoteApiDiarizationOutput or PyannoteApiOrchestrationOutput
        """
        data = diarization_response.json()
        headers = diarization_response.headers
        job_id = data["jobId"]
        logger.debug(f"Starting to poll results for job {job_id}")

        # Get rate limit info from headers with fallback defaults
        remaining_requests = int(headers.get("X-RateLimit-Remaining", 30))
        rate_limit = int(headers.get("X-RateLimit-Limit", 30))
        reset_time = int(headers.get("X-RateLimit-Reset", 0))

        logger.debug(
            f"Initial rate limits - Remaining: {remaining_requests}, Limit: {rate_limit}, Reset: {reset_time}s"
        )

        # Use instance transcription setting if not overridden
        enable_transcription = transcription if transcription is not None else self.transcription

        start_time = time.time()
        elapsed_time = 0
        while elapsed_time < self.timeout:
            try:
                # Check if we need to wait for rate limit reset
                if remaining_requests <= self.request_buffer:
                    logger.debug(
                        f"Running low on requests ({remaining_requests} remaining). Waiting {reset_time}s for reset"
                    )
                    time.sleep(reset_time)
                    remaining_requests = rate_limit

                logger.debug(f"Polling job {job_id}")
                response = requests.get(
                    url=f"{self.jobs_url}/{job_id}",
                    headers={"Authorization": f"Bearer {os.environ['PYANNOTE_TOKEN']}"},
                )
                response.raise_for_status()

                # Update rate limit information
                remaining_requests = int(response.headers.get("X-RateLimit-Remaining", remaining_requests))
                reset_time = int(response.headers.get("X-RateLimit-Reset", reset_time))
                # Add a small buffer to avoid hitting rate limits
                safe_remaining = max(1, remaining_requests - self.request_buffer)
                delay = reset_time / safe_remaining
                logger.debug(
                    f"Rate limit info - Remaining: {remaining_requests}, Reset: {reset_time}s, Delay: {delay * 1000:.0f}ms"
                )

                job_data = response.json()
                job_status = job_data["status"]
                logger.debug(f"Job {job_id} status: {job_status}")

                if job_status == "succeeded":
                    elapsed_time = time.time() - start_time
                    logger.debug(f"Job {job_id} completed successfully after {elapsed_time:.1f}s")
                    job_data["jobPollingElapsedTime"] = elapsed_time

                    # Return appropriate output type based on transcription flag
                    if enable_transcription:
                        return PyannoteApiOrchestrationOutput.model_validate(job_data)
                    else:
                        return PyannoteApiDiarizationOutput.model_validate(job_data)

                elif job_status == "failed":
                    error_msg = job_data.get("error", "No error message provided")
                    logger.error(f"Job {job_id} failed: {error_msg}")
                    raise Exception(f"Job failed with error: {error_msg}")
                elif job_status == "canceled":
                    logger.error(f"Job {job_id} was canceled")
                    raise Exception("Job was canceled")

                elapsed_time = time.time() - start_time
                logger.debug(f"Waiting {delay * 1000:.0f}ms before next request")
                time.sleep(delay)

            except requests.exceptions.RequestException as e:
                logger.error(f"API request failed for job {job_id}: {str(e)}")
                raise RuntimeError(f"API request failed: {str(e)}")

        logger.error(f"Job {job_id} timed out after {elapsed_time:.1f}s")
        raise TimeoutError(f"Job timed out after {elapsed_time:.1f} seconds")

    def __call__(
        self,
        audio_path: str,
        num_speakers: int | None = None,
        transcription: bool | None = None,
    ) -> PyannoteApiDiarizationOutput | PyannoteApiOrchestrationOutput:
        """
        Process an audio file with diarization and optional transcription.

        Args:
            audio_path: Path to the local audio file
            num_speakers: Optional number of speakers hint
            transcription: Whether to enable transcription (overrides instance setting)

        Returns:
            Either PyannoteApiDiarizationOutput or PyannoteApiOrchestrationOutput
        """
        # Use instance transcription setting if not overridden
        enable_transcription = transcription if transcription is not None else self.transcription

        audio_url = self.get_presigned_url(audio_path)
        diarization_response = self.diarize(audio_url, num_speakers, enable_transcription)
        return self.get_job_results(diarization_response, enable_transcription)
