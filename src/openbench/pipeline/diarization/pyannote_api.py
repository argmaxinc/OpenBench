# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import requests
from argmaxtools.utils import get_logger
from pyannote.core import Segment
from pydantic import BaseModel, Field, model_validator

from ...dataset import DiarizationSample
from ...pipeline_prediction import DiarizationAnnotation
from ..base import Pipeline, PipelineType, register_pipeline
from .common import DiarizationOutput, DiarizationPipelineConfig


__all__ = ["PyannoteApiPipeline", "PyannoteApiConfig"]

logger = get_logger(__name__)


class PyannoteApiSegment(BaseModel):
    speaker: str
    start: float
    end: float


class PyannoteApiDiarization(BaseModel):
    diarization: list[PyannoteApiSegment]

    def to_pyannote_annotation(self) -> DiarizationAnnotation:
        annotation = DiarizationAnnotation()
        for segment in self.diarization:
            annotation[Segment(segment.start, segment.end)] = segment.speaker
        return annotation


def to_camel(string: str) -> str:
    components = string.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


class PyannoteApiOutput(BaseModel):
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
    output: PyannoteApiDiarization = Field(
        description="The output of the job",
    )

    class Config:
        alias_generator = to_camel
        validate_by_name = True

    @model_validator(mode="before")
    @classmethod
    def parse_shape(cls, data: dict) -> dict:
        if isinstance(data["createdAt"], str):
            data["createdAt"] = datetime.fromisoformat(data["createdAt"])
        if isinstance(data["updatedAt"], str) and data["updatedAt"] is not None:
            data["updatedAt"] = datetime.fromisoformat(data["updatedAt"])
        return data

    def get_elapsed_time(self) -> float:
        if self.updated_at is not None:
            return (self.updated_at - self.created_at).total_seconds()
        return self.job_polling_elapsed_time


# Expects a env variable `PYANNOTE_TOKEN` to be set with a valid pyannote-ai token
class PyannoteApi:
    diarization_url = "https://api.pyannote.ai/v1/diarize"
    media_url = "https://api.pyannote.ai/v1/media/input"
    jobs_url = "https://api.pyannote.ai/v1/jobs"

    def __init__(
        self,
        timeout: int = 1800,
        request_buffer: int = 30,
    ) -> None:
        self.timeout = timeout
        self.request_buffer = request_buffer

    def get_presigned_url(self, audio_path: str) -> str:
        # We need to push the audio file to temporary storage from pyannote-ai
        # we could also push it to S3 or other storage, but this is the easiest way
        # to avoid setting up a storage bucket
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

    def diarize(self, audio_url: str, num_speakers: int | None = None) -> str:
        # We could also pass a Webhook to get the result, but we can poll the job status
        # and get the response. This is easier to implement although polling can hit
        # rate limits.
        data = {"url": audio_url}

        if num_speakers is not None:
            data["numSpeakers"] = num_speakers

        response = requests.post(
            self.diarization_url,
            headers={"Authorization": f"Bearer {os.environ['PYANNOTE_TOKEN']}"},
            json=data,
        )
        response.raise_for_status()
        return response

    def get_job_results(self, diarization_response: requests.Response) -> PyannoteApiOutput:
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
                    return PyannoteApiOutput.model_validate(job_data)
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

    def __call__(self, audio_path: str, num_speakers: int | None = None) -> PyannoteApiOutput:
        audio_url = self.get_presigned_url(audio_path)
        diarization_response = self.diarize(audio_url, num_speakers)
        return self.get_job_results(diarization_response)


class PyannoteApiConfig(DiarizationPipelineConfig):
    timeout: int = Field(
        default=1800,
        description="Timeout for the diarization job in seconds",
    )
    request_buffer: int = Field(
        default=30,
        description="Buffer for the request rate limit",
    )


TEMP_AUDIO_DIR = Path("audio_temp")


@register_pipeline
class PyannoteApiPipeline(Pipeline):
    _config_class = PyannoteApiConfig
    pipeline_type = PipelineType.DIARIZATION

    def build_pipeline(
        self,
    ) -> Callable[[dict[str, str | int | None]], PyannoteApiOutput]:
        api = PyannoteApi(
            timeout=self.config.timeout,
            request_buffer=self.config.request_buffer,
        )
        return lambda input_sample: api(
            audio_path=input_sample["audio_path"],
            num_speakers=input_sample.get("num_speakers"),
        )

    def parse_input(self, input_sample: DiarizationSample) -> dict[str, str | int | None]:
        audio_path = input_sample.save_audio(TEMP_AUDIO_DIR)
        # setting as attribute to remove after parsing output
        self._audio_path = audio_path
        return {"audio_path": str(audio_path)}

    def parse_output(self, output: PyannoteApiOutput) -> DiarizationOutput:
        output = DiarizationOutput(
            prediction=output.output.to_pyannote_annotation(),
        )
        # remove audio from temp
        self._audio_path.unlink()
        return output
