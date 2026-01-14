from pathlib import Path
from typing import Callable

from argmaxtools.utils import get_logger
from pyannote.core import Segment
from pydantic import Field

from ...dataset import DiarizationSample
from ...engine import ElevenLabsApi, ElevenLabsApiResponse
from ...pipeline_prediction import DiarizationAnnotation
from ..base import Pipeline, PipelineType, register_pipeline
from .common import DiarizationOutput, DiarizationPipelineConfig


__all__ = ["ElevenLabsDiarizationPipeline", "ElevenLabsDiarizationPipelineConfig"]

TEMP_AUDIO_DIR = Path("audio_temp")

logger = get_logger(__name__)


class ElevenLabsDiarizationPipelineConfig(DiarizationPipelineConfig):
    model_id: str = Field(
        default="scribe_v2",
        description="The ElevenLabs speech-to-text model to use",
    )
    num_speakers: int | None = Field(
        default=None,
        description="Maximum number of speakers (helps with diarization). Max 32.",
    )


@register_pipeline
class ElevenLabsDiarizationPipeline(Pipeline):
    _config_class = ElevenLabsDiarizationPipelineConfig
    pipeline_type = PipelineType.DIARIZATION

    def build_pipeline(self) -> Callable[[Path], ElevenLabsApiResponse]:
        api = ElevenLabsApi(model_id=self.config.model_id)

        num_speakers = None
        if self.config.use_exact_num_speakers:
            num_speakers = self.config.num_speakers

        def transcribe(audio_path: Path) -> ElevenLabsApiResponse:
            response = api.transcribe(
                audio_path=audio_path,
                diarize=True,
                num_speakers=num_speakers,
            )
            # Remove temporary audio path
            audio_path.unlink(missing_ok=True)
            return response

        return transcribe

    def parse_input(self, input_sample: DiarizationSample) -> Path:
        return input_sample.save_audio(output_dir=TEMP_AUDIO_DIR)

    def parse_output(self, output: ElevenLabsApiResponse) -> DiarizationOutput:
        annotation = DiarizationAnnotation()
        for word, speaker, start, end in zip(
            output.words, output.speakers, output.start, output.end
        ):
            annotation[Segment(start, end)] = f"SPEAKER_{speaker}"

        return DiarizationOutput(prediction=annotation)

