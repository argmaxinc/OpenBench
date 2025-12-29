from pathlib import Path
from typing import Callable, Literal

from argmaxtools.utils import get_logger
from openai.types.audio import TranscriptionDiarized
from pydantic import Field

from ...dataset import OrchestrationSample
from ...engine import OpenAIApi
from ...pipeline import Pipeline, register_pipeline
from ...pipeline_prediction import Transcript, Word
from ...types import PipelineType
from .common import OrchestrationConfig, OrchestrationOutput


logger = get_logger(__name__)

TEMP_AUDIO_DIR = Path("temp_audio_dir")


class OpenAIOrchestrationPipelineConfig(OrchestrationConfig):
    model_version: Literal["gpt-4o-transcribe-diarize"] = Field(
        default="gpt-4o-transcribe-diarize",
        description="The version of the OpenAI model to use. Currently only `gpt-4o-transcribe-diarize` is supported.",
    )


@register_pipeline
class OpenAIOrchestrationPipeline(Pipeline):
    _config_class = OpenAIOrchestrationPipelineConfig
    pipeline_type = PipelineType.ORCHESTRATION

    def build_pipeline(self) -> Callable[[Path], TranscriptionDiarized]:
        openai_api = OpenAIApi(model=self.config.model_version)

        def orchestrate(audio_path: Path) -> TranscriptionDiarized:
            response = openai_api.transcribe(audio_path, language=self.current_language)
            # Remove temporary audio path
            audio_path.unlink(missing_ok=True)
            return response

        return orchestrate

    def parse_input(self, input_sample: OrchestrationSample) -> Path:
        # Extract language if force_language is enabled
        self.current_language = None
        if self.config.force_language:
            self.current_language = input_sample.language

        return input_sample.save_audio(TEMP_AUDIO_DIR)

    def parse_output(self, output: TranscriptionDiarized) -> OrchestrationOutput:
        words: list[Word] = []
        for segment in output.segments:
            for word in segment.text.split():
                # The timestamps provided are the start and end of the segment, not the word
                # therefore we ignore them
                words.append(Word(word=word, speaker=segment.speaker))

        return OrchestrationOutput(prediction=Transcript(words=words))
