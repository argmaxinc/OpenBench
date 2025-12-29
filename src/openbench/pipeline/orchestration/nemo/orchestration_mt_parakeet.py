# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

from pathlib import Path
from typing import Callable

import torch
from argmaxtools.utils import get_fastest_device, get_logger
from nemo.collections.asr.models import ASRModel, SortformerEncLabelModel

# Use the helper class `SpeakerTaggedASR`, which handles all ASR and diarization cache data for streaming.
from nemo.collections.asr.parts.utils.multispk_transcribe_utils import SpeakerTaggedASR
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
from omegaconf import OmegaConf
from pydantic import Field

from ....dataset import OrchestrationSample
from ....pipeline_prediction import Transcript, Word
from ...base import Pipeline, PipelineType, register_pipeline
from ...diarization.nemo.sortformer_pipeline import NeMoSortformerPipelineInput
from ..common import OrchestrationConfig, OrchestrationOutput

# Use the pre-defined dataclass template `MultitalkerTranscriptionConfig` from `multitalker_transcript_config.py`.
# Configure the diarization model using streaming parameters:
from .multitalker_transcript_config import MultitalkerTranscriptionConfig


logger = get_logger(__name__)


# Constants
TEMP_AUDIO_DIR = Path("./temp_audio")

__all__ = ["NeMoMTParakeetPipeline", "NeMoMTParakeetPipelineConfig"]


class NeMoMTParakeetPipelineConfig(OrchestrationConfig):
    diar_model_id: str = Field(
        default="nvidia/diar_streaming_sortformer_4spk-v2.1",
        description="The ID of the diarization model to use.",
    )
    asr_model_id: str = Field(
        default="nvidia/multitalker-parakeet-streaming-0.6b-v1",
        description="The ID of the ASR model to use.",
    )
    device: str = Field(
        default_factory=get_fastest_device,
        description="PyTorch device where Sortformer will run its inference.",
    )


@register_pipeline
class NeMoMTParakeetPipeline(Pipeline):
    _config_class = NeMoMTParakeetPipelineConfig
    pipeline_type = PipelineType.ORCHESTRATION

    def build_pipeline(self) -> Callable:
        # A speaker diarization model is needed for tracking the speech activity of each speaker.
        diar_model = SortformerEncLabelModel.from_pretrained(
            self.config.diar_model_id, map_location=self.config.device
        ).eval()
        asr_model = ASRModel.from_pretrained(self.config.asr_model_id, map_location=self.config.device).eval()
        cfg = OmegaConf.structured(MultitalkerTranscriptionConfig())
        diar_model = MultitalkerTranscriptionConfig.init_diar_model(cfg, diar_model)

        def inference(sample: NeMoSortformerPipelineInput) -> list[dict[str, str]]:
            cfg.audio_file = sample.audio_path
            samples = [{"audio_filepath": sample.audio_path}]
            streaming_buffer = CacheAwareStreamingAudioBuffer(
                model=asr_model,
                online_normalization=cfg.online_normalization,
                pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,
            )
            streaming_buffer.append_audio_file(audio_filepath=sample.audio_path, stream_id=-1)
            streaming_buffer_iter = iter(streaming_buffer)
            multispk_asr_streamer = SpeakerTaggedASR(cfg, asr_model, diar_model)

            for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer_iter):
                drop_extra_pre_encoded = (
                    0
                    if step_num == 0 and not cfg.pad_and_drop_preencoded
                    else asr_model.encoder.streaming_cfg.drop_extra_pre_encoded
                )
                with torch.inference_mode():
                    multispk_asr_streamer.perform_parallel_streaming_stt_spk(
                        step_num=step_num,
                        chunk_audio=chunk_audio,
                        chunk_lengths=chunk_lengths,
                        is_buffer_empty=streaming_buffer.is_buffer_empty(),
                        drop_extra_pre_encoded=drop_extra_pre_encoded,
                    )

            # Generate the speaker-tagged transcript and print it.
            multispk_asr_streamer.generate_seglst_dicts_from_parallel_streaming(samples=samples)
            # Delete temp audio file
            sample.audio_path.unlink()
            return multispk_asr_streamer.instance_manager.seglst_dict_list

        return inference

    def parse_input(self, input_sample: OrchestrationSample) -> OrchestrationSample:
        assert input_sample.sample_rate == 16000, "Sample rate must be 16kHz"

        # Warn if force_language is enabled (not currently supported)
        if self.config.force_language:
            logger.warning(
                f"{self.__class__.__name__} does not support language hinting. "
                "The force_language flag will be ignored."
            )

        parsed_input = NeMoSortformerPipelineInput(
            audio_path=input_sample.save_audio(TEMP_AUDIO_DIR),
            keep_audio=False,
        )
        return parsed_input

    def parse_output(self, output: list[dict[str, str]]) -> OrchestrationOutput:
        words = []
        for speaker_output in output:
            words.extend(
                [Word(word=word, speaker=speaker_output["speaker"]) for word in speaker_output["words"].split()]
            )
        prediction = Transcript.from_words_info(
            words=[word.word for word in words], speaker=[word.speaker for word in words]
        )
        return OrchestrationOutput(
            prediction=prediction,
        )
