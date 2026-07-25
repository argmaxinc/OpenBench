# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

from multiprocessing import Pool
from typing import NamedTuple

import tqdm
import wandb
import yaml
from argmaxtools.utils import get_logger
from pyannote.metrics.base import BaseMetric

from ..dataset import BaseDataset, BaseSample, DatasetRegistry
from ..metric import MetricRegistry
from ..pipeline import Pipeline
from ..types import PipelineType
from .config import BenchmarkConfig
from .data_models import (
    BenchmarkResult,
    DiarizationSampleResult,
    GlobalResult,
    SpeechGenerationSampleResult,
    TaskResult,
    TranscriptionSampleResult,
)
from .utils import change_directory, get_global_results
from .wandb_logger import DiarizationWandbLogger, SpeechGenerationWandbLogger, TranscriptionWandbLogger


logger = get_logger(__name__)

PIPELINE_TYPE_TO_SAMPLE_RESULT = {
    PipelineType.DIARIZATION: DiarizationSampleResult,
    PipelineType.TRANSCRIPTION: TranscriptionSampleResult,
    PipelineType.ORCHESTRATION: TranscriptionSampleResult,
    PipelineType.STREAMING_TRANSCRIPTION: TranscriptionSampleResult,
    PipelineType.SPEECH_GENERATION: SpeechGenerationSampleResult,
}


class ProcessingResult(NamedTuple):
    sample_result: DiarizationSampleResult | TranscriptionSampleResult | SpeechGenerationSampleResult
    task_results: list[TaskResult]
    sample_id: int
    metrics_string: str
    # Per-sample row for the speech-generation HF results sink (None otherwise).
    result_row: dict | None = None


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig, pipelines: list[Pipeline]):
        """Runs benchmarks for diarization pipelines.

        This class handles:
        - Pipeline execution on multiple datasets
        - Metric calculation and aggregation
        - Parallel processing support
        - Wandb logging

        Args:
            config: Benchmark configuration
            pipelines: List of pipelines to benchmark
        """
        self.config = config
        self.pipelines = pipelines
        self.logger_map = {
            PipelineType.DIARIZATION: DiarizationWandbLogger,
            PipelineType.TRANSCRIPTION: TranscriptionWandbLogger,
            PipelineType.ORCHESTRATION: TranscriptionWandbLogger,
            PipelineType.STREAMING_TRANSCRIPTION: TranscriptionWandbLogger,
            PipelineType.SPEECH_GENERATION: SpeechGenerationWandbLogger,
        }

    def _maybe_create_results_sink(self, pipeline: Pipeline):
        """Create the incremental HF results sink if configured for this pipeline."""
        if not self.config.hf_results_repo or pipeline.pipeline_type != PipelineType.SPEECH_GENERATION:
            return None
        from .speech_generation_sink import SpeechGenerationResultSink

        return SpeechGenerationResultSink(
            repo_id=self.config.hf_results_repo,
            flush_every=self.config.hf_results_flush_every,
            chunk_tag=self.config.hf_results_chunk_tag,
        )

    def _get_metrics(self, pipeline: Pipeline) -> dict[str, BaseMetric]:
        metrics_dict = {}
        available_metrics = MetricRegistry.get_available_metrics(pipeline.pipeline_type)
        for metric_name, kwargs in self.config.metrics.items():
            if metric_name not in available_metrics:
                continue
            metrics_dict[metric_name] = MetricRegistry.get_metric(pipeline.pipeline_type, metric_name, **kwargs)
        return metrics_dict

    def _process_single_sample_wrapper(self, args):
        """
        Wrapper function to unpack arguments correctly
        """
        sample_and_id, pipeline, dataset_name, metrics_dict, dataset_length = args
        return self._process_single_sample(sample_and_id, pipeline, dataset_name, metrics_dict, dataset_length)

    def _process_single_sample(
        self,
        sample_and_id: tuple[int, BaseSample],
        pipeline: Pipeline,
        dataset_name: str,
        metrics_dict: dict,
        dataset_length: int,
    ) -> ProcessingResult:
        sample_id, sample = sample_and_id
        try:
            output = pipeline(sample)
        except Exception as e:
            logger.error(
                f"Failed to process sample {sample_id} ({sample.audio_name}) "
                f"from dataset {dataset_name} with pipeline {pipeline.__class__.__name__}: {e}"
            )
            raise RuntimeError(f"Pipeline execution failed on sample {sample_id} ({sample.audio_name}): {e}") from e

        # For speech-generation pipelines the sample has no real input audio
        # (only a placeholder waveform), so duration is read off the prediction
        # produced by the TTS step.
        if pipeline.pipeline_type == PipelineType.SPEECH_GENERATION:
            audio_duration = output.prediction.duration
        else:
            audio_duration = sample.get_audio_duration()
        prediction_time = output.prediction_time

        # Create sample result
        sample_result_class = PIPELINE_TYPE_TO_SAMPLE_RESULT[pipeline.pipeline_type]
        sample_results_attributes = dict(
            dataset_name=dataset_name,
            sample_id=sample_id,
            audio_name=sample.audio_name,
            pipeline_name=pipeline.__class__.__name__,
            audio_duration=audio_duration,
            **output.model_dump(),
        )

        if pipeline.pipeline_type == PipelineType.DIARIZATION:
            sample_results_attributes["num_speakers_predicted"] = output.prediction.num_speakers
            sample_results_attributes["num_speakers_reference"] = sample.reference.num_speakers

        sample_result = sample_result_class(**sample_results_attributes)

        # Process metrics
        task_results = []
        metrics_logging_string = ""

        for metric_name, metric in metrics_dict.items():
            reference = sample.reference
            kwargs = dict(sample.extra_info)

            # Use metric_keywords for metric calculation if available, falling back to dictionary
            if "metric_keywords" in kwargs:
                kwargs["dictionary"] = kwargs.pop("metric_keywords")

            # The metric returns a dictionary that is also stored in the metric object as a state to compute the global result
            # We copy to avoid any side effects that may happen while interacting with dictionary for reporting
            _metric_output = metric(hypothesis=output.prediction, reference=reference, detailed=True, **kwargs)
            metric_output = _metric_output.copy()

            detailed_result = {
                component_name: component_value
                for component_name, component_value in metric_output.items()
                if component_name != metric.name
            }
            result = metric_output[metric.name]

            task_results.append(
                TaskResult(
                    dataset_name=dataset_name,
                    sample_id=sample_id,
                    pipeline_name=pipeline.__class__.__name__,
                    metric_name=metric_name,
                    result=result,
                    detailed_result=detailed_result,
                )
            )
            formatted_result = f"{result:4g}" if result is not None else "N/A"
            global_metric = abs(metric)
            formatted_metric = f"{global_metric:4g}" if global_metric is not None else "N/A"
            formatted_string = (
                f"{metric_name} - Sample: {formatted_result}\n{metric_name} - Global: {formatted_metric}\n"
            )
            metrics_logging_string += formatted_string

        # Build the per-sample results-sink row BEFORE cleanup, while the
        # generated audio file still exists on disk.
        result_row = None
        if self.config.hf_results_repo and pipeline.pipeline_type == PipelineType.SPEECH_GENERATION:
            result_row = self._build_speech_generation_row(sample, output, task_results, sample_id)

        # Sample fully scored — let the pipeline drop any transient per-sample
        # inputs (e.g. a materialized reference clip). Never let cleanup abort.
        try:
            pipeline.cleanup_sample(output)
        except Exception as e:  # noqa: BLE001 - cleanup is best-effort
            logger.warning(f"cleanup_sample failed for sample {sample_id} ({sample.audio_name}): {e}")

        # Create logging string
        logging_string = (
            "\n=========================================================\n"
            f"Pipeline: {pipeline.__class__.__name__}\n"
            f"Dataset: {dataset_name}\n"
            f"Iteration: {sample_id + 1} of {dataset_length} ({((sample_id + 1) / dataset_length):.2%})\n"
            f"Prediction time: {prediction_time:.4g} seconds\n"
            f"Audio duration: {audio_duration:.4g} seconds\n"
            f"Speed Factor: {audio_duration / prediction_time:.4g}x\n"
            "---------------------------------------------------------\n"
            "Metrics:\n"
            f"{metrics_logging_string}"
            "---------------------------------------------------------\n"
            "=========================================================\n"
        )

        return ProcessingResult(sample_result, task_results, sample_id, logging_string, result_row)

    @staticmethod
    def _normalize_metric_name(name) -> str:
        return str(getattr(name, "value", name)).lower()

    def _build_speech_generation_row(self, sample, output, task_results, sample_id) -> dict | None:
        """Assemble one row for the speech-generation HF results sink.

        Carries three audio columns so per-sample scores can be debugged by
        listening: ``prompt_audio`` (the clone prompt, i.e. the sample
        waveform), ``sim_reference_audio`` (the clip SIM actually compared
        against — the held-out real target when the dataset ships one, else
        the prompt), and ``generated_audio``. Audio is carried as in-memory
        arrays so it embeds into the parquet shard.
        """
        import numpy as np
        import soundfile as sf

        sim = wer = None
        transcription = ""
        for t in task_results:
            name = self._normalize_metric_name(t.metric_name)
            if name == "sim":
                sim = t.result
            elif name == "wer":
                wer = t.result
                # The ASR transcription used to compute WER rides along in the
                # metric's per-sample detailed output (see speech_generation_wer).
                transcription = (t.detailed_result or {}).get("transcription") or ""

        try:
            gen_array, gen_sr = sf.read(output.prediction.audio_path, dtype="float32")
        except Exception as e:  # noqa: BLE001 - skip the row rather than abort the run
            logger.warning(f"Could not read generated audio for sample {sample_id}: {e}")
            return None

        prompt = {"array": np.asarray(sample.waveform, dtype=np.float32), "sampling_rate": int(sample.sample_rate)}

        # The clip the SIM metric compared the generation against (sim_audio /
        # ref_audio, whichever the pipeline recorded on the prediction). None if
        # the pipeline didn't record one or the file can't be read.
        sim_reference = None
        sim_reference_path = getattr(output.prediction, "reference_audio_path", None)
        if sim_reference_path:
            try:
                sim_array, sim_sr = sf.read(sim_reference_path, dtype="float32")
                sim_reference = {"array": sim_array, "sampling_rate": int(sim_sr)}
            except Exception as e:  # noqa: BLE001 - keep the row, just drop the debug audio
                logger.warning(f"Could not read SIM reference audio for sample {sample_id}: {e}")

        return {
            # Prefer the dataset's stable id (e.g. source file name) over the loop index.
            "sample_idx": str(sample.extra_info.get("sample_idx", sample_id)),
            "language": sample.extra_info.get("language") or "",
            "prompt_audio": prompt,
            "sim_reference_audio": sim_reference,
            "generated_audio": {"array": gen_array, "sampling_rate": int(gen_sr)},
            # prompt_text / transcription / WER / SIM kept adjacent for analysis.
            "prompt_text": sample.text,
            "transcription": transcription,
            "WER": wer,
            "SIM": sim,
        }

    def _run_pipeline_on_dataset_parallel(
        self,
        pipeline: Pipeline,
        dataset: BaseDataset,
        dataset_name: str,
    ) -> tuple[
        list[DiarizationSampleResult | TranscriptionSampleResult],
        list[TaskResult],
        list[GlobalResult],
    ]:
        """
        Parallel version of _run_pipeline_on_dataset using multiprocessing.

        Args:
            pipeline: Pipeline to evaluate
            dataset: Dataset to evaluate on
            dataset_name: Name of the dataset
        Returns:
            Tuple of (sample_results, task_results, global_results)

        NOTE: imap (similarly to map but uses lazy evaluation) will chop the iterable into
            chunks based on the (per_worker_chunk_size parameter) and submit them to the worker
            processes as separate tasks.
            A rule of thumb while setting this parameter is to base it on the number of
            worker processes and the number of samples in the dataset.
            If you spawn many worker processes, it's best you keep the chunk size small (you can keep it 1).
            On the contrary, if you spawn fewer worker processes, each one can process larger chunks of data
            without risking the overhead of inter-process communication.

        Ref: https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool.imap
        """
        metrics_dict = self._get_metrics(pipeline)
        dataset_length = len(dataset)

        args_list = [
            ((i, sample), pipeline, dataset_name, metrics_dict, dataset_length) for i, sample in enumerate(dataset)
        ]

        # NOTE: Currently, pipelines that utilize the MPS backend are not supported in parallel mode.
        # This is due to the limitation of sharing tensors across processes.
        # As workaround would be to move tensors to the CPU before processing in separate processes,
        # but this would defeat the purpose of using the MPS backend and would be slower.
        # Ref: https://github.com/pytorch/pytorch/issues/87688
        with Pool(processes=pipeline.config.num_worker_processes) as pool:
            try:
                results = list(
                    tqdm.tqdm(
                        pool.imap(
                            self._process_single_sample_wrapper,
                            args_list,
                            chunksize=pipeline.config.per_worker_chunk_size,
                        ),
                        total=dataset_length,
                        desc=f"Processing {dataset_name}",
                    )
                )
            except Exception as e:
                logger.error(f"Evaluation failed on dataset {dataset_name}: {e}")
                pool.terminate()
                raise

        # Sort results by sample_id to maintain order
        results.sort(key=lambda x: x.sample_id)

        # Separate results
        per_sample_results = [r.sample_result for r in results]
        per_task_results = [task for r in results for task in r.task_results]

        # Log results
        for result in results:
            logger.info(result.metrics_string)

        # Now update the metrics in the main process
        for metric in metrics_dict.values():
            # Clear any existing state
            metric.reset()
            # Update metric with all results
            for sample_result in per_sample_results:
                sample = dataset[sample_result.sample_id]
                kwargs = {}
                if hasattr(sample, "extra_info"):
                    if "uem" in sample.extra_info:
                        kwargs["uem"] = sample.extra_info["uem"]
                    # Use metric_keywords for metric calculation if available, falling back to dictionary
                    if "metric_keywords" in sample.extra_info:
                        kwargs["dictionary"] = sample.extra_info["metric_keywords"]
                    elif "dictionary" in sample.extra_info:
                        kwargs["dictionary"] = sample.extra_info["dictionary"]

                metric(hypothesis=sample_result.prediction, reference=sample.reference, detailed=True, **kwargs)

        # Calculate global results after updating metrics
        global_results = get_global_results(
            metrics_dict=metrics_dict,
            dataset_name=dataset_name,
            pipeline_name=pipeline.__class__.__name__,
        )

        return per_sample_results, per_task_results, global_results

    def _run_pipeline_on_dataset(
        self,
        pipeline: Pipeline,
        dataset: BaseDataset,
        dataset_name: str,
        sink=None,
    ) -> tuple[
        list[DiarizationSampleResult | TranscriptionSampleResult],
        list[TaskResult],
        list[GlobalResult],
    ]:
        per_sample_results: list[DiarizationSampleResult | TranscriptionSampleResult] = []
        per_task_results: list[TaskResult] = []

        metrics_dict = self._get_metrics(pipeline)
        dataset_length = len(dataset)
        failed_samples: list[int] = []

        for sample_id, sample in enumerate(dataset):
            try:
                processing_result = self._process_single_sample(
                    sample_and_id=(sample_id, sample),
                    pipeline=pipeline,
                    dataset_name=dataset_name,
                    metrics_dict=metrics_dict,
                    dataset_length=dataset_length,
                )
            except Exception as e:  # noqa: BLE001 - keep the benchmark going past a bad sample
                if not self.config.continue_on_sample_error:
                    raise
                failed_samples.append(sample_id)
                logger.error(
                    f"Skipping sample {sample_id} ({getattr(sample, 'audio_name', '?')}) after failure: {e}"
                )
                continue

            per_sample_results.append(processing_result.sample_result)
            per_task_results.extend(processing_result.task_results)

            # Incrementally push per-sample results (flushes every N internally).
            if sink is not None and processing_result.result_row is not None:
                sink.add(processing_result.result_row)

            logger.info(processing_result.metrics_string)

        if failed_samples:
            logger.warning(
                f"{len(failed_samples)}/{dataset_length} samples failed and were skipped on "
                f"{dataset_name}: {failed_samples}"
            )

        global_results = get_global_results(
            metrics_dict=metrics_dict,
            dataset_name=dataset_name,
            pipeline_name=pipeline.__class__.__name__,
        )
        return per_sample_results, per_task_results, global_results

    def run(self) -> BenchmarkResult:
        """
        Run the benchmark on the given datasets.
        """
        per_sample_results: list[DiarizationSampleResult | TranscriptionSampleResult] = []
        per_task_results: list[TaskResult] = []
        per_dataset_global_results: list[GlobalResult] = []

        # Getting config to log to wandb
        wandb_config = self.config.get_wandb_config_to_log()

        for pipeline in self.pipelines:
            # Get logger
            wandb_logger = self.logger_map[pipeline.pipeline_type](output_dir=".")
            # Add pipeline info to wandb config
            wandb_config["pipeline_name"] = pipeline.__class__.__name__
            wandb_config["pipeline_config"] = pipeline.config.model_dump()

            with (
                change_directory(pipeline.config.out_dir),
                wandb.init(
                    project=self.config.wandb_config.project_name,
                    name=self.config.wandb_config.run_name,
                    mode=self.config.wandb_config.wandb_mode,
                    tags=(
                        self.config.wandb_config.tags + [pipeline.__class__.__name__]
                        if self.config.wandb_config.tags
                        else [pipeline.__class__.__name__]
                    ),
                    config=wandb_config,
                ),
            ):
                # Save a copy of the wandb_config locally as a .yaml
                with open("config.yaml", "w") as f:
                    yaml.dump(wandb_config, f)

                for dataset_name, dataset_config in self.config.datasets.items():
                    # Use DatasetRegistry to get the appropriate dataset for the pipeline type
                    ds = DatasetRegistry.get_dataset_for_pipeline(
                        pipeline_type=pipeline.pipeline_type, config=dataset_config
                    )

                    logger.info(f"Evaluating {pipeline.__class__.__name__} on {dataset_name}...")

                    # Optional incremental HF results sink (speech generation only).
                    sink = self._maybe_create_results_sink(pipeline)

                    if pipeline.config.num_worker_processes:
                        if sink is not None:
                            logger.warning(
                                "hf_results_repo is set but the pipeline runs in parallel mode; "
                                "the incremental results sink is only supported in sequential mode and will be skipped."
                            )
                        logger.info(f"Executing in parallel mode with {pipeline.config.num_worker_processes} workers")
                        results = self._run_pipeline_on_dataset_parallel(
                            pipeline,
                            ds,
                            dataset_name,
                        )
                    else:
                        logger.info("Executing in sequential mode")
                        try:
                            results = self._run_pipeline_on_dataset(pipeline, ds, dataset_name, sink=sink)
                        finally:
                            if sink is not None:
                                sink.close()

                    sample_results, task_results, global_results = results
                    wandb_logger(global_results, task_results, sample_results)

                    per_sample_results.extend(sample_results)
                    per_task_results.extend(task_results)
                    per_dataset_global_results.extend(global_results)

        return BenchmarkResult(
            sample_results=per_sample_results,
            task_results=per_task_results,
            global_results=per_dataset_global_results,
        )
