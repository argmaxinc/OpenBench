# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

import json
import os
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np
import pandas as pd
import wandb
from argmaxtools.utils import get_logger

from ..metric import MetricOptions
from ..types import PredictionProtocol
from .data_models import (
    BaseSampleResult,
    DiarizationSampleResult,
    GlobalResult,
    TaskResult,
    TranscriptionSampleResult,
)


# Disable all warnings for this module
warnings.filterwarnings("ignore")


SampleResult = TypeVar("SampleResult", bound=BaseSampleResult)


class WandbLogger(ABC, Generic[SampleResult]):
    """Base class for logging benchmark results to Weights & Biases."""

    def __init__(self, output_dir: str | None = None):
        """Initialize the WandbLogger.

        Args:
            output_dir: Directory to save artifacts
        """
        self.output_dir = "." if output_dir is None else output_dir
        self.results_dir = Path(self.output_dir) / "results"

        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def custom_log(
        self,
        global_results: list[GlobalResult],
        task_results: list[TaskResult],
        sample_results: list[SampleResult],
    ) -> dict[str, Any]:
        """Custom logging logic specific to the pipeline type.

        Args:
            global_results: List of GlobalResult objects
            task_results: List of TaskResult objects
            sample_results: List of SampleResult objects

        Returns:
            Dictionary of custom metrics and artifacts to log
        """
        pass

    def get_global_metrics(
        self, global_results: list[GlobalResult], sample_results: list[SampleResult]
    ) -> dict[str, float]:
        """Get global metrics from results.

        Args:
            global_results: List of GlobalResult objects
            sample_results: List of SampleResult objects

        Returns:
            Dictionary mapping metric names to their values
        """
        self.logger.info("Getting global metrics")

        # Store a json file with all global results
        global_results_dict = {
            global_result.metric_name: global_result.global_result for global_result in global_results
        }
        # Compute global Speed Factor
        total_audio_duration = sum(sample_result.audio_duration for sample_result in sample_results)
        total_prediction_time = sum(sample_result.prediction_time for sample_result in sample_results)
        speed_factor = total_audio_duration / total_prediction_time
        global_results_dict["speed_factor"] = speed_factor

        with open(self.results_dir / "global_results.json", "w") as f:
            json.dump(global_results_dict, f)

        return {
            f"{global_result.dataset_name}/{global_result.metric_name}": global_result.global_result
            for global_result in global_results
        }

    def get_task_results_table(self, task_results: list[TaskResult]) -> dict[str, wandb.Table]:
        """Create a table of task results.

        Args:
            task_results: List of TaskResult objects

        Returns:
            Dictionary mapping dataset name to wandb.Table
        """
        self.logger.info("Getting task results table")
        dataset_name = task_results[0].dataset_name
        df = pd.DataFrame([task_result.model_dump() for task_result in task_results])

        # Flatten detailed_result column
        all_keys = set()
        for task_result in task_results:
            all_keys.update(task_result.detailed_result.keys())
        for key in all_keys:
            df[f"detailed_{key}"] = df["detailed_result"].apply(lambda x: x.get(key, None))
        df = df.drop(columns=["detailed_result"])

        # Store a version of the table locally
        df.to_csv(self.results_dir / "task_results_table.csv", index=False)

        return {f"{dataset_name}/task_results_table": wandb.Table(dataframe=df)}

    def generate_prediction_artifact(self, sample_results: list[SampleResult]) -> wandb.Artifact:
        """Create a wandb artifact containing prediction files.

        Args:
            sample_results: List of SampleResult objects

        Returns:
            wandb.Artifact containing prediction files
        """
        dataset_name = sample_results[0].dataset_name
        pipeline_name = sample_results[0].pipeline_name
        save_dir = os.path.join(self.output_dir, dataset_name, "predictions")
        os.makedirs(save_dir, exist_ok=True)

        for sample_result in sample_results:
            prediction = sample_result.prediction
            filename = sample_result.audio_name
            prediction.to_annotation_file(save_dir, filename)

        artifact = wandb.Artifact(
            f"{dataset_name}-{pipeline_name}-predictions",
            type="predictions",
            description=f"Prediction files for the {pipeline_name} pipeline on {dataset_name}",
        )
        artifact.add_dir(save_dir)
        return artifact

    def get_sample_results_table(self, sample_results: list[SampleResult]) -> dict[str, Any]:
        """Create a table of sample results.

        Args:
            sample_results: List of SampleResult objects

        Returns:
            Dictionary mapping dataset name to wandb.Table
        """
        self.logger.info("Creating sample results table")

        # Convert to list of dicts
        rows = [sample_result.model_dump() for sample_result in sample_results]
        # Remove row keys that are np.ndarray
        rows = [{k: v for k, v in row.items() if not isinstance(v, (np.ndarray, PredictionProtocol))} for row in rows]
        dataset_name = rows[0]["dataset_name"]

        # Store a version of the table locally
        df = pd.DataFrame(rows)
        df.to_csv(self.results_dir / "sample_results_table.csv", index=False)

        # Create results table
        return {
            f"{dataset_name}/sample_results_table": wandb.Table(
                data=[list(row.values()) for row in rows], columns=list(rows[0].keys())
            )
        }

    def get_latency_metrics(self, sample_results: list[SampleResult]) -> dict[str, float]:
        """Calculate latency metrics from sample results.

        Args:
            sample_results: List of SampleResult objects

        Returns:
            Dictionary mapping metric names to their values
        """
        dataset_name = sample_results[0].dataset_name
        prediction_times = np.array([sample_result.prediction_time for sample_result in sample_results])
        audio_durations = np.array([sample_result.audio_duration for sample_result in sample_results])

        total_audio_duration = np.sum(audio_durations)
        total_prediction_time = np.sum(prediction_times)
        speed_factor = total_audio_duration / total_prediction_time

        return {
            f"{dataset_name}/prediction_time_mean": np.mean(prediction_times),
            f"{dataset_name}/prediction_time_std": np.std(prediction_times),
            f"{dataset_name}/speed_factor": speed_factor,
            f"{dataset_name}/total_audio_duration": total_audio_duration,
            f"{dataset_name}/total_prediction_time": total_prediction_time,
        }

    def __call__(
        self,
        global_results: list[GlobalResult],
        task_results: list[TaskResult],
        sample_results: list[SampleResult],
    ) -> dict[str, Any]:
        """Log results to wandb.

        Args:
            global_results: List of GlobalResult objects
            task_results: List of TaskResult objects
            sample_results: List of SampleResult objects

        Returns:
            Dictionary of metrics and artifacts to log
        """
        # Check if results and output dir exists otherwise create them
        if not self.results_dir.exists():
            self.results_dir.mkdir(parents=True, exist_ok=True)
        if not Path(self.output_dir).exists():
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        log_dict = {}

        # Get global metrics
        log_dict.update(self.get_global_metrics(global_results, sample_results))

        # Get task results table
        log_dict.update(self.get_task_results_table(task_results))

        # Get sample results table
        log_dict.update(self.get_sample_results_table(sample_results))

        # Get latency metrics
        log_dict.update(self.get_latency_metrics(sample_results))

        # Add custom logging from subclass
        custom_log_dict = self.custom_log(global_results, task_results, sample_results)
        log_dict.update(custom_log_dict)

        # Generate prediction artifact
        prediction_artifact = self.generate_prediction_artifact(sample_results)

        wandb.log(log_dict)
        wandb.log_artifact(prediction_artifact)


class DiarizationWandbLogger(WandbLogger[DiarizationSampleResult]):
    """Logger for diarization pipeline results."""

    def log_embeddings_artifact(self, sample_results: list[DiarizationSampleResult]) -> None:
        dataset_name = sample_results[0].dataset_name
        pipeline_name = sample_results[0].pipeline_name

        # Create save dir for embeddings
        save_dir_embeddings = os.path.join(self.output_dir, dataset_name, "embeddings")
        os.makedirs(save_dir_embeddings, exist_ok=True)

        # Save embeddings for each sample
        for sample_result in sample_results:
            if sample_result.embeddings is not None:
                np.savez_compressed(
                    f"{save_dir_embeddings}/sample_{sample_result.sample_id}.npz",
                    embeddings=sample_result.embeddings,
                    centroids=sample_result.centroids,
                    cluster_labels=sample_result.cluster_labels,
                )

        # Create embeddings artifact
        embeddings_artifact = wandb.Artifact(
            f"{dataset_name}-{pipeline_name}-embeddings",
            type="embeddings",
            description=f"Embeddings for the diarization predictions for each sample in {dataset_name} for {pipeline_name}",
        )
        embeddings_artifact.add_dir(save_dir_embeddings)

        wandb.log_artifact(embeddings_artifact)

    def get_der_components(self, global_results: list[GlobalResult]) -> dict[str, wandb.Table]:
        """Create a table of DER components.

        Args:
            global_results: List of GlobalResult objects

        Returns:
            Dictionary mapping dataset name to wandb.Table with DER breakdown
        """
        self.logger.info("Getting DER components")
        der_global_results = [g for g in global_results if g.metric_name == MetricOptions.DER.value]
        if len(der_global_results) == 0 or len(der_global_results) > 1:
            raise ValueError("There should be only one DER global result as logging is done per dataset")

        dataset_name = der_global_results[0].dataset_name
        der_detailed_result = [g.detailed_result for g in der_global_results]
        der_components_df = (
            pd.DataFrame(der_detailed_result)
            .rename(
                columns={
                    "false alarm": "false_alarm",
                    "missed detection": "missed_detection",
                    "confusion": "confusion",
                }
            )
            .assign(
                pipeline_name=[g.pipeline_name for g in der_global_results],
                dataset_name=[g.dataset_name for g in der_global_results],
                der=[g.global_result for g in der_global_results],
                false_alarm_rate=lambda x: x["false_alarm"] / x["total"],
                missed_detection_rate=lambda x: x["missed_detection"] / x["total"],
                confusion_rate=lambda x: x["confusion"] / x["total"],
            )
        )

        first_cols = ["pipeline_name", "dataset_name", "der"]
        last_cols = [col for col in der_components_df.columns if col not in first_cols]

        return {
            f"{dataset_name}/der_breakdown_table": wandb.Table(dataframe=der_components_df[first_cols + last_cols])
        }

    def custom_log(
        self,
        global_results: list[GlobalResult],
        task_results: list[TaskResult],
        sample_results: list[DiarizationSampleResult],
    ) -> dict[str, Any]:
        """Custom logging logic for diarization pipeline.

        Args:
            global_results: List of GlobalResult objects
            task_results: List of TaskResult objects
            sample_results: List of DiarizationSampleResult objects

        Returns:
            Dictionary of custom metrics and artifacts to log
        """
        self.log_embeddings_artifact(sample_results)

        custom_log_dict = {}
        if any(g.metric_name == MetricOptions.DER.value for g in global_results):
            custom_log_dict.update(self.get_der_components(global_results))
        return custom_log_dict


class TranscriptionWandbLogger(WandbLogger[TranscriptionSampleResult]):
    """Logger for transcription pipeline results."""

    def custom_log(
        self,
        global_results: list[GlobalResult],
        task_results: list[TaskResult],
        sample_results: list[TranscriptionSampleResult],
    ) -> dict[str, Any]:
        """Custom logging logic for transcription pipeline."""
        return {}
