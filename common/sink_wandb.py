# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import TypedDict

import pandas as pd
import wandb
from argmaxtools.utils import get_logger
from huggingface_hub import HfApi
from tqdm import tqdm


logger = get_logger(__name__)


MAIN_WORKING_DIR = Path("wandb_data")
DATA_DIR = Path("data")
RUNS_DIR = DATA_DIR / "runs"


class RunDownloadResult(TypedDict):
    metadata_path: Path
    config_path: Path
    system_metrics_path: Path
    sample_results_table_path: Path
    task_results_tables_path: Path
    predictions_dir: Path


# Check if the destination path exists and if it does return else move the file
def check_if_exists_then_move(src: Path, dst: Path) -> None:
    is_file = src.is_file()

    # If it is a file then we need to check if the destination file exists
    if is_file and dst.exists():
        return

    # If it is a directory then we need to append the source dir name to the destination
    # when moving a directory the dst is just the parent-directory where the src directory is being moved to
    if not is_file and (dst / src.name).exists():
        return

    shutil.move(src, dst)


# A class that downloads the data from runs that were created with openbench and logged in W&B
# It will download the metadata, the system metrics, the sample results table, the task results tables, and the predictions
# And store the raw data (i.e. without post-processing) in a local directory with the following structure:
# data/
#   runs/
#     <run_name>/
#       metadata.json
#       config.json
#       system_metrics.csv
#       sample_results_table.parquet
#       task_results_table.parquet
#       predictions/
#         <file-name>.{rttm,csv,...}
class WandbDownloader:
    def __init__(self, project: str, entity: str) -> None:
        self.project = project
        self.entity = entity
        self.api = wandb.Api()

    def get_run(self, run_name: str) -> wandb.apis.public.runs.Run:
        logger.info(f"Getting run {run_name}")
        runs = self.api.runs(
            f"{self.entity}/{self.project}",
            filters={"displayName": run_name},
        )
        if len(runs) == 0:
            raise ValueError(f"No run found with name {run_name}")
        if len(runs) > 1:
            raise ValueError(f"Multiple runs found with name {run_name} - {runs}")
        return runs[0]

    # Download predictions that are stored in wandb prediction artifacts
    def download_predictions(self, artifacts: list[wandb.sdk.artifacts.artifact.Artifact]) -> Path:
        artifact = [a for a in artifacts if a.type == "predictions"][0]
        artifact_dir = artifact.download()
        return Path(artifact_dir)

    def download_table(self, run_files: list[wandb.apis.public.files.File], table_name: str, run_name: str) -> Path:
        # Get the file matching the filename
        f = [f for f in run_files if table_name in f.name]
        if len(f) == 0:
            raise ValueError(f"File {table_name} not found in run {run_name}")
        if len(f) > 1:
            raise ValueError(f"Multiple files found with name {table_name} in run {run_name} - {f}")
        f = f[0]
        # Downloads a wandb table file
        path = f.download(replace=True)
        path.seek(0)
        content = path.read()
        json_content = json.loads(content)
        columns = json_content["columns"]
        data = json_content["data"]
        df = pd.DataFrame(data, columns=columns)
        parquet_path = Path(f"{table_name}.parquet")
        df.to_parquet(parquet_path, index=False)
        return parquet_path

    def download_run(self, run_name: str) -> RunDownloadResult:
        run = self.get_run(run_name)

        logger.info(f"Downloading run {run.name} - {run.id}")
        # Get metadata
        keys_to_keep = ["apple", "git", "codePath", "host", "os", "python", "startedAt"]
        metadata = {k: v for k, v in run.metadata.items() if k in keys_to_keep}
        metadata["run_name"] = run.name
        metadata["run_id"] = run.id
        # Save metadata to a json file
        logger.info(f"Saving metadata for {run.name} - {run.id}")
        metadata_path = Path("metadata.json")
        with metadata_path.open("w") as f:
            json.dump(metadata, f)

        # Save run config to a json file
        logger.info(f"Saving config for {run.name} - {run.id}")
        config_path = Path("config.json")
        with config_path.open("w") as f:
            json.dump(run.config, f)

        # Get system metrics
        system_metrics = run.history(stream="system")
        logger.info(f"Saving system metrics for {run.name} - {run.id}")
        system_metrics_path = Path("system_metrics.csv")
        system_metrics.to_csv(system_metrics_path, index=False)

        # Get run files
        run_files = run.files()
        # Get Sample Result Table
        logger.info(f"Downloading sample results table for {run.name} - {run.id}")
        sample_results_table_path = self.download_table(run_files, "sample_results_table", run.name)
        # Get Task Results Tables
        logger.info(f"Downloading task results tables for {run.name} - {run.id}")
        task_results_tables_path = self.download_table(run_files, "task_results_table", run.name)

        # Get Predictions Annotation Files
        artifacts = run.logged_artifacts()
        logger.info(f"Downloading predictions for {run.name} - {run.id}")
        predictions_dir = self.download_predictions(artifacts)

        return RunDownloadResult(
            metadata_path=metadata_path,
            config_path=config_path,
            system_metrics_path=system_metrics_path,
            sample_results_table_path=sample_results_table_path,
            task_results_tables_path=task_results_tables_path,
            predictions_dir=predictions_dir,
        )

    def structure_run_data(self, run_download_result: RunDownloadResult) -> None:
        # Read metadata.json to get the run_name and run_id
        with run_download_result["metadata_path"].open("r") as f:
            metadata = json.load(f)
        run_name = metadata["run_name"]

        logger.info(f"Structuring run data for {run_name}")
        run_dir = RUNS_DIR / f"{run_name}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Moving files to the appropriate directory
        logger.info(f"Moving metadata for {run_name}")
        check_if_exists_then_move(run_download_result["metadata_path"], run_dir / "metadata.json")

        logger.info(f"Moving config for {run_name}")
        check_if_exists_then_move(run_download_result["config_path"], run_dir / "config.json")

        logger.info(f"Moving system metrics for {run_name}")
        check_if_exists_then_move(run_download_result["system_metrics_path"], run_dir / "system_metrics.csv")

        logger.info(f"Moving sample results table for {run_name}")
        check_if_exists_then_move(
            run_download_result["sample_results_table_path"], run_dir / "sample_results_table.parquet"
        )

        logger.info(f"Moving task results tables for {run_name}")
        check_if_exists_then_move(
            run_download_result["task_results_tables_path"], run_dir / "task_results_table.parquet"
        )

        logger.info(f"Moving predictions for {run_name}")
        check_if_exists_then_move(run_download_result["predictions_dir"].rename("predictions"), run_dir)

        logger.info(f"Structured run data for {run_name}")

    def download(self, run_name: str) -> None:
        # Create the runs directory if it doesn't exist
        # it will also create the data directory if it doesn't exist
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        run_download_result = self.download_run(run_name)
        self.structure_run_data(run_download_result)


class DataProcessor:
    def __init__(self) -> None:
        pass

    # this function will split the raw metrics table into a table for each metric
    def split_metrics_table(
        self, metrics_table: pd.DataFrame, columns_to_drop: list[str] | None = None
    ) -> dict[str, pd.DataFrame]:
        metrics_table = metrics_table.copy()  # Copying just for good measure
        columns_to_drop = columns_to_drop or ["metric_name", "result"]
        # Preprocess the task results table and returns a dictionary with a dataframe for each metric
        unique_metrics = metrics_table["metric_name"].unique()
        dataframes = {}
        logger.info(f"Processing {len(unique_metrics)} unique metrics")

        for metric in tqdm(unique_metrics, desc="Processing metrics"):
            # Get only the rows for the current metric
            df_subset = (
                metrics_table.query(f"metric_name == '{metric}'")  # get current metric rows
                .dropna(how="all", axis=1)  # drop columns that are unrelated to the current metric
                .reset_index(drop=True)  # reset index to new index based on the current metric rows
                .pipe(
                    lambda df: df.rename(columns={k: k.replace("detailed_", "") for k in df.columns})
                )  # rename columns named `detailed_<component>` to just `<component>`
                .pipe(lambda df: df.assign(**{metric: df["result"]}))  # add the metric as a column
                .drop(columns=columns_to_drop)  # drop the result column
            )
            dataframes[metric] = df_subset

        return dataframes

    def process(self, run_name: str, runs_dir: Path = RUNS_DIR) -> None:
        run_dir = runs_dir / f"{run_name}"
        metrics_dir = run_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        if not run_dir.exists():
            raise ValueError(f"Run directory {run_dir} does not exist")

        # Load task_results_table.parquet
        task_results_table = pd.read_parquet(run_dir / "task_results_table.parquet")

        # Split the task_results_table into a table for each metric
        metrics_tables = self.split_metrics_table(task_results_table)

        # Save the metrics tables
        for metric, df in metrics_tables.items():
            df.to_parquet(metrics_dir / f"per_sample_{metric}.parquet", index=False)


def main(args: argparse.Namespace) -> None:
    # Create the main working directory if it doesn't exist
    MAIN_WORKING_DIR.mkdir(parents=True, exist_ok=True)
    # Set the curret working directory to `wandb_data`
    os.chdir(MAIN_WORKING_DIR)

    # Download the run data
    wandb_downloader = WandbDownloader(project=args.project, entity=args.entity)
    wandb_downloader.download(args.run_name)

    # Process the run data
    data_processor = DataProcessor()
    data_processor.process(args.run_name)

    hf_api = HfApi()

    # Make sure the repo exists
    hf_api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True, private=True)

    # Upload the processed data to the repo
    hf_api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=RUNS_DIR / args.run_name,
        path_in_repo=str(RUNS_DIR / args.run_name),
        commit_message=f"Adding run {args.run_name} data to the repo at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--entity", type=str, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--repo-id", type=str, default="openbench-data")
    args = parser.parse_args()
    main(args)
