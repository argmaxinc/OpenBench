<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/openbench-light.png">
  <source media="(prefers-color-scheme: light)" srcset="assets/openbench-dark.png">
  <img alt="OpenBench Logo" src="assets/openbench-light.png">
</picture>

[![Paper](https://img.shields.io/badge/Paper-📄-blue)](https://www.arxiv.org/abs/2507.16136)
[![Discord](https://img.shields.io/discord/1171912382512115722?style=flat&logo=discord&logoColor=969da4&label=Discord&labelColor=353a41&color=32d058&link=https%3A%2F%2Fdiscord.gg%2FG5F5GZGecC)](https://discord.gg/G5F5GZGecC)

> [!NOTE]
> The OpenBench code is licensed under the MIT License. However, please note that:
> - SpeakerKit CLI and other integrated systems have their own licenses that apply
> - The datasets used in this benchmark have their own licenses and usage restrictions (see [Diarization Datasets](#diarization-datasets) section for details)

> [!IMPORTANT]
> **OpenBench** is the evolution of **SDBench**, originally introduced in the paper *["SDBench: A Comprehensive Benchmark Suite for Speaker Diarization"](https://arxiv.org/abs/2507.16136)*. While SDBench focused specifically on speaker diarization, OpenBench has expanded to encompass a broader range of speech processing tasks and is designed to accommodate future modalities beyond speech.

`OpenBench` is an open-source benchmarking framework for speech processing systems. Originally focused on speaker diarization (as SDBench), the framework has evolved to support comprehensive evaluation of:

- **Speaker Diarization**: Identifying "who spoke when" in audio recordings
- **Speech Transcription**: Converting speech to text (ASR)
- **Orchestration**: Combined diarization and transcription systems
- **Streaming Transcription**: Real-time speech-to-text
- **Future Extensions**: Designed to accommodate additional speech tasks and potentially other modalities

The primary objective is to promote standardized, reproducible, and continuous evaluation of open-source and proprietary speech processing systems across on-device and server-side implementations.

Key features include:
- **Command Line Interface (CLI)**: Easy-to-use CLI for evaluation, inference, and exploration
- Simple interface to wrap your diarization, ASR, or ASR + diarization system
- Easily accessible and extensible metrics following `pyannote` standard metric implementations
- Modular and convenient configuration management through `hydra`
- Out-of-the-box `Weights & Biases` logging
- Availability of 13+ commonly used datasets (Original dataset license restrictions apply)

> [!TIP]
> Want to add your own diarization, ASR, or orchestration pipeline? Check out our [Adding a New Diarization Pipeline](#adding-a-new-diarization-pipeline) section for a step-by-step guide!

> [!IMPORTANT]
> Before getting started, please note that some datasets in our [Datasets](#datasets) section require special access or have license restrictions. While we provide dataset preparation utilities in `common/download_dataset`, you'll need to procure the raw data independently for these datasets. See the dataset table for details on access requirements.

## Results

These are the results reported in the `SDBench` paper. For more details please refer to the [paper](https://arxiv.org/abs/2507.16136).

> **Note:** Each cell in the table below shows `DER / Speed Factor`, where **DER** is the Diarization Error Rate (lower is better) and **Speed Factor** indicates how many times faster than real time the system processes the audio (higher is better). Cells with `-` indicate that the system/dataset combination was not evaluated.

| Dataset               | AWS Transcribe   | Deepgram     | Picovoice   | Pyannote    | Pyannote-AI   | SpeakerKit   |
|-----------------------|------------------|--------------|-------------|-------------|---------------|--------------|
| AISHELL-4             | 22.5% / 10x      | 71.7% / 130x | -           | 12.2% / 55x | 11.2% / 62x   | 12.7% / 476x |
| AMI-IHM               | 28.7% / 11x      | 35.4% / 216x | 35.4% / 59x | 18.9% / 53x | 15.8% / 45x   | 20.6% / 463x |
| AMI-SDM               | 37.0% / 10x      | 42.5% / 241x | -           | 22.9% / 54x | 18.3% / 62x   | 23.7% / 458x |
| AVA-AVD               | 61.5% / 3x       | 68.2% / 187x | -           | 48.5% / 28x | 46.5% / 35x   | 51.9% / 426x |
| AliMeeting            | 42.3% / 9x       | 80.7% / 157x | -           | 25.1% / 29x | 19.2% / 45x   | 26.5% / 442x |
| American-Life-Podcast | 23.5% / 10x      | 28.9% / 231x | -           | 28.8% / 54x | 29.3% / 58x   | 36.9% / 481x |
| CallHome              | 36.8% / 2x       | 63.9% / 63x  | 54.2% / 61x | 28.8% / 53x | 19.8% / 20x   | 31.0% / 263x |
| DIHARD-III            | 35.5% / 8x       | 36.8% / 154x | -           | 23.6% / 28x | 17.1% / 39x   | 23.6% / 433x |
| EGO4D                 | 60.9% / 6x       | 70.7% / 127x | -           | 51.7% / 54x | 45.8% / 34x   | 54.0% / 436x |
| Earnings-21           | 17.8% / 9x       | -            | -           | 9.5% / 54x  | 9.3% / 47x    | 9.5% / 496x  |
| ICSI                  | 46.0% / 11x      | -            | -           | 34.4% / 52x | 31.4% / 62x   | 35.1% / 447x |
| MSDWILD               | 40.2% / 1x       | 64.5% / 43x  | -           | 31.7% / 53x | 26.2% / 15x   | 35.2% / 216x |
| VoxConverse           | 13.0% / 6x       | 36.4% / 210x | -           | 11.0% / 53x | 9.8% / 50x    | 12.1% / 462x |


## 🚀 Roadmap

- [x] Distribute SpeakerKit CLI for reproduction
- [ ] Living Benchmark, running every other month

## SpeakerKit Reproduction

If you want to reproduce `SpeakerKit` benchmark values please contact to get access to a CLI and an api-key [speakerkitpro@argmaxinc.com](mailto:speakerkitpro@argmaxinc.com).

If you already have access please update the `speakerkit.yaml` and follow the rest of instructions to setup the environment and run evals

## Setting up the environment
<details>
<summary> Click to expand </summary>

In order to get started, first make sure you have `uv` installed. The [official documentation](https://docs.astral.sh/uv/getting-started/installation/) has instructions for how to install the `uv` CLI.

If you already have `uv` installed you can run `make setup` to install the dependencies and set up the environment.
If you use `conda` or `venv` directly to manage your python environment you can install uv with `pip install uv` and then run `make setup` to install the dependencies.

Example with `conda`:
```bash
conda create -n <your-env-name> python=3.11
conda activate <your-env-name>
pip install uv
make setup
```

Alternatively, you can use `uv` directly to manage the environment:
```bash
# Install dependencies and create virtual environment
uv sync

# Activate the environment (if needed)
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows
```
</details>

## Command Line Interface (CLI)
<details>
<summary> Click to expand </summary>

OpenBench provides a powerful command-line interface for easy interaction with the benchmarking framework. The CLI offers three main commands for different use cases:

### Available Commands

#### `evaluate` - Run Benchmark Evaluations
Run comprehensive evaluations of your pipelines on datasets with configurable metrics.

```bash
# Evaluate using pipeline and dataset aliases
openbench-cli evaluate \
    --pipeline pyannote \
    --dataset voxconverse \
    --metrics der \
    --metrics jer \
    --use-wandb \
    --wandb-project my-evaluation

# Evaluate using a configuration file
openbench-cli evaluate \
    --evaluation-config config/my_evaluation.yaml \
    --evaluation-config-overrides wandb.project=my-project

# Get help and see available options
openbench-cli evaluate --help
```

#### `inference` - Run Single Audio Inference
Test your pipeline on individual audio files for quick validation.

```bash
# Run inference on a single audio file
openbench-cli inference \
    --pipeline pyannote \
    --audio-path path/to/audio.wav \
    --output-path path/to/output.json

# Run inference with custom configuration
openbench-cli inference \
    --pipeline pyannote \
    --audio-path path/to/audio.wav \
    --pipeline-config '{"min_speakers": 2, "max_speakers": 5}'
```

#### `summary` - Explore Available Resources
Get an overview of all available pipelines, datasets, metrics, and their compatibility.

```bash
# Show all available pipelines, datasets, and metrics
openbench-cli summary

# Show only pipelines
openbench-cli summary --disable-datasets --disable-metrics --disable-compatibility

# Show only compatibility matrix
openbench-cli summary --disable-pipelines --disable-datasets --disable-metrics

# Get detailed information
openbench-cli summary --verbose
```

### CLI Features

- **Pipeline Aliases**: Use friendly names like `pyannote`, `aws-diarization`, `whisperx` instead of class names
- **Dataset Aliases**: Access datasets with simple names like `voxconverse`, `earnings21`
- **Metric Selection**: Choose from available metrics like `der`, `jer`, `wer`
- **Weights & Biases Integration**: Built-in support for experiment tracking
- **Configuration Files**: Support for Hydra-based configuration management
- **Verbose Output**: Detailed logging for debugging and monitoring

### Environment Variables

Some pipelines require specific environment variables to be set:

```bash
# AWS Transcribe
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"

# Pyannote API
export PYANNOTE_API_KEY="your-api-key"

# SpeakerKit (contact speakerkitpro@argmaxinc.com for access)
export SPEAKERKIT_CLI_PATH="/path/to/speakerkit/cli"
export SPEAKERKIT_API_KEY="your-api-key"

# Other API-based pipelines
export PICOVOICE_API_KEY="your-api-key"
export DEEPGRAM_API_KEY="your-api-key"
export FIREWORKS_API_KEY="your-api-key"
export GLADIA_API_KEY="your-api-key"
export OPENAI_API_KEY="your-api-key"
```

For more details about pipeline requirements, run `openbench-cli summary` to see the full list of available pipelines and their descriptions.

</details>

## Datasets
<details>
<summary> Click to expand </summary>

OpenBench supports different types of pipelines (Diarization, Transcription, Orchestration, and Streaming Transcription) with specific dataset schemas for each task type.

### Diarization Datasets

The benchmark suite uses several speaker diarization datasets that are stored on the HuggingFace Hub. You can find all the datasets used in our evaluation in this [collection](https://huggingface.co/collections/argmaxinc/diarization-datasets-67646304c9b5e2cf9720ec48). The datasets available in the aforementioned collection are:

| Dataset Name | Out-of-the-box | License | How to Access |
|-------------|--------------|----------|---------------|
| [earnings21](https://github.com/revdotcom/speech-datasets/tree/main/earnings21) | ✅ | CC BY-SA 4.0 | Provided |
| [msdwild](https://github.com/X-LANCE/MSDWILD/tree/master) | ❌ | [MSDWild License Agreement](https://github.com/X-LANCE/MSDWILD/blob/master/MSDWILD_license_agreement.pdf) | Use `common/download_dataset.py` script |
| [icsi-meetings](https://groups.inf.ed.ac.uk/ami/icsi/download/) | ✅ | CC BY 4.0 | Provided |
| [aishell-4](https://www.openslr.org/111/) | ✅ | CC BY-SA 4.0 | Provided |
| [ali-meetings](https://www.openslr.org/119/) | ✅ | CC BY-SA 4.0 | Provided |
| [voxconverse](https://github.com/joonson/voxconverse) | ✅ | CC BY 4.0 | Provided |
| [ava-avd](https://github.com/zcxu-eric/AVA-AVD/tree/main/dataset) | ✅ | MIT | Provided |
| [ami-sdm](https://groups.inf.ed.ac.uk/ami/corpus/) | ✅ | CC BY 4.0 | Provided |
| [ami-ihm](https://groups.inf.ed.ac.uk/ami/corpus/) | ✅ | CC BY 4.0 | Provided |
| [american-life-podcast](https://github.com/jovistos/TALAD) | ❌ | Not disclosed | Use `common/download_dataset.py` script |
| [dihard-III](https://catalog.ldc.upenn.edu/LDC2022S14) | ❌ | [LDC License Agreement](https://catalog.ldc.upenn.edu/license/ldc-non-members-agreement.pdf) | Request access to LDC and use `common/download_dataset.py` script to parse |
| [callhome](https://catalog.ldc.upenn.edu/LDC2001S97) | ❌ | [LDC License Agreement](https://catalog.ldc.upenn.edu/license/ldc-non-members-agreement.pdf) | Request access to LDC and use `common/download_dataset.py` script to parse |
| [ego-4d](https://ego4d-data.org/docs/start-here/) | ❌ | [Ego4D License Agreement](https://ego4ddataset.com/ego4d-license/) | Request access to Ego4D and use `common/download_dataset.py` script to parse |

### Additional Dataset Collections

For other pipeline types, additional dataset collections are available:

- **[Transcription Datasets](https://huggingface.co/collections/argmaxinc/speech-to-text-datasets-687e885d80f794ec4b15d66d)**: Speech-to-text datasets compatible with transcription pipelines
- **[Orchestration Datasets](https://huggingface.co/collections/argmaxinc/diarized-speech-to-text-datasets-687fb17472f844e89a9ce98a)**: Diarized speech-to-text datasets for orchestration pipelines

From these datasets `voxconverse` and `ami` are not present as download options as they were already present in the HuggingFace Hub uploaded by [diarizers-community](https://huggingface.co/diarizers-community).

**Note**: You can use `openbench-cli summary` to see all available pre-registered datasets and their compatibility with different pipeline types.

### Dataset Schemas

OpenBench supports different pipeline types, each requiring specific dataset schemas:

#### Diarization Pipeline Schema
**Required columns:**
- `audio`: Audio column containing:
  - `array`: Audio waveform as numpy array of shape `(n_samples,)`
  - `sampling_rate`: Sample rate as integer
- `timestamps_start`: List of `float` containing start timestamps of segments in seconds
- `timestamps_end`: List of `float` containing end timestamps of segments in seconds
- `speakers`: List of `str` containing speaker IDs for each segment

**Optional columns:**
- `uem_timestamps`: List of tuples `[(start, end), ...]` containing Universal Evaluation Map (UEM) timestamps for evaluation

#### Transcription Pipeline Schema
**Required columns:**
- `audio`: Audio column containing:
  - `array`: Audio waveform as numpy array of shape `(n_samples,)`
  - `sampling_rate`: Sample rate as integer
- `transcript`: List of strings containing the words in the transcript

**Optional columns:**
- `word_timestamps_start`: List of `float` containing start timestamps for each word in seconds
- `word_timestamps_end`: List of `float` containing end timestamps for each word in seconds

#### Orchestration Pipeline Schema
**Required columns:**
- `audio`: Audio column containing:
  - `array`: Audio waveform as numpy array of shape `(n_samples,)`
  - `sampling_rate`: Sample rate as integer
- `transcript`: List of strings containing the words in the transcript
- `word_speakers`: List of strings containing speaker IDs for each word

**Optional columns:**
- `word_timestamps_start`: List of `float` containing start timestamps for each word in seconds
- `word_timestamps_end`: List of `float` containing end timestamps for each word in seconds

**Validation rules:**
- `word_speakers` and `transcript` must have the same length
- If `word_timestamps_start` and `word_timestamps_end` are provided, they must have the same length as `transcript`

#### Streaming Transcription Pipeline Schema
**Required columns:**
- `audio`: Audio column containing:
  - `array`: Audio waveform as numpy array of shape `(n_samples,)`
  - `sampling_rate`: Sample rate as integer
- `text`: String containing the reference transcript

**Optional columns:**
- `word_detail`: List of dictionaries containing word-level information with `start` and `stop` timestamps in samples (will be converted to seconds)

**Note**: Currently, most available datasets are optimized for diarization tasks. For transcription, orchestration, and streaming transcription pipelines, you may need to prepare additional annotations or use datasets that include the required fields for each task type.

### Downloading Datasets

If you want to reproduce the exact dataset downloads and processing, you can use our dataset downloading scripts. First, make sure you have the required dependencies installed as mentioned in the `Getting Started` section and also install the `dataset` dependencies doing `uv sync --group dataset`

After installing the dependencies, you can run the dataset downloading script at `common/download_dataset.py`. For example, to download the ICSI meetings dataset, you can run:

```bash
uv run python common/download_dataset.py --dataset icsi-meetings --hf-repo-owner <your-huggingface-username>
```

This will download the dataset and store locally at `raw_datasets/icsi-meetings` directory and upload it to the designated HuggingFace organization at `<your-huggingface-username>/icsi-meetings`. In case you only want to download and not push to HuggingFace, you can use the `--generate-only` flag.

For simplicity if you want to download all the datasets you can run:

```bash
# This will download all the datasets and store them in the raw_datasets directory
# Will not push to HuggingFace
make download-datasets
```

### NOTE:
- For datasets requiring Hugging Face access, make sure you have your `HF_TOKEN` environment variable set
- For the `American Life Podcast` dataset, you'll need Kaggle API credentials in `~/.kaggle/kaggle.json`
- For [`Callhome`](https://catalog.ldc.upenn.edu/LDC2001S97) and [`Dihard-III`](https://catalog.ldc.upenn.edu/LDC2022S14) you need to acquire the datasets from LDC first and then set their paths in the following env variables:
    - `DIHARD_DATASET_DIR` if not specified it will assume the directory lives at `~/third_dihard_challenge_eval/data`
    - `CALLHOME_AUDIO_ROOT` if not specified it will assume the directory lives at `~/callhome/nist_recognition_evaluation/r65_8_1/sid00sg1/data`
- The downloaded datasets will be stored in the `raw_datasets` directory (which is gitignored):

</details>

## Adding a New Pipeline

<details>
<summary> Click to expand </summary>

OpenBench can be used as a library to evaluate your own diarization, transcription, or orchestration pipelines. The framework supports three types of pipelines:

1. **Diarization Pipeline**: For speaker diarization tasks
2. **Transcription Pipeline**: For ASR/transcription tasks
3. **Orchestration Pipeline**: For combined diarization and transcription tasks

### Creating Your Pipeline

1. Create a new Python file (e.g., `my_pipeline.py`) and implement your pipeline:

```python
from typing import Callable

from openbench.dataset import DiarizationSample
from openbench.types import PipelineType
from openbench.pipeline.base import Pipeline, register_pipeline
from openbench.pipeline.diarization.common import DiarizationOutput, DiarizationPipelineConfig
from openbench.pipeline_prediction import DiarizationAnnotation

@register_pipeline
class MyDiarizationPipeline(Pipeline):
    _config_class = MyDiarizationConfig
    pipeline_type = PipelineType.DIARIZATION

    def build_pipeline(self) -> Callable[[dict], dict]:
        # Initialize your model/function and return a callable
        return my_diarizer_function

    def parse_input(self, input_sample: DiarizationSample) -> dict:
        # Convert DiarizationSample to your model's input format
        return {
            "waveform": input_sample.waveform,
            "sample_rate": input_sample.sample_rate
        }

    def parse_output(self, output: dict) -> DiarizationOutput:
        # Convert your model's output to DiarizationOutput
        return DiarizationOutput(prediction=annotation)
```

2. Create a configuration class for your pipeline:

```python
from pydantic import Field
from openbench.pipeline.diarization.common import DiarizationPipelineConfig

class MyDiarizationConfig(DiarizationPipelineConfig):
    model_path: str = Field(..., description="Path to model weights")
    threshold: float = Field(0.5, description="Detection threshold")
    num_speakers: int | None = Field(None, description="Number of speakers (optional)")
```

3. Create a configuration file for your pipeline:

```yaml
# my_pipeline_config.yaml
out_dir: ./my_pipeline_logs
model_path: /path/to/model
threshold: 0.5
num_speakers: null
```

### Using Your Pipeline

The CLI is currently limited to the pre-implemented pipelines in the library. For custom pipelines, you'll need to use the library directly:

```python
from openbench.runner import BenchmarkConfig, BenchmarkRunner, WandbConfig
from openbench.metric import MetricOptions
from openbench.dataset import DiarizationDatasetConfig

from my_pipeline import MyDiarizationPipeline, MyDiarizationConfig

# Create pipeline configuration
pipeline_config = MyDiarizationConfig(
    model_path="/path/to/model",
    threshold=0.5,
    num_speakers=None,
    out_dir="./my_pipeline_logs"
)

# Create benchmark configuration
benchmark_config = BenchmarkConfig(
    wandb_config=WandbConfig(
        project_name="my-diarization-benchmark",
        run_name="my-pipeline-evaluation",
        tags=["my-pipeline", "evaluation"],
        wandb_mode="online"  # or "offline" for local testing
    ),
    metrics={
        MetricOptions.DER: {},  # Diarization Error Rate
        MetricOptions.JER: {},  # Jaccard Error Rate
    },
    datasets={
        "voxconverse": DiarizationDatasetConfig(
            dataset_id="diarizers-community/voxconverse",
            split="test"
        )
    }
)

# Create pipeline instance
pipeline = MyDiarizationPipeline(pipeline_config)

# Create and run benchmark
runner = BenchmarkRunner(benchmark_config, [pipeline])
benchmark_result = runner.run()

print(benchmark_result.global_results[0])
```

2. For parallel processing, you can configure the number of worker processes in your pipeline config:

```python
pipeline_config = MyDiarizationConfig(
    model_path="/path/to/model",
    threshold=0.5,
    num_speakers=None,
    out_dir="./my_pipeline_logs",
    num_worker_processes=4,  # Number of parallel workers
    per_worker_chunk_size=2  # Samples per worker
)
```

3. To use Weights & Biases for experiment tracking, make sure to:
   - Set up your W&B account and get your API key
   - Make sure you're logged into your W&B account otherwise run `wandb login`
   - Configure the `wandb_config` in your benchmark configuration

The BenchmarkRunner will automatically:
- Run your pipeline on the specified datasets
- Calculate metrics for each sample
- Aggregate results globally
- Log everything to Weights & Biases (if configured)
- Handle parallel processing if enabled (specially interesting for APIs)
- Generate detailed reports and artifacts

### Pipeline Types and Requirements

#### Diarization Pipeline
- Must implement `build_pipeline()`, `parse_input()`, and `parse_output()`
- Input parsing should convert `DiarizationSample` to your model's expected format
- Output parsing should return a `DiarizationOutput` with a `prediction` field

#### Transcription Pipeline
- Must implement `build_pipeline()`, `parse_input()`, and `parse_output()`
- Input parsing should convert `DiarizationSample` to your model's expected format
- Output parsing should return a `TranscriptionOutput` with a `prediction` field

#### Orchestration Pipeline
- Must implement `build_pipeline()`, `parse_input()`, and `parse_output()`
- Can either:
  - Implement end-to-end diarization and transcription
  - Use `PostInferenceMergePipeline` to combine separate diarization and transcription pipelines
- Output parsing should return an `OrchestrationOutput` with a `prediction` field and optionaly `diarization` and `transcription` results


</details>

## Hydra Configuration
<details>
<summary> Click to expand </summary>

The benchmark suite uses Hydra for configuration management, providing a flexible and modular way to configure evaluation runs. The configuration files are organized in the following structure:

```
config
├── evaluation_config.yaml                      # Main evaluation configuration
├── benchmark_config                            # Base configurations for benchmarking
│   ├── datasets                                # Dataset-specific configs
│   ├── wandb_config                            # Weights & Biases logging configs
│   └── base.yaml                               # Default benchmark_config used in evaluation_config.yaml
└── pipeline_configs                            # Predefined pipeline configurations for ease of use
    ├── my_pipeline
    │   ├── base.yaml                           # Default config used in my_pipeline.yaml
    │   └── config
    │       ├── base.yaml                       # Default config used in MyPipeline
    │       └── diarization_config
    │           ├── chunking_config             # Defines different useful chunking configurations
    │           ├── cluster_definition          # Defines different useful cluster definitions
    │           ├── speaker_embedder_config     # Defines different useful speaker embedder configurations
    │           ├── speaker_segmenter_config    # Defines different useful speaker segmenter configurations
    │           └── base.yaml                   # Default diarization_config used in evaluation_config.yaml
    ├── my_pipeline.yaml                        # Uses MyPipeline as default pipeline
    └── pyannote.yaml                           # Defines configuration for PyAnnotePipeline
```

### Running Evaluations with Different Configurations

You can easily customize your evaluation runs using Hydra's override syntax. Here are some common usage patterns:

#### Using the CLI

All Hydra configuration features work with the CLI using `--evaluation-config` and `--evaluation-config-overrides`:

```bash
# Run evaluation with a specific config file
openbench-cli evaluate --evaluation-config config/my_evaluation.yaml

# Override configuration parameters
openbench-cli evaluate \
    --evaluation-config config/my_evaluation.yaml \
    --evaluation-config-overrides wandb.project=my-project pipeline_configs.MyPipeline.config.threshold=0.7

# See the resulting configuration
openbench-cli evaluate --evaluation-config config/my_evaluation.yaml --help
```

#### Using the evaluation.py script provided in the repo (old-way)

1. **Selecting Specific Pipelines**
```bash
# Run evaluation with only MyPipeline
uv run python evaluation.py pipeline_configs=my_pipeline
```

2. **Modifying Pipeline Parameters**
You can override specific configuration parameters in two ways:

a. **Override by Value**:
```bash
# Change the speaker segmenter stride
uv run python evaluation.py \
    pipeline_configs=my_pipeline \
    pipeline_configs.MyPipeline.config.diarization_config.speaker_segmenter_config.variant_name=stride_2
```

b. **Override by Config**:
```bash
# Use a predefined speaker segmenter configuration
uv run python evaluation.py \
    pipeline_configs=my_pipeline \
    pipeline_configs/MyPipeline/config/diarization_config/speaker_segmenter_config=stride_2
```

Note: Use `-h` flag with any command to see the resulting configuration:
```bash
uv run python evaluation.py pipeline_configs=my_pipeline -h
```
</details>
