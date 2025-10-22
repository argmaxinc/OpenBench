# OpenBench Benchmarks

<br/>

- [Speaker Diarization](#speaker-diarization)
  - [Benchmarked Systems](#benchmarked-systems)
  - [Benchmarked Datasets](#benchmarked-datasets)
  - [Diarization Error Rate (DER)](#diarization-error-rate-der)
  - [Speed Factor (SF)](#speed-factor-sf)
  - [Speaker Count Accuracy (SCA)](#speaker-count-accuracy-sca)
- [Real-time Transcription](#real-time-transcription)
  - [Benchmarked Systems](#benchmarked-systems-1)
  - [Word Error Rate (WER)](#word-error-rate-wer)
  - [Streaming Latency](#streaming-latency)
  - [Confirmed Streaming Latency](#confirmed-streaming-latency)
- [Speaker-Attributed Transcription](#speaker-attributed-transcription)
  - [Benchmarked Systems](#benchmarked-systems-3)
  - [Benchmarked Datasets](#benchmarked-datasets-2)
  - [Word Error Rate (WER)](#word-error-rate-wer-2)
  - [Word Diarization Error Rate (WDER)](#word-diarization-error-rate-wder)
  - [Speed Factor (SF)](#speed-factor-sf-2)
- [Custom Vocabulary](#custom-vocabulary)
  - [Benchmarked Systems](#benchmarked-systems-4)
  - [Benchmarked Datasets](#benchmarked-datasets-3)
  - [Word Error Rate (WER)](#word-error-rate-wer-3)
  - [Precision](#precision)
  - [Recall](#recall)
  - [F-score](#f-score)

<br/>

# Speaker Diarization 

## Benchmarked Systems

<details>
<summary>Click to expand</summary>

> **Note:** If a cell in the tables below is `-`, it means that the system/dataset combination was not evaluated due to timeout constaints or lack of credits.

### AWS Transcribe
- **Latest Run:** `2025-02-17`
- **Model Version:** `default`
- **Configuration:** Using `AWS Transcribe` API with `ShowSpeakerLabels` enabled and `MaxSpeakerLabels` set to 30 (maximum allowed by the API). See [AWS Transcribe Documentation](https://docs.aws.amazon.com/transcribe/latest/dg/diarization.html) for more details.
- **Code Reference:** [openbench/pipeline/diarization/aws](https://github.com/argmaxinc/OpenBench/blob/main/src/openbench/pipeline/diarization/aws.py)
- **Hardware**: Unknown (Cloud API)

### Deepgram
- **Latest Run:** `2025-06-27`
- **Model Version:** `nova-3`
- **Configuration:** Using `Deepgram`'s Python SDK for transcription with `diarize` and `detect_language` enabled. See [deepgram-python-sdk](https://github.com/deepgram/deepgram-python-sdk) for more details.
- **Code Reference:** [openbench/pipeline/diarization/diarization_deepgram](https://github.com/argmaxinc/OpenBench/blob/main/src/openbench/pipeline/diarization/diarization_deepgram.py)
- **Hardware**: Unknown (Cloud API)

### pyannote
- **Latest Run:** `2025-02-17`
- **Model Version:** `speaker-diarization-3.1`
- **Configuration:** `Pyannote` OSS using [pyannote-audio](https://github.com/pyannote/pyannote-audio) version v3.3.2 and default settings for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) running inference with `float16` precision.
- **Code Reference:** [openbench/pipeline/diarization/pyannote/pipeline](https://github.com/argmaxinc/OpenBench/blob/main/src/openbench/pipeline/diarization/pyannote/pipeline.py)
- **Hardware**: M2 Ultra Mac Studio

### pyannoteAI
- **Latest Run:** `2025-02-17`
- **Model Version:** `pyannote/precision-1`
- **Configuration:** Job polling based on `X-RateLimit-Remaining` and `X-RateLimit-Reset` headers which leads to sub-second polling checks. See [pyannoteAI Documentation](https://docs.pyannote.ai/api-reference/diarize) for more details.
- **Code Reference:** [openbench/pipeline/diarization/pyannote-api](https://github.com/argmaxinc/OpenBench/blob/main/src/openbench/pipeline/diarization/pyannote_api.py)
- **Hardware**: Unknown (Cloud API)

### Argmax
- **Latest Run:** `2025-09-29`
- **Model Version:** `pyannote/community-1` (`pyannote-v4`)
- **Configuration:** Argmax SDK 1.8.2 `SpeakerKit` CLI with default settings. See [Interspeech 2025 Paper](https://www.isca-archive.org/interspeech_2025/durmus25_interspeech.html) for more details.
- **Code Reference:** [openbench/pipeline/diarization/speakerkit](https://github.com/argmaxinc/OpenBench/blob/main/src/openbench/pipeline/diarization/speakerkit.py)
- **Hardware**: M2 Ultra Mac Studio

</details>

<br/>

## Benchmarked Datasets

<details>
<summary>Click to expand</summary>

### AISHELL-4
- **Language:** Chinese
- **Domain:** In-Person Meeting
- **Description:** A large-scale Chinese meeting dataset containing multi-speaker conversations recorded in real meeting rooms with multiple microphones.

### AMI-IHM
- **Language:** English
- **Domain:** In-Person Meeting
- **Description:** The AMI Individual Headset Microphone dataset contains English meeting recordings where each participant wears a headset microphone, providing clean individual speaker audio.

### AMI-SDM
- **Language:** English
- **Domain:** In-Person Meeting
- **Description:** The AMI Single Distant Microphone dataset contains the same meetings as AMI-IHM but recorded using a single microphone placed in the center of the room, creating more challenging audio conditions.

### AVA-AVD
- **Language:** Multilingual
- **Domain:** YouTube/In-the-Wild
- **Description:** The AVA Audio-Visual Diarization dataset contains YouTube videos with diverse content types, languages, and recording conditions, making it challenging for speaker diarization systems.

### AliMeeting
- **Language:** Chinese
- **Domain:** In-Person Meeting
- **Description:** A Chinese meeting dataset featuring real-world business meetings with multiple speakers, overlapping speech, and natural conversation patterns.

### CALLHOME
- **Language:** Multilingual
- **Domain:** Phone Call
- **Description:** A collection of telephone conversations in multiple languages, featuring natural speech patterns and the audio quality challenges typical of phone calls.

### DIHARD-III
- **Language:** Multilingual
- **Domain:** Multi Domain
- **Description:** The DIHARD-III challenge dataset contains diverse audio recordings from multiple domains (meetings, courts, audiobooks, etc.) in various languages, designed to test robust speaker diarization systems.

### EGO4D
- **Language:** Multilingual
- **Domain:** In-the-Wild
- **Description:** A large-scale egocentric video dataset captured from first-person perspectives, containing natural conversations and interactions in real-world environments with varying audio conditions.

### Earnings-21
- **Language:** English
- **Domain:** Meeting
- **Description:** A dataset of corporate earnings call recordings featuring financial presentations and Q&A sessions with executives, analysts, and investors.

### ICSI
- **Language:** English
- **Domain:** In-Person Meeting
- **Description:** The ICSI Meeting Corpus contains academic research meetings with multiple participants, featuring technical discussions and natural conversation flow.

### MSDWILD
- **Language:** Multilingual
- **Domain:** YouTube/In-the-Wild
- **Description:** A diverse collection of YouTube videos featuring multiple speakers in various languages and contexts, including interviews, discussions, and entertainment content.

### VoxConverse
- **Language:** English
- **Domain:** YouTube/In-the-Wild
- **Description:** A dataset of English YouTube videos containing multi-speaker conversations from various content types, including interviews, debates, and talk shows.

</details>

<br/>

## Diarization Error Rate (DER)

<details>
<summary>Click to expand</summary>


**What it measures:** DER quantifies how accurately a system identifies "who spoke when" in an audio recording. It measures the total time that speakers are incorrectly labeled, including missed speech, falsely detected speech, and speaker confusion.

**How to interpret:** Lower values are better. A DER of 0.0 would be perfect (no errors), while 1.0 means 100% error. A DER of 0.20 means 20% of the audio time has speaker labeling errors.

**Example:** In a 10-minute conversation, a DER of 0.15 means that for 1.5 minutes total, the system either missed speech, detected non-existent speech, or confused which speaker was talking.

</details>

| Dataset                | AWS Transcribe            | Deepgram             | pyannote | pyannoteAI              | Argmax     |
|------------------------|---------------------------|----------------------|----------|-------------------------|------------|
| AISHELL-4              | 0.22                      | 0.72                 | 0.12     | 0.11                    | 0.12       |
| AMI-IHM                | 0.29                      | 0.35                 | 0.19     | 0.16                    | 0.18       |
| AMI-SDM                | 0.37                      | 0.42                 | 0.23     | 0.18                    | 0.21       |
| AVA-AVD                | 0.61                      | 0.68                 | 0.48     | 0.47                    | 0.48       |
| AliMeeting             | 0.42                      | 0.81                 | 0.25     | 0.19                    | 0.23       |
| CallHome               | 0.37                      | 0.64                 | 0.29     | 0.20                    | 0.30       |
| DIHARD-III             | 0.36                      | 0.37                 | 0.24     | 0.17                    | 0.22       |
| EGO4D                  | 0.61                      | 0.71                 | 0.52     | 0.46                    | 0.48       |
| Earnings-21            | 0.18                      | -                    | 0.10     | 0.09                    | 0.10       |
| ICSI                   | 0.46                      | -                    | 0.34     | 0.31                    | 0.35       |
| MSDWILD                | 0.40                      | 0.64                 | 0.32     | 0.26                    | 0.33       |
| VoxConverse            | 0.13                      | 0.36                 | 0.11     | 0.10                    | 0.11       |

<br/><br/>

## Speed Factor (SF)

<details>
<summary>Click to expand</summary>


**What it measures:** Speed Factor compares how much faster (or slower) a system processes audio compared to real-time. It's calculated as $SF = \dfrac{Duration_{audio}}{Duration_{prediction}}$.

**How to interpret:** Values above 1x mean the system is faster than real-time. Values below 1x mean slower than real-time. Higher values indicate faster processing.

**Example:** An SF of 10x means the system processes 10 seconds of audio in 1 second. An SF of 0.5x means it takes 2 seconds to process 1 second of audio.

</details>

| Dataset                 | AWS Transcribe | Deepgram | Picovoice | pyannote | pyannoteAI | Argmax |
|-------------------------|---------------------------|----------------------|-----------|----------|-------------------------|------------|
| AISHELL-4               | 10                       | 130                  | -         | 55       | 62                     | 476        |
| AMI-IHM                 | 11                       | 216                  | 59        | 53       | 45                     | 463        |
| AMI-SDM                 | 10                       | 241                  | -         | 54       | 62                     | 458        |
| AVA-AVD                 | 3                        | 187                  | -         | 28       | 35                     | 426        |
| AliMeeting              | 9                        | 157                  | -         | 29       | 45                     | 442        |
| American-Life-Podcast   | 10                       | 231                  | -         | 54       | 58                     | 481        |
| CallHome                | 2                        | 63                   | 61        | 53       | 20                     | 263        |
| DIHARD-III              | 8                        | 154                  | -         | 28       | 39                     | 433        |
| EGO4D                   | 6                        | 127                  | -         | 54       | 34                     | 436        |
| Earnings-21             | 9                        | -                    | -         | 54       | 47                     | 496        |
| ICSI                    | 11                       | -                    | -         | 52       | 62                     | 447        |
| MSDWILD                 | 1                        | 43                   | -         | 53       | 15                     | 216        |
| VoxConverse             | 6                        | 210                  | -         | 53       | 50                     | 462        |

<br/><br/>

## Speaker Count Accuracy (SCA)

<details>
<summary>Click to expand</summary>


**What it measures:** SCA measures how accurately a system identifies the total number of unique speakers in an audio recording, regardless of when they spoke.

**How to interpret:** Expressed as a percentage, where 100% means perfect speaker count detection. Lower percentages indicate the system overestimated or underestimated the number of speakers.

**Example:** If there are 4 speakers in a recording and the system detects 3 speakers, the SCA would be 0%.

</details>

| Dataset                 | AWS Transcribe | Deepgram |  pyannote | pyannoteAI | Argmax |
|-------------------------|----------------|----------|-----------|------------|--------|
| AISHELL-4               | 75             | 30       | 5         | 15         | 60     |
| AMI-IHM                 | 94             | 56       | 0         | 12         | 75     |
| AMI-SDM                 | 56             | 88       | 6         | 12         | 69     |
| AVA-AVD                 | 13             | 6        | 13        | 9          | 13     |
| AliMeeting              | 90             | 5        | 40        | 55         | 65     |
| American-Life-Podcast   | 11             | 14       | 8         | 8          | 8      |
| CallHome                | 60             | 33       | 74        | 48         | 42     |
| DIHARD-III              | 72             | 60       | 60        | 58         | 45     |
| EGO4D                   | 34             | 16       | 24        | 24         | 48     |
| Earnings-21             | 50             | -        | 50        | 64         | 55     |
| ICSI                    | 43             | -        | 7         | 13         | 7      |
| MSDWILD                 | 39             | 15       | 34        | 35         | 28     |
| VoxConverse             | 46             | 39       | 42        | 38         | 45     |

</br><br/>
# Real-time Transcription 

## Benchmarked Systems

<details>
<summary>Click to expand</summary>

### Deepgram
- **Latest Run:** `08-12-2025`
- **Configuration:** [Code](https://github.com/argmaxinc/OpenBench/blob/main/src/openbench/pipeline/streaming_transcription/deepgram.py#L67)
- **Code Reference:** [openbench/pipeline/streaming_transcription/deepgram](https://github.com/argmaxinc/OpenBench/blob/main/src/openbench/pipeline/streaming_transcription/deepgram.py)
- **Hardware**: Unknown (Cloud API)

### OpenAI
- **Latest Run:** `08-12-2025`
- **Configuration:** [Code](https://github.com/argmaxinc/OpenBench/blob/main/src/openbench/pipeline/streaming_transcription/openai.py#L63)
- **Code Reference:** [openbench/pipeline/streaming_transcription/openai](https://github.com/argmaxinc/OpenBench/blob/main/src/openbench/pipeline/streaming_transcription/openai.py)
- **Hardware**: Unknown (Cloud API)

### Gladia
- **Latest Run:** `09-16-2025`
- **Configuration:** [Code](https://github.com/argmaxinc/OpenBench/blob/main/src/openbench/pipeline/streaming_transcription/gladia.py#L112)
- **Code Reference:** [openbench/pipeline/streaming_transcription/gladia](https://github.com/argmaxinc/OpenBench/blob/main/src/openbench/pipeline/streaming_transcription/gladia.py)
- **Hardware**: Unknown (Cloud API)

### Argmax (Parakeet V3)¹
- **Latest Run:** `09-12-2025`
- **Configuration:** Reuses the Deepgram pipeline with `DEEPGRAM_HOST_URL=ws://localhost:port` while [Argmax Local Server](https://www.argmaxinc.com/blog/argmax-local-server) is running with our compressed optimized model `--model parakeet-v3_494MB` at `ws://localhost:port`
- **Code Reference:** [openbench/pipeline/streaming_transcription/deepgram](https://github.com/argmaxinc/OpenBench/blob/main/src/openbench/pipeline/streaming_transcription/deepgram.py)
- **Hardware**: M2 Ultra Mac Studio

### Argmax (Whisper Large V3 Turbo)¹
- **Latest Run:** `09-12-2025`
- **Configuration:** Reuses the Deepgram pipeline with `DEEPGRAM_HOST_URL=ws://localhost:port` while [Argmax Local Server](https://www.argmaxinc.com/blog/argmax-local-server) is running with our compressed optimized model `--model large-v3-v20240930_626MB` at `ws://localhost:port`
- **Code Reference:** [openbench/pipeline/streaming_transcription/deepgram](https://github.com/argmaxinc/OpenBench/blob/main/src/openbench/pipeline/streaming_transcription/deepgram.py)
- **Hardware**: M2 Ultra Mac Studio

</details>
<br/>

## Word Error Rate (WER)

<details>
<summary>Click to expand</summary>


**What it measures:** WER measures speech-to-text accuracy by counting the word-level edits - substitutions, deletions, and insertions — needed to turn a transcript into the reference, then dividing by the reference length to give a percentage.

**How to interpret:** Lower values are better. A WER of 0.0% would be perfect (no errors), while 100% means complete error and values may exceeed 100%.

**Example:** In a 100-word reference transcript, a WER of 15% means there are 15 total word-level mistakes — some mix of substitutions (confusion), deletions (omission), and insertions (hallucination).

</details>

| Dataset        | Deepgram<br/>(nova-3) | OpenAI <br/>(GPT-4o) | Gladia |  Argmax <br/>(Parakeet V3) |  Argmax <br/>(Whisper Large V3 Turbo) |
|----------------|-----------------------|----------------------|--------|----------------------------|---------------------------------------|
| timit-stitched | 2.36                  | 2.47                 | 2.1    |        2.08                | 2.17                                  |

<br/><br/>
## Streaming Latency

<details>
<summary>Click to expand</summary>


**What it measures:** Streaming Latency measures the delay between when audio is sent to the system and when interim transcription (subject to change) results are received. Interim results are also referred to as partial, hypothesis and mutable results. Please refer to the [implementation](https://github.com/argmaxinc/OpenBench/blob/main/src/openbench/metric/streaming_latency_metrics/latency_metrics.py#L42) for details. This metric is adapted from Deepgram's [definition](https://developers.deepgram.com/docs/measuring-streaming-latency). The difference is that we rely on ground-truth timestamps from the dataset instaed of model-predicted timestamps.

**How to interpret:** Lower values are better. This represents how quickly the system provides interim transcription results during real-time transcription. Values closer to 0 indicate near real-time responsiveness. N/A indicates that the system does not allow interim results.

**Example:** A streaming latency of 0.5s means that on average, interim transcription results arrive 0.5 seconds after the corresponding audio was sent to the system.

</details>

| Dataset        | Deepgram<br/>(nova-3) | OpenAI <br/>(GPT-4o) | Gladia |  Argmax <br/>(Parakeet V3) |  Argmax <br/>(Whisper Large V3 Turbo) |
|----------------|-----------------------|----------------------|--------|----------------------------|---------------------------------------|
| timit-stitched | 1.03                  | N/A                  | 0.64   |        0.54                | 0.94                                  |


</br></br>

## Confirmed Streaming Latency

<details>
<summary>Click to expand</summary>


**What it measures:** Confirmed Streaming Latency measures the delay between when audio is sent to the system and when final transcription results are received. Final results are also referred to as confirmed, and immutable results. Please refer to the [implementation](https://github.com/argmaxinc/OpenBench/blob/main/src/openbench/metric/streaming_latency_metrics/latency_metrics.py#L42) for details. This metric is adapted from Deepgram's [definition](https://developers.deepgram.com/docs/measuring-streaming-latency). The difference is that we rely on ground-truth timestamps from the dataset instaed of model-predicted timestamps.

**How to interpret:** Lower values are better. This represents how quickly the system provides finalized transcription results during real-time transcription, in contrast to interim results which may still change. Values closer to 0 indicate near real-time responsiveness.

**Example:** A confirmed streaming latency of 2.0s means that on average, confirmed transcription results arrive 2.0 seconds after the corresponding audio was sent to the system. 

</details>

| Dataset        | Deepgram<br/>(nova-3) | OpenAI <br/>(GPT-4o) | Gladia |  Argmax <br/>(Parakeet V3) |  Argmax <br/>(Whisper Large V3 Turbo) |
|----------------|-----------------------|----------------------|--------|----------------------------|---------------------------------------|
| timit-stitched | 2.37                  | 56.95                | 2.72   |        5.51                | 2.51                                  |



---

¹ **Note:** Argmax's default configuration includes a 0.3-second system sleep to allow the accumulation of new audio frames. Developers can reconfigure to remove this sleep to achieve lower latency.



# Speaker-Attributed Transcription

## Benchmarked Systems

<details>
<summary>Click to expand</summary>

### Deepgram
- **Latest Run:** `2025-09-05`
- **Model Version:** `nova-3`
- **Configuration:** Deepgram's Python SDK for file transcription with `diarize` and `detect_language` enabled. See [deepgram-python-sdk](https://github.com/deepgram/deepgram-python-sdk) for more details.
- **Code Reference:** [openbench/pipeline/orchestration/orchestration_deepgram.py](https://github.com/argmaxinc/OpenBench/blob/main/src/openbench/pipeline/orchestration/orchestration_deepgram.py)
- **Hardware**: Unknown (Cloud API)

### Argmax (Whisper Large V3 Turbo)
- **Latest Run:** `2025-09-29`
- **Model Version:** `whisper-large-v3-turbo`
- **Configuration:** Argmax WhisperKit Pro with compressed Whisper Large V3 Turbo model (i.e. `large-v3-v20240930_626MB`) for speaker-attributed transcription and SpeakerKit Pro with `pyannote-v3-pro`.
- **Code Reference:** [openbench/pipeline/orchestration/orchestration_whisperkitpro.py](https://github.com/argmaxinc/OpenBench/blob/main/src/openbench/pipeline/orchestration/orchestration_whisperkitpro.py)
- **Hardware**: M2 Ultra Mac Studio


### Argmax (Parakeet V3)
- **Latest Run:** `2025-09-29`
- **Model Version:** `parakeet-v3`
- **Configuration:** Argmax WhisperKit Pro with compressed Parakeet V3 model (i.e. `parakeet-v3_494MB`) for speaker-attributed transcription and SpeakerKit Pro with `pyannote-v3-pro`.
- **Code Reference:** [openbench/pipeline/orchestration/orchestration_whisperkitpro.py](https://github.com/argmaxinc/OpenBench/blob/main/src/openbench/pipeline/orchestration/orchestration_whisperkitpro.py)
- **Hardware**: M2 Ultra Mac Studio

</details>

<br/>

## Benchmarked Datasets

<details>
<summary>Click to expand</summary>

### CallHome English (callhome-english)
- **Language:** English
- **Domain:** Phone Call
- **Description:** [LDC97S42](https://catalog.ldc.upenn.edu/LDC97S42) English subset of the CallHome dataset containing speaker labeled transcripts of telephone conversations with natural speech patterns and the audio quality challenges typical of phone calls.

### Earnings21
- **Language:** English
- **Domain:** Meeting
- **Description:** A dataset of corporate earnings call recordings featuring financial presentations and Q&A sessions with executives, analysts, and investors.

</details>

<br/>

## Word Error Rate (WER)

<details>
<summary>Click to expand</summary>


**What it measures:** WER measures speech-to-text accuracy by counting the word-level edits - substitutions, deletions, and insertions — needed to turn a transcript into the reference, then dividing by the reference length to give a percentage.

**How to interpret:** Lower values are better. A WER of 0.0% would be perfect (no errors), while 100% means complete error and values may exceeed 100%.

**Example:** In a 100-word reference transcript, a WER of 15% means there are 15 total word-level mistakes — some mix of substitutions (confusion), deletions (omission), and insertions (hallucination).

</details>

| Dataset          | Deepgram<br/>(nova-3) | Argmax<br/>(Whisper Large V3 Turbo) | Argmax<br/>(Parakeet V3) |
|------------------|-----------------------|:-----------------------------------:|:------------------------:|
| callhome-english | 10.22                 |               10.67                 |          9.78            |
| earnings21       | 7.38                  |                7.99                 |          6.97            |

<br/><br/>

## Word Diarization Error Rate (WDER)

<details>
<summary>Click to expand</summary>


**What it measures:** WDER measures the accuracy of speaker-attributed transcription by counting word-level errors in both transcription accuracy and speaker attribution. It combines word error rate with speaker diarization errors at the word level.

**How to interpret:** Lower values are better. A WDER of 0.0% would be perfect (no errors in both transcription and speaker attribution), while 100% means complete error.

**Example:** In a 100-word reference transcript with speaker labels, a WDER of 20% means there are 20 total word-level mistakes in either transcription accuracy or speaker attribution.

</details>

| Dataset          | Deepgram<br/>(nova-3) | Argmax<br/>(Whisper Large V3 Turbo) | Argmax<br/>(Parakeet V3) |
|------------------|-----------------------|:-----------------------------------:|:------------------------:|
| callhome-english | 5.01                  |                5.72                 |           5.99           |
| earnings21       | 6.18                  |                4.94                 |           5.16           |

<br/><br/>

## Speed Factor (SF)

<details>
<summary>Click to expand</summary>


**What it measures:** Speed Factor compares how much faster (or slower) a system processes audio compared to real-time. It's calculated as $SF = \dfrac{Duration_{audio}}{Duration_{prediction}}$.

**How to interpret:** Values above 1x mean the system is faster than real-time. Values below 1x mean slower than real-time. Higher values indicate faster processing.

**Example:** An SF of 10x means the system processes 10 seconds of audio in 1 second. An SF of 0.5x means it takes 2 seconds to process 1 second of audio.

</details>

| Dataset          | Deepgram<br/>(nova-3) | Argmax<br/>(Whisper Large V3 Turbo) | Argmax<br/>(Parakeet V3) |
|------------------|:---------------------:|:-----------------------------------:|:------------------------:|
| callhome-english |   102                 |                 13                  |           98             |
| earnings21       |   213                 |                 21                  |           232            |

<br/><br/>

---

# Custom Vocabulary

## Benchmarked Systems

<details>
<summary>Click to expand</summary>

### Deepgram
- **Latest Run:** `YYYY-MM-DD`
- **Model Version:** `nova-3`
- **Configuration:** Deepgram's Python SDK for transcription with custom vocabulary/keyword boosting enabled. See [deepgram-python-sdk](https://github.com/deepgram/deepgram-python-sdk) for more details.
- **Code Reference:** [openbench/pipeline/transcription/transcription_deepgram.py](https://github.com/argmaxinc/OpenBench/blob/main/src/openbench/pipeline/transcription/transcription_deepgram.py)
- **Hardware**: Unknown (Cloud API)

### OpenAI
- **Latest Run:** `YYYY-MM-DD`
- **Model Version:** `whisper-1`
- **Configuration:** OpenAI's Whisper API for transcription with prompt-based keyword boosting. See [OpenAI Whisper API](https://platform.openai.com/docs/guides/speech-to-text) for more details.
- **Code Reference:** [openbench/pipeline/transcription/transcription_openai.py](https://github.com/argmaxinc/OpenBench/blob/main/src/openbench/pipeline/transcription/transcription_openai.py)
- **Hardware**: Unknown (Cloud API)

### AssemblyAI
- **Latest Run:** `YYYY-MM-DD`
- **Model Version:** `default`
- **Configuration:** AssemblyAI's Python SDK for transcription with word boost feature enabled. See [AssemblyAI Python SDK](https://github.com/AssemblyAI/assemblyai-python-sdk) for more details.
- **Code Reference:** [openbench/pipeline/transcription/transcription_assemblyai.py](https://github.com/argmaxinc/OpenBench/blob/main/src/openbench/pipeline/transcription/transcription_assemblyai.py)
- **Hardware**: Unknown (Cloud API)

### Whisper OSS
- **Latest Run:** `YYYY-MM-DD`
- **Model Version:** `base`
- **Configuration:** Open-source Whisper implementation with custom vocabulary prompting. See [openai-whisper](https://github.com/openai/whisper) for more details.
- **Code Reference:** [openbench/pipeline/transcription/transcription_oss_whisper.py](https://github.com/argmaxinc/OpenBench/blob/main/src/openbench/pipeline/transcription/transcription_oss_whisper.py)
- **Hardware**: M2 Ultra Mac Studio

### Argmax (Parakeet V2)
- **Latest Run:** `YYYY-MM-DD`
- **Model Version:** `parakeet-v2`
- **Configuration:** Argmax WhisperKit Pro with compressed Parakeet V2 model (i.e. `parakeet-v2_476MB`) with custom vocabulary support.
- **Code Reference:** [openbench/pipeline/transcription/transcription_whisperkitpro.py](https://github.com/argmaxinc/OpenBench/blob/main/src/openbench/pipeline/transcription/transcription_whisperkitpro.py)
- **Hardware**: M2 Ultra Mac Studio

### Argmax (Parakeet V3)
- **Latest Run:** `YYYY-MM-DD`
- **Model Version:** `parakeet-v3`
- **Configuration:** Argmax WhisperKit Pro with compressed Parakeet V3 model (i.e. `parakeet-v3_494MB`) with custom vocabulary support.
- **Code Reference:** [openbench/pipeline/transcription/transcription_whisperkitpro.py](https://github.com/argmaxinc/OpenBench/blob/main/src/openbench/pipeline/transcription/transcription_whisperkitpro.py)
- **Hardware**: M2 Ultra Mac Studio

</details>

<br/>

## Benchmarked Datasets

<details>
<summary>Click to expand</summary>

### Earnings22 Keywords (earnings22-keywords)
- **Language:** English
- **Domain:** Corporate Earnings Calls
- **Description:** A specialized subset of corporate earnings call recordings designed to evaluate custom vocabulary and keyword boosting capabilities. Contains financial terminology, company names, and domain-specific jargon that benefits from vocabulary customization.

</details>

<br/>

## Word Error Rate (WER)

<details>
<summary>Click to expand</summary>

**What it measures:** WER measures speech-to-text accuracy by counting the word-level edits - substitutions, deletions, and insertions — needed to turn a transcript into the reference, then dividing by the reference length to give a percentage.

**How to interpret:** Lower values are better. A WER of 0.0% would be perfect (no errors), while 100% means complete error and values may exceed 100%.

**Example:** In a 100-word reference transcript, a WER of 15% means there are 15 total word-level mistakes — some mix of substitutions (confusion), deletions (omission), and insertions (hallucination).

</details>

| Dataset              | Deepgram<br/>(nova-3) | OpenAI<br/>(whisper-1) | AssemblyAI | Whisper OSS<br/>(large-v3-turbo) | Argmax<br/>(Parakeet V2) | Argmax<br/>(Parakeet V3) |
|----------------------|:---------------------:|:----------------------:|:----------:|:----------------------:|:------------------------:|:------------------------:|
| earnings22-keywords  | 9.32               | 9.74                | 9.68    | 10               | 9.9                 | 9.46                  |

<br/><br/>

## Precision

<details>
<summary>Click to expand</summary>

**What it measures:**  
Precision measures how *reliable* the predicted keywords are — i.e., the proportion of keywords predicted by the model that are actually correct. It focuses on avoiding *false positives* (incorrectly predicted keywords).

**How to interpret:**  
Higher values are better. A precision of 100% means every predicted keyword was indeed a correct one, while lower values indicate the model is predicting irrelevant or spurious keywords.

$\text{Precision} = \dfrac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$

**Example:**  
If the model predicts 20 keywords and 15 of them match the ground truth, precision = 15 / 20 = **75%** — meaning 1 in 4 predicted keywords were incorrect.


</details>

| Dataset              | Deepgram<br/>(nova-3) | OpenAI<br/>(whisper-1) | AssemblyAI | Whisper OSS<br/>(large-v3-turbo) | Argmax<br/>(Parakeet V2) | Argmax<br/>(Parakeet V3) |
|----------------------|:---------------------:|:----------------------:|:----------:|:----------------------:|:------------------------:|:------------------------:|
| earnings22-keywords  | 0.98               | 0.98                | 0.98    | 0.97                | 0.97                  | 0.97                  |

<br/><br/>

## Recall

<details>
<summary>Click to expand</summary>

**What it measures:**  
Recall measures how *complete* the model's keyword predictions are — i.e., the proportion of ground-truth keywords that the model successfully found. It focuses on avoiding *false negatives* (missed keywords).

**How to interpret:**  
Higher values are better. A recall of 100% means the model caught all the correct keywords, while lower recall indicates it missed some.

$\text{Recall} = \dfrac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$

**Example:**  
If the ground-truth transcript has 25 keywords and the model correctly finds 15, recall = 15 / 25 = **60%** — meaning the model missed 40% of the true keywords.


</details>

| Dataset              | Deepgram<br/>(nova-3) | OpenAI<br/>(whisper-1) | AssemblyAI | Whisper OSS<br/>(large-v3-turbo) | Argmax<br/>(Parakeet V2) | Argmax<br/>(Parakeet V3) |
|----------------------|:---------------------:|:----------------------:|:----------:|:----------------------:|:------------------------:|:------------------------:|
| earnings22-keywords  | 0.89               | 0.69                | 0.7    | 0.76                | 0.81                  | 0.8                  |

<br/><br/>

## F-score

<details>
<summary>Click to expand</summary>

**What it measures:**  
The F-Score (or F1-Score) combines **precision** and **recall** into a single metric that balances both correctness and completeness. It's the harmonic mean of precision and recall, so it penalizes models that do well on one but poorly on the other.

**How to interpret:**  
Higher values are better. A perfect F1 of 100% means the model predicted all and only the correct keywords.

$\text{F1} = 2 \times \dfrac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$

**Example:**  
If precision = 75% and recall = 60%,  
F1 = 2 × (0.75 × 0.6) / (0.75 + 0.6) = **66.7%**, reflecting the model's overall balance between accuracy and coverage.


</details>

| Dataset              | Deepgram<br/>(nova-3) | OpenAI<br/>(whisper-1) | AssemblyAI | Whisper OSS<br/>(large-v3-turbo) | Argmax<br/>(Parakeet V2) | Argmax<br/>(Parakeet V3) |
|----------------------|:---------------------:|:----------------------:|:----------:|:----------------------:|:------------------------:|:------------------------:|
| earnings22-keywords  | 0.93              | 0.81                | 0.82    | 0.85                | 0.88                  | 0.88                  |

<br/><br/>

