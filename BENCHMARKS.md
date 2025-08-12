# OpenBench Benchmarks

This document contains the benchmark results for OpenBench, organized by task type.

# Speaker Diarization 

> **Note:** If a cell in the tables below is `-`, it means that the system/dataset combination was not evaluated due to timeout constaints or lack of credits.

## Diarization Error Rate (DER)

**What it measures:** DER quantifies how accurately a system identifies "who spoke when" in an audio recording. It measures the total time that speakers are incorrectly labeled, including missed speech, falsely detected speech, and speaker confusion.

**How to interpret:** Lower values are better. A DER of 0.0 would be perfect (no errors), while 1.0 means 100% error. A DER of 0.20 means 20% of the audio time has speaker labeling errors.

**Example:** In a 10-minute conversation, a DER of 0.15 means that for 1.5 minutes total, the system either missed speech, detected non-existent speech, or confused which speaker was talking.


| Dataset                | AWS Transcribe (v20250217) | Deepgram (v20250627) | Picovoice | Pyannote | Pyannote-AI (v20250217) | SpeakerKit |
|------------------------|---------------------------|----------------------|-----------|----------|-------------------------|------------|
| AISHELL-4              | 0.22                      | 0.72                 | -         | 0.12     | 0.11                    | 0.13       |
| AMI-IHM                | 0.29                      | 0.35                 | 0.35      | 0.19     | 0.16                    | 0.21       |
| AMI-SDM                | 0.37                      | 0.42                 | -         | 0.23     | 0.18                    | 0.24       |
| AVA-AVD                | 0.61                      | 0.68                 | -         | 0.48     | 0.47                    | 0.52       |
| AliMeeting             | 0.42                      | 0.81                 | -         | 0.25     | 0.19                    | 0.26       |
| American-Life-Podcast  | 0.23                      | 0.29                 | -         | 0.29     | 0.29                    | 0.37       |
| CallHome               | 0.37                      | 0.64                 | 0.54      | 0.29     | 0.20                    | 0.31       |
| DIHARD-III             | 0.36                      | 0.37                 | -         | 0.24     | 0.17                    | 0.24       |
| EGO4D                  | 0.61                      | 0.71                 | -         | 0.52     | 0.46                    | 0.54       |
| Earnings-21            | 0.18                      | -                    | -         | 0.10     | 0.09                    | 0.09       |
| ICSI                   | 0.46                      | -                    | -         | 0.34     | 0.31                    | 0.35       |
| MSDWILD                | 0.40                      | 0.64                 | -         | 0.32     | 0.26                    | 0.35       |
| VoxConverse            | 0.13                      | 0.36                 | -         | 0.11     | 0.10                    | 0.12       |


## Speed Factor (SF)

**What it measures:** Speed Factor compares how much faster (or slower) a system processes audio compared to real-time. It's calculated as $SF = \dfrac{Duration_{audio}}{Duration_{prediction}}$.

**How to interpret:** Values above 1x mean the system is faster than real-time. Values below 1x mean slower than real-time. Higher values indicate faster processing.

**Example:** An SF of 10x means the system processes 10 seconds of audio in 1 second. An SF of 0.5x means it takes 2 seconds to process 1 second of audio.

| Dataset                 | AWS Transcribe (v20250217) | Deepgram (v20250627) | Picovoice | Pyannote | Pyannote-AI (v20250217) | SpeakerKit |
|-------------------------|---------------------------|----------------------|-----------|----------|-------------------------|------------|
| AISHELL-4               | 10x                       | 130x                 | -         | 55x      | 62x                     | 476x       |
| AMI-IHM                 | 11x                       | 216x                 | 59x       | 53x      | 45x                     | 463x       |
| AMI-SDM                 | 10x                       | 241x                 | -         | 54x      | 62x                     | 458x       |
| AVA-AVD                 | 3x                        | 187x                 | -         | 28x      | 35x                     | 426x       |
| AliMeeting              | 9x                        | 157x                 | -         | 29x      | 45x                     | 442x       |
| American-Life-Podcast   | 10x                       | 231x                 | -         | 54x      | 58x                     | 481x       |
| CallHome                | 2x                        | 63x                  | 61x       | 53x      | 20x                     | 263x       |
| DIHARD-III              | 8x                        | 154x                 | -         | 28x      | 39x                     | 433x       |
| EGO4D                   | 6x                        | 127x                 | -         | 54x      | 34x                     | 436x       |
| Earnings-21             | 9x                        | -                    | -         | 54x      | 47x                     | 496x       |
| ICSI                    | 11x                       | -                    | -         | 52x      | 62x                     | 447x       |
| MSDWILD                 | 1x                        | 43x                  | -         | 53x      | 15x                     | 216x       |
| VoxConverse             | 6x                        | 210x                 | -         | 53x      | 50x                     | 462x       |



## Speaker Count Accuracy (SCA)

**What it measures:** SCA measures how accurately a system identifies the total number of unique speakers in an audio recording, regardless of when they spoke.

**How to interpret:** Expressed as a percentage, where 100% means perfect speaker count detection. Lower percentages indicate the system overestimated or underestimated the number of speakers.

**Example:** If there are 4 speakers in a recording and the system detects 3 speakers, the SCA would be 75% (3 correct out of 4 total speakers).

| Dataset                 | AWS Transcribe (v20250217) | Deepgram (v20250627) | Picovoice | Pyannote | Pyannote-AI (v20250217) | SpeakerKit |
|-------------------------|---------------------------|----------------------|-----------|----------|-------------------------|------------|
| AISHELL-4               | 75%                       | 30%                  | -         | 5%       | 15%                     | 5%         |
| AMI-IHM                 | 94%                       | 56%                  | 12%       | 0%       | 12%                     | 0%         |
| AMI-SDM                 | 56%                       | 88%                  | -         | 6%       | 12%                     | 0%         |
| AVA-AVD                 | 13%                       | 6%                   | -         | 13%      | 9%                      | 11%        |
| AliMeeting              | 90%                       | 5%                   | -         | 40%      | 55%                     | 10%        |
| American-Life-Podcast   | 11%                       | 14%                  | -         | 8%       | 8%                      | 8%         |
| CallHome                | 60%                       | 33%                  | 15%       | 74%      | 48%                     | 48%        |
| DIHARD-III              | 72%                       | 60%                  | -         | 60%      | 58%                     | 25%        |
| EGO4D                   | 34%                       | 16%                  | -         | 24%      | 24%                     | 32%        |
| Earnings-21             | 50%                       | -                    | -         | 50%      | 64%                     | 9%         |
| ICSI                    | 43%                       | -                    | -         | 7%       | 13%                     | 7%         |
| MSDWILD                 | 39%                       | 15%                  | -         | 34%      | 35%                     | 26%        |
| VoxConverse             | 46%                       | 39%                  | -         | 42%      | 38%                     | 23%        |
