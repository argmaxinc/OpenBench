# OpenBench Benchmarks

This document contains the benchmark results for OpenBench, organized by task type.

## Speaker Diarization Benchmarks

These are the results reported in the `SDBench` paper. For more details please refer to the [paper](https://arxiv.org/abs/2507.16136).

> **Note:** Each cell in the table below shows `DER`, where **DER** is the Diarization Error Rate (lower is better). Cells with `-` indicate that the system/dataset combination was not evaluated.

| Dataset               |   AWS Transcribe (v20250217) | Deepgram (v20250627)  | Picovoice   |   Pyannote |   Pyannote-AI (v20250217) |   SpeakerKit |
|-----------------------|------------------------------|------------|-------------|------------|----------------------------|--------------|
| AISHELL-4             |           0.2247 | 0.7169     | -           |     0.1219 |        0.1119 |       0.1267 |
| AMI-IHM               |           0.2865 | 0.3538     | 0.3538      |     0.1891 |        0.1578 |       0.2064 |
| AMI-SDM               |           0.3697 | 0.4247     | -           |     0.2295 |        0.1830 |       0.2370 |
| AVA-AVD               |           0.6148 | 0.6816     | -           |     0.4845 |        0.4654 |       0.5188 |
| AliMeeting            |           0.4232 | 0.8068     | -           |     0.2514 |        0.1918 |       0.2648 |
| American-Life-Podcast |           0.2347 | 0.2888     | -           |     0.2880  |        0.2928 |       0.3686 |
| CallHome              |           0.3677 | 0.6385     | 0.5421      |     0.2877 |        0.1976 |       0.3100 |
| DIHARD-III            |           0.3550 | 0.3685     | -           |     0.2363 |        0.1706 |       0.2362 |
| EGO4D                 |           0.6088 | 0.7072     | -           |     0.5171 |        0.4579 |       0.5401 |
| Earnings-21           |           0.1776 | -          | -           |     0.0954 |        0.0928 |       0.0948 |
| ICSI                  |           0.4598 | -          | -           |     0.3443 |        0.3135 |       0.3506 |
| MSDWILD               |           0.4015 | 0.6449     | -           |     0.3171 |        0.2617 |       0.3519 |
| VoxConverse           |           0.1301 | 0.3642     | -           |     0.1104 |        0.0982 |       0.1205 |
