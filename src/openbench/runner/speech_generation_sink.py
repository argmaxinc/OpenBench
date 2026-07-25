# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""Incremental, append-only results sink for speech-generation benchmarks.

Buffers per-sample rows and, every `flush_every` samples, writes a *new*
parquet shard to a Hugging Face dataset repo (``data/chunk-NNNNN.parquet``).
Each flush only uploads a new file — existing shards are never rewritten — so
results accumulate safely and the HF dataset viewer auto-concatenates them.

Each row carries three playable Audio columns (embedded in the parquet) for
listening-based debugging — ``prompt_audio`` (the clone prompt),
``sim_reference_audio`` (the clip SIM compared the generation against), and
``generated_audio`` — plus ``prompt_text``, the ASR ``transcription``, and
per-sample ``SIM`` / ``WER``.
"""

import re
import tempfile
from collections.abc import Iterable
from pathlib import Path

from argmaxtools.utils import get_logger
from huggingface_hub import HfApi


logger = get_logger(__name__)

# Where shards live inside the repo. The HF datasets viewer auto-loads every
# parquet file under `data/`, concatenating them into a single table.
_DATA_DIR = "data"
_CHUNK_RE = re.compile(r"chunk-(\d+)(?:-[A-Za-z0-9._-]+)?\.parquet$")
_TAG_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")


class SpeechGenerationResultSink:
    """Accumulates per-sample rows and flushes parquet shards to an HF dataset.

    The chunk counter is resolved once at startup from the shards already in the
    repo, so several jobs writing to one repo concurrently would all pick the
    same next index and overwrite each other's uploads. Give each concurrent
    writer a distinct `chunk_tag` (e.g. its shard index) to keep filenames — and
    therefore uploads — disjoint.
    """

    def __init__(
        self,
        repo_id: str,
        flush_every: int = 100,
        private: bool = True,
        chunk_tag: str | None = None,
    ) -> None:
        self.repo_id = repo_id
        self.flush_every = max(1, int(flush_every))
        self.chunk_tag = _TAG_SANITIZE_RE.sub("-", chunk_tag).strip("-") if chunk_tag else None
        self._buffer: list[dict] = []
        self._api = HfApi()

        # Ensure the repo exists (idempotent) and resume the chunk counter past
        # any shards already present, so re-runs append instead of overwriting.
        self._api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=private)
        self._chunk_index = self._next_chunk_index()
        logger.info(
            f"Speech-generation results sink → hf://datasets/{repo_id} "
            f"(flush every {self.flush_every}, starting at chunk {self._chunk_index}"
            f"{f', tag {self.chunk_tag}' if self.chunk_tag else ''})"
        )

    def _next_chunk_index(self) -> int:
        try:
            files = self._api.list_repo_files(self.repo_id, repo_type="dataset")
        except Exception as e:  # noqa: BLE001 - fresh/empty repo or transient API error
            logger.warning(f"Could not list existing shards in {self.repo_id}: {e}")
            return 0
        indices = [int(m.group(1)) for f in files if (m := _CHUNK_RE.search(f))]
        return max(indices) + 1 if indices else 0

    def add(self, row: dict) -> None:
        """Buffer one per-sample row, flushing automatically at the threshold."""
        self._buffer.append(row)
        if len(self._buffer) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        """Write the buffered rows as a new parquet shard and upload it."""
        if not self._buffer:
            return

        # Imported lazily so importing the runner never pulls in the (heavy)
        # datasets Audio stack unless a sink is actually used.
        from datasets import Audio, Dataset, Features, Value

        features = Features(
            {
                "sample_idx": Value("string"),
                "language": Value("string"),
                # Three audio columns for listening-based debugging: the clone
                # prompt, the clip SIM compared against (held-out real target
                # when the dataset ships one; None when the pipeline recorded
                # none), and the generated clip.
                "prompt_audio": Audio(),
                "sim_reference_audio": Audio(),
                "generated_audio": Audio(),
                # Kept adjacent for easy analysis: synthesized text, its ASR
                # transcription, and the two scores.
                "prompt_text": Value("string"),
                "transcription": Value("string"),
                "WER": Value("float32"),
                "SIM": Value("float32"),
            }
        )
        rows, self._buffer = self._buffer, []
        dataset = Dataset.from_list(rows, features=features)

        tag_suffix = f"-{self.chunk_tag}" if self.chunk_tag else ""
        chunk_name = f"chunk-{self._chunk_index:05d}{tag_suffix}.parquet"
        with tempfile.TemporaryDirectory() as tmp:
            local_path = Path(tmp) / chunk_name
            # to_parquet embeds the Audio columns as bytes, so each shard is
            # self-contained (no external file references). batch_size=1 puts
            # every sample in its own row group, so a reader can range-request a
            # single sample's audio instead of pulling the whole shard — one
            # long reference clip is already tens of MB.
            dataset.to_parquet(str(local_path), batch_size=1)
            self._api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=f"{_DATA_DIR}/{chunk_name}",
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message=f"Add {len(rows)} samples (chunk {self._chunk_index})",
            )
        logger.info(f"Uploaded {len(rows)} samples → {self.repo_id}:{_DATA_DIR}/{chunk_name}")
        self._chunk_index += 1

    def close(self) -> None:
        """Flush any remaining buffered rows."""
        self.flush()


def completed_sample_ids(repo_ids: Iterable[str], column: str = "sample_idx") -> set[str]:
    """Collect the sample ids already scored in the given HF results repos.

    Only `column` is fetched from each parquet shard, so the (much larger)
    embedded audio columns are never downloaded — reading the ids of a few
    hundred results costs megabytes, not gigabytes. Repos that do not exist yet
    contribute nothing, which makes a first run behave like a full sweep.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    from huggingface_hub.utils import HfHubHTTPError

    fs = HfFileSystem()
    completed: set[str] = set()
    for repo_id in repo_ids:
        repo_id = repo_id.strip()
        if not repo_id:
            continue
        try:
            shards = fs.glob(f"datasets/{repo_id}/**/*.parquet")
        except (FileNotFoundError, HfHubHTTPError) as e:
            logger.info(f"No results to resume from in {repo_id}: {e}")
            continue

        found = 0
        for shard in shards:
            with fs.open(shard, "rb") as handle:
                table = pq.read_table(handle, columns=[column])
            ids = {str(v) for v in table.column(column).to_pylist() if v is not None}
            found += len(ids)
            completed |= ids
        logger.info(f"Found {found} scored samples across {len(shards)} shards in {repo_id}")

    logger.info(f"Resuming past {len(completed)} unique already-scored samples")
    return completed
