from typing import Any

import texterrors
from argmaxtools.utils import logger
from pyannote.metrics.base import BaseMetric

from ...pipeline_prediction import Transcript
from ...types import PipelineType
from ..metric import MetricOptions
from ..registry import MetricRegistry


class BaseKeywordMetric(BaseMetric):
    """Base class for keyword boosting metrics."""

    def __init__(self):
        """Initialize keyword metric."""
        super().__init__()
        # Use Whisper's BasicTextNormalizer
        from transformers.models.whisper.english_normalizer import BasicTextNormalizer

        self.text_normalizer = BasicTextNormalizer()

    def compute_keyword_stats(
        self, reference: Transcript, hypothesis: Transcript, dictionary: list[str]
    ) -> dict[str, Any]:
        """Compute keyword statistics between reference and hypothesis."""

        # Convert transcripts to text
        ref_text = reference.get_transcript_string()
        hyp_text = hypothesis.get_transcript_string()

        logger.debug(f"Reference text: '{ref_text}'")
        logger.debug(f"Hypothesis text: '{hyp_text}'")
        logger.debug(f"Keywords: {dictionary}")

        # Apply normalization to BOTH reference and hypothesis
        ref_text = self.text_normalizer(ref_text)
        hyp_text = self.text_normalizer(hyp_text)

        # Normalize keywords as well
        normalized_keywords = [self.text_normalizer(kw) for kw in dictionary]

        logger.debug(f"Normalized Reference: '{ref_text}'")
        logger.debug(f"Normalized Hypothesis: '{hyp_text}'")
        logger.debug(f"Normalized Keywords: {normalized_keywords}")

        # Get alignment using texterrors
        ref_words = ref_text.split()
        hyp_words = hyp_text.split()
        # Word-level alignment between reference and hypothesis text
        # returns two aligned sequences where insertions/deletions are
        # marked with <eps> tokens. This alignment is crucial for accurate keyword matching
        # as it ensures we compare the right words even when there are transcription errors.
        texterrors_ali = texterrors.align_texts(ref_words, hyp_words, False)

        # Create alignment pairs
        ali = []
        for i in range(len(texterrors_ali[0])):
            ali.append((texterrors_ali[0][i], texterrors_ali[1][i]))

        # Compute max ngram order
        max_ngram_order = max([len(item.split()) for item in normalized_keywords])
        key_words_stat = {}
        for word in normalized_keywords:
            key_words_stat[word] = [0, 0, 0]  # [tp, gt, fp]

        eps = "<eps>"

        # 1-grams
        for idx in range(len(ali)):
            word_ref = ali[idx][0]
            word_hyp = ali[idx][1]
            if word_ref in key_words_stat:
                key_words_stat[word_ref][1] += 1  # add to gt
                if word_ref == word_hyp:
                    key_words_stat[word_ref][0] += 1  # add to tp
            elif word_hyp in key_words_stat:
                key_words_stat[word_hyp][2] += 1  # add to fp

        # 2-grams and higher
        for ngram_order in range(2, max_ngram_order + 1):
            # For reference phrase
            idx = 0
            item_ref = []
            while idx < len(ali):
                if item_ref:
                    item_ref = [item_ref[1]]
                    idx = item_ref[0][1] + 1
                while len(item_ref) != ngram_order and idx < len(ali):
                    word = ali[idx][0]
                    idx += 1
                    if word == eps:
                        continue
                    else:
                        item_ref.append((word, idx - 1))
                if len(item_ref) == ngram_order:
                    phrase_ref = " ".join([item[0] for item in item_ref])
                    phrase_hyp = " ".join([ali[item[1]][1] for item in item_ref])
                    if phrase_ref in key_words_stat:
                        key_words_stat[phrase_ref][1] += 1  # add to gt
                        if phrase_ref == phrase_hyp:
                            key_words_stat[phrase_ref][0] += 1  # add to tp

            # For false positive hypothesis phrase
            idx = 0
            item_hyp = []
            while idx < len(ali):
                if item_hyp:
                    item_hyp = [item_hyp[1]]
                    idx = item_hyp[0][1] + 1
                while len(item_hyp) != ngram_order and idx < len(ali):
                    word = ali[idx][1]
                    idx += 1
                    if word == eps:
                        continue
                    else:
                        item_hyp.append((word, idx - 1))
                if len(item_hyp) == ngram_order:
                    phrase_hyp = " ".join([item[0] for item in item_hyp])
                    phrase_ref = " ".join([ali[item[1]][0] for item in item_hyp])
                    if phrase_hyp in key_words_stat and phrase_hyp != phrase_ref:
                        key_words_stat[phrase_hyp][2] += 1  # add to fp

        # Compute totals
        tp = sum([key_words_stat[x][0] for x in key_words_stat])
        gt = sum([key_words_stat[x][1] for x in key_words_stat])
        fp = sum([key_words_stat[x][2] for x in key_words_stat])

        logger.debug(f"TP: {tp}, FP: {fp}, FN: {gt - tp}")

        # Print detailed keyword statistics
        fp_keywords = []
        for i, keyword in enumerate(normalized_keywords, 1):
            if keyword in key_words_stat:
                kw_tp = key_words_stat[keyword][0]
                kw_gt = key_words_stat[keyword][1]
                kw_fp = key_words_stat[keyword][2]

                if kw_gt > 0:  # Only print for keywords that exist in ground truth
                    if kw_tp > 0:
                        status = f"TP ({kw_tp}/{kw_gt})"
                    else:
                        status = f"FN (0/{kw_gt})"
                    logger.debug(f"{keyword} {i}/{len(normalized_keywords)} - {status}")

                # Collect FP keywords
                if kw_fp > 0:
                    fp_keywords.append(f"{keyword} ({kw_fp})")

        # Print FP keywords separately
        if fp_keywords:
            logger.debug(f"FP keywords: {fp_keywords} - FP rate: {fp}/{len(normalized_keywords)}")

        logger.debug("---")
        logger.info(f"Keyword statistics computed: TP={tp}, FP={fp}, FN={gt - tp}, GT={gt}")

        return {"true_positives": tp, "ground_truth": gt, "false_positives": fp, "keyword_stats": key_words_stat}


@MetricRegistry.register_metric(PipelineType.TRANSCRIPTION, MetricOptions.KEYWORD_FSCORE)
class KeywordFScore(BaseKeywordMetric):
    """Keyword F-Score metric for boosting transcription evaluation."""

    @classmethod
    def metric_name(cls) -> str:
        return "keyword_fscore"

    @classmethod
    def metric_components(cls) -> list[str]:
        return ["true_positives", "ground_truth", "false_positives"]

    def compute_components(
        self, reference: Transcript, hypothesis: Transcript, dictionary: list[str]
    ) -> dict[str, int]:
        """Compute keyword F-score components."""
        logger.debug("Computing KeywordFScore components")
        stats = self.compute_keyword_stats(reference, hypothesis, dictionary)
        return {
            "true_positives": stats["true_positives"],
            "ground_truth": stats["ground_truth"],
            "false_positives": stats["false_positives"],
        }

    def compute_metric(self, detail: dict[str, int]) -> float:
        """Compute F-score from components."""
        tp = detail["true_positives"]
        gt = detail["ground_truth"]
        fp = detail["false_positives"]

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (gt + 1e-8)
        fscore = 2 * (precision * recall) / (precision + recall + 1e-8)

        logger.debug(f"F-score computed: precision={precision:.4f}, recall={recall:.4f}, f_score={fscore:.4f}")
        return fscore


@MetricRegistry.register_metric(PipelineType.TRANSCRIPTION, MetricOptions.KEYWORD_PRECISION)
class KeywordPrecision(BaseKeywordMetric):
    """Keyword Precision metric for boosting transcription evaluation."""

    @classmethod
    def metric_name(cls) -> str:
        return "keyword_precision"

    @classmethod
    def metric_components(cls) -> list[str]:
        return ["true_positives", "false_positives"]

    def compute_components(
        self, reference: Transcript, hypothesis: Transcript, dictionary: list[str]
    ) -> dict[str, int]:
        """Compute keyword precision components."""
        logger.debug("Computing KeywordPrecision components")
        stats = self.compute_keyword_stats(reference, hypothesis, dictionary)
        return {"true_positives": stats["true_positives"], "false_positives": stats["false_positives"]}

    def compute_metric(self, detail: dict[str, int]) -> float:
        """Compute precision from components."""
        tp = detail["true_positives"]
        fp = detail["false_positives"]

        logger.debug(f"Computing Precision: TP={tp}, FP={fp}")
        precision = tp / (tp + fp + 1e-8)
        logger.debug(f"Precision computed: {precision:.4f}")
        return precision


@MetricRegistry.register_metric(PipelineType.TRANSCRIPTION, MetricOptions.KEYWORD_RECALL)
class KeywordRecall(BaseKeywordMetric):
    """Keyword Recall metric for boosting transcription evaluation."""

    @classmethod
    def metric_name(cls) -> str:
        return "keyword_recall"

    @classmethod
    def metric_components(cls) -> list[str]:
        return ["true_positives", "ground_truth"]

    def compute_components(
        self, reference: Transcript, hypothesis: Transcript, dictionary: list[str]
    ) -> dict[str, int]:
        """Compute keyword recall components."""
        logger.debug("Computing KeywordRecall components")
        stats = self.compute_keyword_stats(reference, hypothesis, dictionary)
        return {"true_positives": stats["true_positives"], "ground_truth": stats["ground_truth"]}

    def compute_metric(self, detail: dict[str, int]) -> float:
        """Compute recall from components."""
        tp = detail["true_positives"]
        gt = detail["ground_truth"]

        logger.debug(f"Computing Recall: TP={tp}, GT={gt}")
        recall = tp / (gt + 1e-8)
        logger.debug(f"Recall computed: {recall:.4f}")
        return recall
