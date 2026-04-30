# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""Tests for parse_pipeline_config_overrides.

The helper turns the raw `--pipeline-config key=value` strings collected
by typer into a dict that's later merged into a pipeline alias's default
config and passed to PipelineRegistry.create_pipeline. Values stay as
strings — Pydantic does the type coercion when the config model is
instantiated.

These tests exercise the parser surface and verify that downstream
Pydantic coercion handles ints, floats, bools, and string values for a
representative speech-generation config.
"""

import unittest

import typer
from pydantic import BaseModel

from openbench.cli.command_utils import parse_pipeline_config_overrides
from openbench.pipeline.speech_generation.common import SpeechGenerationConfig


def _build_config(**overrides) -> SpeechGenerationConfig:
    """Build a SpeechGenerationConfig with required fields filled in."""
    base = {"cli_path": "/fake/whisperkit-cli"}
    base.update(overrides)
    return SpeechGenerationConfig(**base)


class TestParsePipelineConfigOverrides(unittest.TestCase):
    """Surface-level parsing tests."""

    def test_returns_empty_dict_when_none(self) -> None:
        self.assertEqual(parse_pipeline_config_overrides(None), {})

    def test_returns_empty_dict_when_empty_list(self) -> None:
        self.assertEqual(parse_pipeline_config_overrides([]), {})

    def test_single_string_pair(self) -> None:
        self.assertEqual(
            parse_pipeline_config_overrides(["speaker=serena"]),
            {"speaker": "serena"},
        )

    def test_multiple_pairs(self) -> None:
        self.assertEqual(
            parse_pipeline_config_overrides(["speaker=serena", "seed=42", "temperature=0.7"]),
            {"speaker": "serena", "seed": "42", "temperature": "0.7"},
        )

    def test_value_with_equals_sign_is_preserved(self) -> None:
        # Splits on the first '=' only, so the value can itself contain '='.
        self.assertEqual(
            parse_pipeline_config_overrides(["tokenizer=org/repo=v2"]),
            {"tokenizer": "org/repo=v2"},
        )

    def test_value_can_be_empty_string(self) -> None:
        self.assertEqual(parse_pipeline_config_overrides(["language="]), {"language": ""})

    def test_key_is_stripped_of_surrounding_whitespace(self) -> None:
        self.assertEqual(parse_pipeline_config_overrides(["  seed =42"]), {"seed": "42"})

    def test_missing_equals_raises_bad_parameter(self) -> None:
        with self.assertRaises(typer.BadParameter) as ctx:
            parse_pipeline_config_overrides(["seed42"])
        self.assertIn("Expected key=value", str(ctx.exception))

    def test_empty_key_raises_bad_parameter(self) -> None:
        with self.assertRaises(typer.BadParameter) as ctx:
            parse_pipeline_config_overrides(["=42"])
        self.assertIn("Empty key", str(ctx.exception))

    def test_later_value_overrides_earlier_value_for_same_key(self) -> None:
        self.assertEqual(parse_pipeline_config_overrides(["seed=1", "seed=2"]), {"seed": "2"})


class TestPydanticCoercionForOverrides(unittest.TestCase):
    """Verify the parser's string output coerces correctly when Pydantic builds the config.

    These guard the `--pipeline-config` user contract advertised in the
    flag's help text: ints, floats, strings, and bools all "just work".
    """

    def test_int_string_coerces_to_int(self) -> None:
        overrides = parse_pipeline_config_overrides(["seed=42"])
        config = _build_config(**overrides)
        self.assertEqual(config.seed, 42)
        self.assertIsInstance(config.seed, int)

    def test_float_string_coerces_to_float(self) -> None:
        overrides = parse_pipeline_config_overrides(["temperature=0.7"])
        config = _build_config(**overrides)
        self.assertAlmostEqual(config.temperature, 0.7)
        self.assertIsInstance(config.temperature, float)

    def test_string_value_stays_string(self) -> None:
        overrides = parse_pipeline_config_overrides(["speaker=serena"])
        config = _build_config(**overrides)
        self.assertEqual(config.speaker, "serena")

    def test_bool_string_coerces_to_bool_on_pydantic_field(self) -> None:
        """SpeechGenerationConfig has no native bool field; use a minimal model."""

        class _M(BaseModel):
            flag: bool

        cases = [
            ("true", True),
            ("True", True),
            ("false", False),
            ("False", False),
            ("1", True),
            ("0", False),
        ]
        for raw_value, expected in cases:
            with self.subTest(raw_value=raw_value):
                overrides = parse_pipeline_config_overrides([f"flag={raw_value}"])
                self.assertIs(_M(**overrides).flag, expected)


if __name__ == "__main__":
    unittest.main()
