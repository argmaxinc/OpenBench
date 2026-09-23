from pathlib import Path

from openbench.pipeline.diarization.speakerkit import SpeakerKitPipelineConfig


def _config(**overrides) -> SpeakerKitPipelineConfig:
    kwargs = {"cli_path": "/opt/speakerkitpro-cli", "out_dir": "./out"}
    kwargs.update(overrides)
    return SpeakerKitPipelineConfig(**kwargs)


def _inputs(num_speakers=None) -> dict:
    return {"audio_path": Path("a.flac"), "output_path": Path("a.rttm"), "num_speakers": num_speakers}


def test_cli_args_use_diarizer_flag_for_pyannote(monkeypatch):
    monkeypatch.delenv("SPEAKERKIT_API_KEY", raising=False)
    cmd = _config(engine="pyannote").generate_cli_args(_inputs())
    assert cmd[:2] == ["/opt/speakerkitpro-cli", "diarize"]
    assert "--diarizer" in cmd and cmd[cmd.index("--diarizer") + 1] == "pyannote"
    assert "--engine" not in cmd
    assert "--num-speakers" not in cmd and "--api-key" not in cmd


def test_cli_args_sortformer_num_speakers_and_api_key(monkeypatch):
    monkeypatch.setenv("SPEAKERKIT_API_KEY", "secret")
    cmd = _config(engine="sortformer", model_path="/models").generate_cli_args(_inputs(num_speakers=3))
    assert cmd[cmd.index("--diarizer") + 1] == "sortformer"
    assert cmd[cmd.index("--model-path") + 1] == "/models"
    assert cmd[cmd.index("--num-speakers") + 1] == "3"
    assert cmd[cmd.index("--api-key") + 1] == "secret"


def test_cli_args_sortformer_model_version_and_variant(monkeypatch):
    monkeypatch.delenv("SPEAKERKIT_API_KEY", raising=False)
    cmd = _config(
        engine="sortformer",
        sortformer_model_version="nemotron-3-diarization",
        sortformer_model_variant="684_74MB",
    ).generate_cli_args(_inputs())
    assert cmd[cmd.index("--sortformer-model-version") + 1] == "nemotron-3-diarization"
    assert cmd[cmd.index("--sortformer-model-variant") + 1] == "684_74MB"


def test_cli_args_sortformer_model_flags_omitted_when_unset(monkeypatch):
    monkeypatch.delenv("SPEAKERKIT_API_KEY", raising=False)
    cmd = _config(engine="sortformer").generate_cli_args(_inputs())
    assert "--sortformer-model-version" not in cmd
    assert "--sortformer-model-variant" not in cmd


def test_cli_args_sortformer_model_flags_ignored_for_pyannote(monkeypatch):
    monkeypatch.delenv("SPEAKERKIT_API_KEY", raising=False)
    cmd = _config(
        engine="pyannote", sortformer_model_version="v2-1", sortformer_model_variant="384_94MB"
    ).generate_cli_args(_inputs())
    assert "--sortformer-model-version" not in cmd
    assert "--sortformer-model-variant" not in cmd


def _alias_cli_args(alias: str, monkeypatch) -> list[str]:
    import openbench.pipeline  # noqa: F401 - importing the package registers the aliases
    from openbench.pipeline.pipeline_registry import PipelineRegistry

    monkeypatch.delenv("SPEAKERKIT_API_KEY", raising=False)
    config = dict(PipelineRegistry.get_alias_info(alias).default_config)
    config["cli_path"] = "/opt/speakerkitpro-cli"
    return SpeakerKitPipelineConfig(**config).generate_cli_args(_inputs())


def test_sortformer_compressed_alias_pins_v2_model(monkeypatch):
    cmd = _alias_cli_args("speakerkit-sortformer-compressed", monkeypatch)
    assert cmd[cmd.index("--diarizer") + 1] == "sortformer"
    assert cmd[cmd.index("--sortformer-model-version") + 1] == "v2-1"
    assert cmd[cmd.index("--sortformer-model-variant") + 1] == "384_94MB"


def test_nemotron_3_diarization_alias_pins_v3_model(monkeypatch):
    cmd = _alias_cli_args("speakerkit-nemotron-3-diarization", monkeypatch)
    assert cmd[cmd.index("--diarizer") + 1] == "sortformer"
    assert cmd[cmd.index("--sortformer-model-version") + 1] == "nemotron-3-diarization"
    assert cmd[cmd.index("--sortformer-model-variant") + 1] == "684_74MB"
