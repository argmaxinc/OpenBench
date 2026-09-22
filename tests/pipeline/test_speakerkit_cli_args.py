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
