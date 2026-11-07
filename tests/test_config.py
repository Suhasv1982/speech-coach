import imageio_ffmpeg

import config


def test_ffmpeg_path_setting_wins(monkeypatch):
    monkeypatch.setenv("FFMPEG_PATH", "/custom/ffmpeg")
    assert config.ffmpeg_exe() == "/custom/ffmpeg"


def test_ffmpeg_on_path_is_used_next(monkeypatch):
    monkeypatch.delenv("FFMPEG_PATH", raising=False)
    monkeypatch.setattr(config.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    assert config.ffmpeg_exe() == "/usr/bin/ffmpeg"


def test_bundled_ffmpeg_is_the_fallback(monkeypatch):
    monkeypatch.delenv("FFMPEG_PATH", raising=False)
    monkeypatch.setattr(config.shutil, "which", lambda name: None)
    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", lambda: "/bundled/ffmpeg")
    assert config.ffmpeg_exe() == "/bundled/ffmpeg"
