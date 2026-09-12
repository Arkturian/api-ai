"""Regression coverage for narration duration metadata."""

import asyncio

from ai.services.narration_service import (
    NarrationCharacter,
    NarrationConfig,
    NarrationRequest,
    NarrationService,
)


def test_generate_returns_measured_audio_duration(monkeypatch):
    service = NarrationService()

    async def fake_generate_tts(_text, _request):
        # seit with_timestamps (13.09.): (audio_bytes, word_timestamps|None)
        return b"generated-audio", None

    async def fake_save_audio(_audio_bytes, _request):
        return 116819, "https://api-storage.arkturian.com/storage/media/116819"

    monkeypatch.setattr(service, "_generate_tts", fake_generate_tts)
    monkeypatch.setattr(service, "_save_audio", fake_save_audio)
    monkeypatch.setattr(service, "_measure_audio_duration", lambda *_args: 6.583)

    request = NarrationRequest(
        text="Eine kurze Probe.",
        character=NarrationCharacter(
            name="dr_tschauko",
            voice_id="llAvE3WPPWbZOjaYywej",
        ),
        config=NarrationConfig(preprocessing=False),
        save_options={"is_public": True},
        collection_id="tts-audio",
    )

    result = asyncio.run(service.generate(request))

    assert result.audio_id == 116819
    assert result.duration_seconds == 6.583
