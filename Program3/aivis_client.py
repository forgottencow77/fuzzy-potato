"""
Shared AivisSpeech client used for text-to-speech synthesis.

This module consolidates the duplicated client implementations so that
multiple entrypoints (batch scripts, realtime assistants, web gateways)
can generate speech through the same code path.
"""

from __future__ import annotations

import requests


class AivisClient:
    """Minimal VOICEVOX-compatible text-to-speech client."""

    def __init__(self, base: str, speaker_id: int, timeout_query: int = 30, timeout_synth: int = 120):
        self.base = base.rstrip("/")
        self.speaker_id = speaker_id
        self.timeout_query = timeout_query
        self.timeout_synth = timeout_synth

    def initialize_speaker(self) -> None:
        """Best-effort speaker initialization to avoid cold-start latency."""
        try:
            res = requests.post(
                f"{self.base}/initialize_speaker",
                params={"speaker": self.speaker_id},
                timeout=60,
            )
            if res.status_code >= 400:
                print(f"[AIVIS] initialize_speaker {res.status_code}: {res.text[:200]}")
        except Exception as exc:  # noqa: BLE001 - log and continue
            print(f"[AIVIS] initialize_speaker skipped: {exc}")

    def build_query(self, text: str) -> dict:
        """
        Construct audio synthesis query payload.

        The API expects text within the query parameters. If the server
        rejects the schema with 422, fall back to the VOICEVOX-compatible
        text/plain body.
        """
        response = requests.post(
            f"{self.base}/audio_query",
            params={"speaker": self.speaker_id, "text": text},
            timeout=self.timeout_query,
        )
        if response.status_code == 422:
            response = requests.post(
                f"{self.base}/audio_query",
                params={"speaker": self.speaker_id},
                data=text.encode("utf-8"),
                headers={"Content-Type": "text/plain; charset=utf-8"},
                timeout=self.timeout_query,
            )
        response.raise_for_status()
        return response.json()

    def synthesize(self, query_json: dict) -> bytes:
        response = requests.post(
            f"{self.base}/synthesis",
            params={"speaker": self.speaker_id},
            json=query_json,
            timeout=self.timeout_synth,
        )
        response.raise_for_status()
        return response.content

    def tts(self, text: str) -> bytes:
        """Synthesize speech audio for the supplied text."""
        self.initialize_speaker()
        query = self.build_query(text)
        return self.synthesize(query)


__all__ = ["AivisClient"]
