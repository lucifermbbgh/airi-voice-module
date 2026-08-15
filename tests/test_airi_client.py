"""
Tests for the AIRI WebSocket client.

These tests validate the client's message construction, protocol handshake,
and connection state management without requiring an actual AIRI server.

Protocol expectations are verified against moeru-ai/airi's plugin-protocol:
- input:text:voice data field is `transcription` (not `text`)
- every outgoing message carries metadata.source (ModuleIdentity) + metadata.event.id
- module:announced (for self) marks the client ready
"""

from __future__ import annotations

import asyncio
import json

import pytest

from src.airi.websocket_client import AIRIClient


class _FakeWebSocket:
    """Minimal stand-in for a websockets client protocol."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, raw: str) -> None:
        self.sent.append(raw)

    async def close(self) -> None:
        pass


def _connected_client(name: str = "voice-module") -> tuple[AIRIClient, _FakeWebSocket]:
    """Return a client whose socket is faked as connected."""
    client = AIRIClient(name=name)
    ws = _FakeWebSocket()
    client._ws = ws  # noqa: SLF001
    client._connected = True
    return client, ws


class TestAIRIClient:
    """Test AIRI WebSocket client."""

    def test_init_defaults(self):
        """Test default initialization."""
        client = AIRIClient()
        assert client.host == "localhost"
        assert client.port == 6121
        assert not client.is_connected
        assert client.url == "ws://localhost:6121/ws"

    def test_custom_params(self):
        """Test custom initialization."""
        client = AIRIClient(
            host="192.168.1.100",
            port=8080,
            path="/custom",
            token="abc123",
            reconnect_interval=10,
            max_attempts=3,
        )
        assert client.host == "192.168.1.100"
        assert client.port == 8080
        assert client.path == "/custom"
        assert client.token == "abc123"
        assert client.reconnect_interval == 10
        assert client.max_attempts == 3
        assert client.url == "ws://192.168.1.100:8080/custom"

    def test_event_handler_registration(self):
        """Test event handler registration."""
        client = AIRIClient()

        async def handler(data):
            pass

        client.on("input:text:voice", handler)
        assert "input:text:voice" in client._event_handlers
        assert len(client._event_handlers["input:text:voice"]) == 1

    def test_multiple_handlers(self):
        """Test multiple handlers for same event."""
        client = AIRIClient()

        def handler1(data):
            pass

        def handler2(data):
            pass

        client.on("input:text", handler1)
        client.on("input:text", handler2)
        assert len(client._event_handlers["input:text"]) == 2

    def test_send_when_disconnected(self):
        """Test send when not connected returns False."""
        client = AIRIClient()
        result = asyncio.run(client.send({"type": "test", "data": {}}))
        assert result is False

    def test_input_text_message(self):
        """Test input:text message construction."""
        msg = {
            "type": "input:text",
            "data": {
                "text": "Hello AIRI",
            },
        }
        assert msg["type"] == "input:text"
        assert msg["data"]["text"] == "Hello AIRI"

    def test_input_text_voice_message(self):
        """Test input:text:voice message construction."""
        msg = {
            "type": "input:text:voice",
            "data": {
                "transcription": "Hello via voice",
            },
        }
        assert msg["type"] == "input:text:voice"
        assert msg["data"]["transcription"] == "Hello via voice"

    # ── Protocol (verified against moeru-ai/airi) ──────────────

    def test_identity_structure(self):
        """ModuleIdentity matches the AIRI protocol shape."""
        client = AIRIClient(name="voice-module")
        ident = client._identity
        assert ident["kind"] == "plugin"
        assert ident["plugin"]["id"] == "voice-module"
        assert ident["id"].startswith("voice-module-")

    def test_send_attaches_metadata(self):
        """Every outgoing message carries metadata.source + metadata.event.id."""
        client, ws = _connected_client("test-module")
        assert asyncio.run(client.send({
            "type": "input:text:voice",
            "data": {"transcription": "hi"},
        })) is True

        payload = json.loads(ws.sent[0])
        assert payload["metadata"]["source"]["kind"] == "plugin"
        assert payload["metadata"]["source"]["plugin"]["id"] == "test-module"
        assert payload["metadata"]["source"]["id"] == client._instance_id
        assert isinstance(payload["metadata"]["event"]["id"], str)
        assert payload["metadata"]["event"]["id"]

    def test_send_input_text_voice_uses_transcription(self):
        """send_input_text_voice emits the `transcription` field, not `text`."""
        client, ws = _connected_client("test")
        assert asyncio.run(client.send_input_text_voice("你好")) is True
        payload = json.loads(ws.sent[0])
        assert payload["type"] == "input:text:voice"
        assert payload["data"]["transcription"] == "你好"
        assert "text" not in payload["data"]

    def test_parse_message_standard_json(self):
        """Standard JSON parses as-is."""
        msg = AIRIClient._parse_message('{"type": "x", "data": {"a": 1}}')
        assert msg["type"] == "x"
        assert msg["data"]["a"] == 1

    def test_parse_message_superjson_no_meta(self):
        """superjson envelope without meta (AIRI's actual wire format)."""
        msg = AIRIClient._parse_message(
            '{"json": {"type": "module:authenticated", "data": {"authenticated": true}}}'
        )
        assert msg["type"] == "module:authenticated"
        assert msg["data"]["authenticated"] is True

    def test_parse_message_superjson_with_meta(self):
        """superjson envelope with meta also unwraps to the inner json."""
        msg = AIRIClient._parse_message(
            '{"json": {"type": "x", "data": {}}, "meta": {"values": {}}}'
        )
        assert msg["type"] == "x"

    def test_announced_sets_ready(self):
        """module:announced for self marks the client ready."""
        client, ws = _connected_client("test-module")

        async def run():
            await client._handle_control_message(
                "module:announced",
                {"name": "test-module", "identity": {"id": client._instance_id}},
            )

        asyncio.run(run())
        assert client.is_ready is True

    def test_announced_ignores_other_modules(self):
        """module:announced for another module does not mark self ready."""
        client, ws = _connected_client("test-module")

        async def run():
            await client._handle_control_message(
                "module:announced",
                {"name": "other-module", "identity": {"id": "other-id"}},
            )

        asyncio.run(run())
        assert client.is_ready is False

    def test_authenticated_triggers_announce(self):
        """module:authenticated (success) triggers module:announce."""
        client, ws = _connected_client("test-module")

        async def run():
            await client._handle_control_message(
                "module:authenticated", {"authenticated": True}
            )

        asyncio.run(run())
        assert len(ws.sent) == 1
        sent = json.loads(ws.sent[0])
        assert sent["type"] == "module:announce"
        assert sent["data"]["name"] == "test-module"

    def test_announce_includes_identity(self):
        """module:announce payload includes name + identity."""
        client, ws = _connected_client("test-module")

        asyncio.run(client._announce())
        sent = json.loads(ws.sent[0])
        assert sent["type"] == "module:announce"
        assert sent["data"]["name"] == "test-module"
        assert sent["data"]["identity"]["id"] == client._instance_id
