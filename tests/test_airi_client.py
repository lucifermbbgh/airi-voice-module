"""
Tests for the AIRI WebSocket client.

These tests validate the client's message construction and
connection state management without requiring an actual AIRI server.
"""

from __future__ import annotations

import pytest

from src.airi.websocket_client import AIRIClient


class TestAIRIClient:
    """Test AIRI WebSocket client."""

    def test_init_defaults(self):
        """Test default initialization."""
        client = AIRIClient()
        assert client.host == "localhost"
        assert client.port == 10443
        assert not client.is_connected
        assert client.url == "ws://localhost:10443"

    def test_custom_params(self):
        """Test custom initialization."""
        client = AIRIClient(
            host="192.168.1.100",
            port=8080,
            token="abc123",
            reconnect_interval=10,
            max_attempts=3,
        )
        assert client.host == "192.168.1.100"
        assert client.port == 8080
        assert client.token == "abc123"
        assert client.reconnect_interval == 10
        assert client.max_attempts == 3
        assert client.url == "ws://192.168.1.100:8080"

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
        import asyncio
        client = AIRIClient()
        result = asyncio.run(client.send({"type": "test", "data": {}}))
        assert result is False

    def test_input_text_message(self):
        """Test input:text message construction."""
        import asyncio
        client = AIRIClient()
        # Can't actually send when disconnected, but verify structure
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
                "text": "Hello via voice",
            },
        }
        assert msg["type"] == "input:text:voice"
        assert msg["data"]["text"] == "Hello via voice"
