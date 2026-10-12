"""The LLM prior through the scripted provider, and the HTTP adapter against a local stub."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar

import numpy as np
import pytest

from finqa_search.context import QuestionContext
from finqa_search.llm import (
    LLMPrior,
    OpenAICompatibleProvider,
    ScriptedProvider,
    build_messages,
    parse_proposal,
    scripted_from_steps,
)
from finqa_search.prior import UniformPrior
from finqa_search.space import STOP_ACTION, ActionSpace


def test_proposals_map_onto_legal_actions(space: ActionSpace) -> None:
    root = space.root()
    act = parse_proposal(space, root, "Next step: subtract(5829, 5735)")
    assert act is not None
    assert space.step(root, act).render() == "subtract(5829, 5735)"
    assert parse_proposal(space, root, "subtract(1, 2)") is None  # 1 and 2 are not leaves here
    assert parse_proposal(space, root, "stop") is None  # nothing to stop on yet
    assert parse_proposal(space, root, "I am not sure") is None
    child = space.child(root, act)
    assert parse_proposal(space, child, "stop") == STOP_ACTION
    nxt = parse_proposal(space, child, "divide(#0, 5735)")
    assert nxt is not None
    assert space.step(child, nxt).render() == "divide(#0, 5735)"


def test_llm_prior_mixes_votes_into_the_base_prior(space: ActionSpace) -> None:
    provider = scripted_from_steps(["subtract(5829, 5735)"])
    prior = LLMPrior(provider, UniformPrior(), samples=4, weight=0.5)
    root = space.root()
    batch = space.legal(root)
    (p,) = prior(space, [root], [batch])
    assert p.sum() == pytest.approx(1.0)
    best = batch.action(int(np.argmax(p)))
    assert space.step(root, best).render() == "subtract(5829, 5735)"
    assert p.max() == pytest.approx(0.5 + 0.5 / len(batch))
    assert len(provider.calls) == 1
    system, user = provider.calls[0]
    assert system["role"] == "system"
    assert "Program so far: (empty)" in user["content"]


def test_llm_prior_falls_back_when_nothing_parses(space: ActionSpace) -> None:
    provider = ScriptedProvider(lambda _messages, n: ["no idea"] * n)
    prior = LLMPrior(provider, UniformPrior(), samples=3)
    root = space.root()
    batch = space.legal(root)
    (p,) = prior(space, [root], [batch])
    assert np.allclose(p, 1.0 / len(batch))
    assert prior.rejected == 3


def test_prompt_lists_the_numbers_and_rows(space: ActionSpace, ctx: QuestionContext) -> None:
    _, user = build_messages(space, space.root())
    for leaf in ctx.leaves:
        assert leaf.arg.render() in user["content"]


class _Handler(BaseHTTPRequestHandler):
    seen: ClassVar[list[dict[str, object]]] = []

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        _Handler.seen.append({"path": self.path, "auth": self.headers.get("Authorization"), **body})
        reply = {"choices": [{"message": {"content": "add(1, 2)"}} for _ in range(body["n"])]}
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        return


@pytest.fixture
def stub_server() -> Iterator[str]:
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()
    server.server_close()


def test_openai_compatible_adapter_posts_chat_completions(stub_server: str) -> None:
    _Handler.seen.clear()
    provider = OpenAICompatibleProvider(stub_server, "local-model", api_key="test-key")
    out = provider.complete([{"role": "user", "content": "hi"}], n=2, temperature=0.3)
    assert out == ["add(1, 2)", "add(1, 2)"]
    (req,) = _Handler.seen
    assert req["path"] == "/v1/chat/completions"
    assert req["auth"] == "Bearer test-key"
    assert req["model"] == "local-model"
    assert req["temperature"] == 0.3
