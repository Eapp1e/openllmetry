import asyncio

from opentelemetry.instrumentation.vertexai import _abuild_from_streaming_response
from opentelemetry.instrumentation.vertexai.config import Config
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


class _Chunk:
    def __init__(self, text):
        self.text = text
        self.usage_metadata = None


async def _stream(chunks):
    for chunk in chunks:
        yield chunk


def _consume_async_stream(span, chunks):
    async def consume():
        async for _ in _abuild_from_streaming_response(
            span, None, _stream(chunks), "gemini-2.5-flash"
        ):
            pass

    asyncio.run(consume())


def test_async_streaming_records_completion_text(tracer_provider, span_exporter, monkeypatch):
    monkeypatch.setattr(Config, "use_legacy_attributes", True)
    tracer = tracer_provider.get_tracer(__name__)
    chunks = [_Chunk("Hello"), _Chunk(" world"), _Chunk("!")]

    with tracer.start_as_current_span("generate_content") as span:
        _consume_async_stream(span, chunks)

    (finished,) = span_exporter.get_finished_spans()
    content = finished.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"]

    assert content == "Hello world!"
    assert "<async_generator" not in content


def test_async_streaming_records_completion_role(tracer_provider, span_exporter, monkeypatch):
    monkeypatch.setattr(Config, "use_legacy_attributes", True)
    tracer = tracer_provider.get_tracer(__name__)

    with tracer.start_as_current_span("generate_content") as span:
        _consume_async_stream(span, [_Chunk("ok")])

    (finished,) = span_exporter.get_finished_spans()
    role_key = f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role"

    assert finished.attributes[role_key] == "assistant"
