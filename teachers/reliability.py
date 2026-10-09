"""
Classify wrapped API failures and observe retries without logging secrets.

An evaluator installs a context-local sink. Training callers that do not
install one retain the existing retry behavior and produce no extra logs.
"""

import os
import re
from contextlib import contextmanager
from contextvars import ContextVar


_ATTEMPT_CONTEXT = ContextVar('teacher_attempt_context', default=(None, {}))


def exception_chain(exc):
    """
    Follow causes without losing the SDK type under teacher wrappers.
    """

    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or (
            None if exc.__suppress_context__ else exc.__context__
        )


def failure_details(exc):
    """
    Classify the innermost cause; never classify a prompt by keywords.

    The teacher's generic wrapper says 'response parsing failed' even
    for timeouts. A string search therefore misclassifies those errors.
    Unknown, malformed-response and authorization errors stop the sweep.
    """

    import openai

    cause = list(exception_chain(exc))[-1]
    # HTTPX may be the deepest cause under an OpenAI timeout wrapper.
    # Inspect each type from the inside out, retaining SDK status codes.
    kind = 'fatal'
    status = None
    for item in reversed(list(exception_chain(exc))):
        if isinstance(item, openai.APIStatusError):
            status = item.status_code
            kind = ('transient' if status == 429 or status >= 500
                    else 'fatal')
            break
        if isinstance(item, (openai.APIConnectionError,
                             TimeoutError, ConnectionError)):
            kind = 'transient'
            break
    return {
        'kind': kind, 'error_type': type(cause).__name__,
        'http_status': status, 'message': redact(str(exc)),
    }


def redact(value):
    """
    Remove known credentials and common authorization header formats.
    """

    text = str(value)
    for name in ('TYK_KEY', 'ALEPH_API_KEY', 'OPENAI_API_KEY',
                 'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_API_KEY'):
        secret = os.getenv(name, '').strip()
        if secret:
            text = text.replace(secret, f'<{name}:redacted>')
    text = re.sub(r'(?i)bearer\s+[^\s\"\',;}]+',
                  'Bearer <redacted>', text)
    return re.sub(r'\bsk-[A-Za-z0-9._-]+', '<key:redacted>', text)


@contextmanager
def attempt_context(sink=None, **fields):
    """
    Attach seed and step identifiers, independently in each worker.
    """

    old_sink, old_fields = _ATTEMPT_CONTEXT.get()
    token = _ATTEMPT_CONTEXT.set((
        sink if sink is not None else old_sink, {**old_fields, **fields},
    ))
    try:
        yield
    finally:
        _ATTEMPT_CONTEXT.reset(token)


def record_attempt(event, **fields):
    """
    Send only explicit metadata; never serialize a request or response.
    """

    sink, context = _ATTEMPT_CONTEXT.get()
    if sink is not None:
        sink({'event': event, **context, **fields})
