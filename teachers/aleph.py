"""
Translate the teachers' request format to Aleph Chat Completions.

Aleph support confirmed on 2026-09-08 that Responses is not served.
This adapter preserves the teacher prompt and schema, and exposes only
the visible answer and token usage expected by the existing teachers.
"""

import json
import os
import threading
import time
from types import SimpleNamespace

from openai import OpenAI


# Serialize requests across teacher instances in this process. Jobs
# must also be submitted one at a time; this is not a cluster lock.
_REQUEST_LOCK = threading.Lock()
_NEXT_REQUEST_AT = 0.0
REQUEST_GAP_S = 2.0
REASONING_EFFORTS = ('none', 'minimal', 'low', 'medium', 'high', 'xhigh')


def output_token_limit(value=None):
    """
    Resolve the explicit or environment token cap before any API call.
    """

    if value is None:
        value = os.getenv('ALEPH_MAX_OUTPUT_TOKENS', '4096')
    try:
        limit = int(str(value))
    except ValueError:
        raise ValueError('Aleph output token limit must be an integer')
    if limit < 1:
        raise ValueError('Aleph output token limit must be positive')
    return limit


def reasoning_effort(value=None):
    """
    Resolve an optional explicit setting without disabling thinking.

    An unset value preserves gateway defaults. Supported effort levels
    still depend on the selected model and the gateway deployment.
    """

    if value is None:
        value = os.getenv('ALEPH_REASONING_EFFORT', '')
    value = str(value).strip().lower()
    if not value:
        return None
    if value not in REASONING_EFFORTS:
        raise ValueError('Unsupported Aleph reasoning effort')
    return value


def reasoning_field_counts(message):
    """
    Count exposed reasoning fields without returning their contents.

    These counts help distinguish missing output from an unfinished
    reasoning channel. They do not measure reasoning-token usage.
    """

    counts = {}
    for name in ('reasoning', 'reasoning_content'):
        value = getattr(message, name, None)
        counts[f'{name}_chars'] = (
            len(value) if isinstance(value, str) else None
        )
    return counts


class AlephIncompleteReplyError(ValueError):
    """
    Report truncation with counts, without exposing generated text.
    """

    def __init__(self, choice, usage, limit):
        """
        Preserve generation diagnostics even when advice cannot be used.
        """

        details = getattr(usage, 'completion_tokens_details', None)
        prompt_tokens = getattr(usage, 'prompt_tokens', None)
        completion_tokens = getattr(usage, 'completion_tokens', None)
        reasoning_tokens = getattr(details, 'reasoning_tokens', None)
        answer_chars = len(choice.message.content or '')
        fields = reasoning_field_counts(choice.message)
        super().__init__(
            f'Aleph completion did not finish: {choice.finish_reason}; '
            f'max_tokens={limit}; prompt_tokens={prompt_tokens}; '
            f'completion_tokens={completion_tokens}; '
            f'reasoning_tokens={reasoning_tokens}; '
            f'visible_answer_chars={answer_chars}; '
            f'reasoning_chars={fields["reasoning_chars"]}; '
            f'reasoning_content_chars={fields["reasoning_content_chars"]}'
        )


def _chat_messages(value):
    """
    Translate text and image inputs without changing their contents.
    """

    if isinstance(value, str):
        return [{'role': 'user', 'content': value}]
    messages = []
    for message in value:
        content = message['content']
        if isinstance(content, list):
            parts = []
            for part in content:
                if part['type'] == 'input_text':
                    parts.append({'type': 'text', 'text': part['text']})
                elif part['type'] == 'input_image':
                    image = {'url': part['image_url']}
                    if 'detail' in part:
                        image['detail'] = part['detail']
                    parts.append({'type': 'image_url', 'image_url': image})
                else:
                    raise ValueError('Unsupported Aleph input content type')
            content = parts
        messages.append({'role': message['role'], 'content': content})
    return messages


class AlephClient(OpenAI):
    """
    An SDK client with an explicit, paced teacher Chat API adapter.
    """

    def teacher_response(
        self, *, model, input, text=None, reasoning=None,
        max_output_tokens=None,
    ):
        """
        Make one request, then validate and normalize its answer.

        Unsupported schema parameters fail openly. There is no
        automatic retry with a different schema, endpoint, or provider.
        """

        global _NEXT_REQUEST_AT

        limit = output_token_limit(max_output_tokens)
        kwargs = {
            'model': model,
            'messages': _chat_messages(input),
            # Aleph documents max_tokens for its Chat endpoint.
            'max_tokens': limit,
        }
        schema = None
        if text is not None:
            format_spec = dict(text['format'])
            if format_spec.pop('type') != 'json_schema':
                raise ValueError('Aleph teachers require a JSON schema')
            schema = format_spec['schema']
            kwargs['response_format'] = {
                'type': 'json_schema', 'json_schema': format_spec,
            }
        if reasoning is not None:
            if set(reasoning) != {'effort'}:
                raise ValueError('Unsupported Aleph reasoning options')
        effort = reasoning_effort(
            reasoning['effort'] if reasoning is not None else None
        )
        if effort is not None:
            kwargs['reasoning_effort'] = effort

        # Space even very fast successes and failures. Disable SDK
        # retries at construction so retries use the outer backoff.
        with _REQUEST_LOCK:
            delay = _NEXT_REQUEST_AT - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            try:
                response = self.chat.completions.create(**kwargs)
            finally:
                _NEXT_REQUEST_AT = time.monotonic() + REQUEST_GAP_S

        if not response.choices:
            raise ValueError('Aleph returned no completion choices')
        choice = response.choices[0]
        if choice.finish_reason != 'stop':
            raise AlephIncompleteReplyError(choice, response.usage, limit)
        message = choice.message
        if message.refusal or not message.content:
            raise ValueError('Aleph returned no usable visible answer')
        output_text = message.content
        if schema is not None:
            # Validate locally: gateway compatibility alone does not
            # establish that a particular backend enforces the schema.
            from jsonschema import Draft202012Validator

            data = json.loads(output_text)
            if next(Draft202012Validator(schema).iter_errors(data), None):
                raise ValueError('Aleph answer does not match teacher schema')

        # Reasoning-token counts can be included in completion_tokens,
        # but hidden reasoning fields are never explanation targets.
        usage = response.usage
        normalized_usage = None if usage is None else SimpleNamespace(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
        )
        return SimpleNamespace(
            output_text=output_text, usage=normalized_usage,
            model=getattr(response, 'model', None),
            response_id=getattr(response, 'id', None),
            reasoning_field_counts=reasoning_field_counts(message),
        )


def create_teacher_response(client, **kwargs):
    """
    Route Aleph to Chat while preserving OpenAI Responses requests.
    """

    if isinstance(client, AlephClient):
        return client.teacher_response(**kwargs)
    # Standard-rate reservations must not inherit a project's Fast tier.
    # Aleph has a different API contract and never receives this field.
    kwargs.setdefault('service_tier', 'default')
    return client.responses.create(**kwargs)
