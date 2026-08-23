from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import types
import unittest
import urllib.error

MODULE_PATH = Path('/mnt/data/chat_completions_replacement.py')


def install_stubs() -> None:
    openai = types.ModuleType('openai')
    openai.__path__ = []
    sys.modules['openai'] = openai

    openai_types = types.ModuleType('openai.types')
    openai_types.__path__ = []
    sys.modules['openai.types'] = openai_types

    openai_chat = types.ModuleType('openai.types.chat')
    openai_chat.ChatCompletionSystemMessageParam = dict
    sys.modules['openai.types.chat'] = openai_chat

    package_names = [
        'fakepkg',
        'fakepkg.providers',
        'fakepkg.tools',
    ]
    for name in package_names:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module

    agent = types.ModuleType('fakepkg.agent')
    agent.ChatMessage = dict
    sys.modules['fakepkg.agent'] = agent

    context = types.ModuleType('fakepkg.context')
    context.ExecutionContext = object
    sys.modules['fakepkg.context'] = context

    tool_mod = types.ModuleType('fakepkg.tools.tool')
    class Tool:
        def get_as_tool(self):
            return {'type': 'function'}
    tool_mod.Tool = Tool
    sys.modules['fakepkg.tools.tool'] = tool_mod

    memory_mod = types.ModuleType('fakepkg.tools.session_memory')
    class MemoryTool(Tool):
        def format_for_llm(self):
            return ''
    memory_mod.MemoryTool = MemoryTool
    sys.modules['fakepkg.tools.session_memory'] = memory_mod

    api_mod = types.ModuleType('fakepkg.providers.api')
    api_mod.chat_completions_url = lambda host: f'{host}/v1/chat/completions'
    sys.modules['fakepkg.providers.api'] = api_mod

    prompt_mod = types.ModuleType('fakepkg.providers.prompt')
    prompt_mod.build_system_prompt = lambda context: 'system prompt'
    sys.modules['fakepkg.providers.prompt'] = prompt_mod

    terminal_mod = types.ModuleType('fakepkg.providers.terminal')
    for name in ['BLUE', 'BOLD', 'CYAN', 'DIM', 'GREEN', 'RED', 'RESET', 'YELLOW']:
        setattr(terminal_mod, name, '')
    terminal_mod.separator = lambda *args, **kwargs: ''
    sys.modules['fakepkg.providers.terminal'] = terminal_mod


def load_module():
    install_stubs()
    name = 'fakepkg.providers.chat_completions_replacement'
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RetryConfig:
    max_attempts = 3
    request_timeout = 5.0
    initial_backoff = 0.0
    max_backoff = 0.0


class Model:
    id = 'test-model'
    max_output_tokens = 1024
    host = 'https://example.invalid'
    retry = RetryConfig()

    def decrypt_api_key(self):
        return 'secret'


class Config:
    def __init__(self):
        self._model = Model()

    def model(self):
        return self._model


class Context:
    def __init__(self):
        self.config = Config()


class FakeResponse:
    def __init__(self, payload, status=200):
        if isinstance(payload, (dict, list)):
            payload = json.dumps(payload)
        self._data = payload.encode('utf-8')
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self._data


def http_error(code: int, body: dict | str):
    if isinstance(body, dict):
        body = json.dumps(body)
    return urllib.error.HTTPError(
        'https://example.invalid/v1/chat/completions',
        code,
        'error',
        {},
        io.BytesIO(body.encode('utf-8')),
    )


class CallApiTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()
        self.context = Context()
        self.messages = [{'role': 'user', 'content': 'Improve logging'}]

    def run_call(self, urlopen, *, max_attempts=3):
        self.mod.urllib.request.urlopen = urlopen
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = self.mod.call_api(
                self.context,
                self.messages,
                {},
                max_attempts=max_attempts,
                initial_backoff=0.0,
                max_backoff=0.0,
                request_timeout=1.0,
                sys_prompt='system',
            )
        return result, output.getvalue()

    def test_absolute_empty_200_is_retried(self):
        responses = iter([
            FakeResponse({
                'choices': [{
                    'finish_reason': 'stop',
                    'message': {'role': 'assistant', 'content': '   '},
                }]
            }),
            FakeResponse({
                'choices': [{
                    'finish_reason': 'stop',
                    'message': {'role': 'assistant', 'content': 'done'},
                }]
            }),
        ])
        calls = []

        def urlopen(request, timeout):
            calls.append(request)
            return next(responses)

        result, log = self.run_call(urlopen, max_attempts=2)
        self.assertEqual(result['choices'][0]['message']['content'], 'done')
        self.assertEqual(len(calls), 2)
        self.assertIn("finish_reason(s): 0='stop'", log)
        self.assertIn('without usable assistant output', log)
        self.assertIn('Retrying in', log)

    def test_tool_call_with_empty_text_is_usable(self):
        calls = []
        response = FakeResponse({
            'choices': [{
                'finish_reason': 'tool_calls',
                'message': {
                    'role': 'assistant',
                    'content': '',
                    'tool_calls': [{'id': 'call_1', 'type': 'function'}],
                },
            }]
        })

        def urlopen(request, timeout):
            calls.append(request)
            return response

        result, log = self.run_call(urlopen, max_attempts=1)
        self.assertEqual(result['choices'][0]['finish_reason'], 'tool_calls')
        self.assertEqual(len(calls), 1)
        self.assertIn("finish_reason(s): 0='tool_calls'", log)

    def test_stealth_error_appends_continue_without_consuming_retry_slot(self):
        calls = []

        def urlopen(request, timeout):
            calls.append(json.loads(request.data.decode('utf-8')))
            if len(calls) == 1:
                raise http_error(400, {
                    'error': {
                        'message': '[Stealth] ERROR: provider requires explicit continuation'
                    }
                })
            return FakeResponse({
                'choices': [{
                    'finish_reason': 'stop',
                    'message': {'role': 'assistant', 'content': 'continued'},
                }]
            })

        result, log = self.run_call(urlopen, max_attempts=1)
        self.assertEqual(result['choices'][0]['message']['content'], 'continued')
        self.assertEqual(len(calls), 2)
        self.assertEqual(
            calls[1]['messages'][-1],
            {'role': 'user', 'content': 'continue your work'},
        )
        self.assertIn('Stealth provider failure detected', log)
        self.assertIn('stealth continuation recovery', log)
        self.assertNotIn('Retrying in', log)

    def test_repeated_stealth_error_surfaces_after_one_recovery(self):
        calls = []

        def urlopen(request, timeout):
            calls.append(request)
            raise http_error(400, {
                'error': {'message': '[Stealth] ERROR'}
            })

        self.mod.urllib.request.urlopen = urlopen
        with self.assertRaisesRegex(RuntimeError, "one-time 'continue your work' recovery"):
            with contextlib.redirect_stdout(io.StringIO()):
                self.mod.call_api(
                    self.context,
                    self.messages,
                    {},
                    max_attempts=1,
                    initial_backoff=0.0,
                    max_backoff=0.0,
                    request_timeout=1.0,
                    sys_prompt='system',
                )
        self.assertEqual(len(calls), 2)

    def test_ordinary_503_retry_is_preserved(self):
        calls = []

        def urlopen(request, timeout):
            calls.append(request)
            if len(calls) == 1:
                raise http_error(503, {
                    'error': {'message': 'service unavailable'}
                })
            return FakeResponse({
                'choices': [{
                    'finish_reason': 'stop',
                    'message': {'role': 'assistant', 'content': 'ok'},
                }]
            })

        result, log = self.run_call(urlopen, max_attempts=2)
        self.assertEqual(result['choices'][0]['message']['content'], 'ok')
        self.assertEqual(len(calls), 2)
        self.assertIn('Retrying in', log)

    def test_nonretryable_auth_error_does_not_append_continue(self):
        calls = []

        def urlopen(request, timeout):
            calls.append(json.loads(request.data.decode('utf-8')))
            raise http_error(401, {
                'error': {'message': 'invalid api key'}
            })

        self.mod.urllib.request.urlopen = urlopen
        with self.assertRaisesRegex(RuntimeError, 'HTTP 401'):
            with contextlib.redirect_stdout(io.StringIO()):
                self.mod.call_api(
                    self.context,
                    self.messages,
                    {},
                    max_attempts=3,
                    initial_backoff=0.0,
                    max_backoff=0.0,
                    request_timeout=1.0,
                    sys_prompt='system',
                )
        self.assertEqual(len(calls), 1)
        self.assertNotEqual(
            calls[0]['messages'][-1].get('content'),
            'continue your work',
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
