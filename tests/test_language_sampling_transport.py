"""Explicit sampler requests reach the configured backend without new defaults."""
import dataclasses

import pytest
from test_omni_adapter import _adapter_config, _base_request

from qwen_omni_adapters.contract import parse_adapter_request
from runtime.adapter_server import AdapterStageError, build_language_payload

OPTIONS = {'temperature': 1.0, 'top_p': .95, 'top_k': 20, 'min_p': 0.0,
           'presence_penalty': 1.5, 'frequency_penalty': 0.0, 'repeat_penalty': 1.0,
           'repeat_last_n': 64, 'typical_p': 1.0, 'seed': 42, 'num_predict': 8192}
EXTENSIONS = {'top_k', 'min_p', 'typical_p', 'repeat_penalty', 'repeat_last_n'}


def config(**kwargs):
    return dataclasses.replace(_adapter_config(), language_api='openai',
        language_url='http://127.0.0.1:8901/v1/chat/completions', **kwargs)


def test_local_native_sampler_options_preserve_values_and_output_cap():
    request = _base_request(options=OPTIONS, think=False)
    parsed = parse_adapter_request(request)
    payload = build_language_payload(parsed, None, 'local', 'openai', config(language_llama_context_probe=True))
    assert {k: payload[k] for k in OPTIONS if k != 'num_predict'} == {k:v for k,v in OPTIONS.items() if k != 'num_predict'}
    assert payload['max_tokens'] == 4096
    assert payload['chat_template_kwargs']['enable_thinking'] is False
    assert parsed.passthrough['options'] == OPTIONS
    assert 'options' not in payload


def test_generic_openai_forwards_standard_penalties_without_native_extensions():
    payload = build_language_payload(parse_adapter_request(_base_request(options=OPTIONS)), None,
        'generic', 'openai', config())
    assert payload['presence_penalty'] == 1.5 and payload['frequency_penalty'] == 0.0
    assert not EXTENSIONS & payload.keys()


@pytest.mark.parametrize('options', [{}, {'presence_penalty': None, 'min_p': None}])
def test_omission_does_not_install_sampler_defaults(options):
    payload = build_language_payload(parse_adapter_request(_base_request(options=options)), None,
        'local', 'openai', config(language_llama_context_probe=True))
    assert not (EXTENSIONS | {'presence_penalty', 'frequency_penalty'}) & payload.keys()


def test_ollama_receives_original_options_and_no_top_level_duplicates():
    parsed = parse_adapter_request(_base_request(options=OPTIONS))
    payload = build_language_payload(parsed, None, 'model', 'ollama', _adapter_config())
    assert payload['options'] == OPTIONS | {'num_predict': 4096}
    assert not EXTENSIONS & payload.keys()


def test_declared_local_capability_cannot_target_an_external_endpoint():
    c = dataclasses.replace(config(language_llama_context_probe=True), language_url='https://remote.example/v1/chat/completions')
    with pytest.raises(AdapterStageError):
        build_language_payload(parse_adapter_request(_base_request(options=OPTIONS)), None, 'model', 'openai', c)
