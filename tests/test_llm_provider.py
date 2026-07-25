"""LLM provider chain (mocked, no network): auto mode tries cloud first, falls
back to Ollama, then to the rule-based parser; explicit pins skip the other."""
from unittest.mock import patch
from backend.agents.encoder_agent import EncoderAgent

VALID = '{"rooms":[{"type":"bedroom","name":"Bed","area_min":12,"area_max":16}],"adjacencies":[]}'


def _enc(provider='auto', cloud=False, ollama=False):
    e = object.__new__(EncoderAgent)
    e.llm_provider = provider
    e.cloud_available = cloud
    e.llm_available = cloud or ollama
    e.openai_api_key = 'k' if cloud else None
    e.openai_base_url = 'https://api.groq.com/openai/v1'
    e.cloud_model = 'llama-3.3-70b-versatile'
    e.cloud_timeout = 5; e.llm_timeout = 5
    e.llm_model = 'llama3.2:latest'; e.ollama_url = 'http://localhost:11434'
    return e


def test_auto_uses_cloud_when_available():
    e = _enc('auto', cloud=True, ollama=True)
    with patch.object(e, '_call_cloud_llm', return_value=VALID) as mc, \
         patch.object(e, '_call_ollama_llm') as mo:
        out = e._extract_spatial_program_with_llm("a bedroom")
        assert out['rooms']
        mc.assert_called_once(); mo.assert_not_called()


def test_auto_falls_back_to_ollama_on_cloud_failure():
    e = _enc('auto', cloud=True, ollama=True)
    with patch.object(e, '_call_cloud_llm', side_effect=RuntimeError("timeout")) as mc, \
         patch.object(e, '_call_ollama_llm', return_value=VALID) as mo:
        out = e._extract_spatial_program_with_llm("a bedroom")
        assert out['rooms']
        mc.assert_called_once(); mo.assert_called_once()


def test_both_fail_falls_back_to_rule_parser():
    e = _enc('auto', cloud=True, ollama=True)
    with patch.object(e, '_call_cloud_llm', side_effect=RuntimeError("down")), \
         patch.object(e, '_call_ollama_llm', side_effect=RuntimeError("refused")):
        out = e._extract_spatial_program_with_llm(
            "A 2-bedroom home with a kitchen connected to the dining area.")
        types = {r['type'] for r in out['rooms']}
        assert 'bedroom' in types and 'kitchen' in types


def test_explicit_ollama_skips_cloud():
    e = _enc('ollama', cloud=True, ollama=True)
    with patch.object(e, '_call_cloud_llm') as mc, \
         patch.object(e, '_call_ollama_llm', return_value=VALID) as mo:
        e._extract_spatial_program_with_llm("a bedroom")
        mc.assert_not_called(); mo.assert_called_once()
