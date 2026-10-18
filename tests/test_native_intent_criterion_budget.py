"""Choice descriptions fit unchanged bounds without shrinking native execution state.

The two compressed request fixtures are exact logical requests from the actual
Qwen LHN047 run, receipts 6b082e49a2bd43e29992bb6baadfad2a and
093df360df424200be2babc6343d01e0. Their canonical hashes are asserted below.
They reproduce a local parser fault; these offline checks claim no model skill.
"""
import base64
import copy
import json
import zlib

import httpx
import pytest
from test_ui_intent_choice import configured, run, subgoal

from portal.computer_use import (
    SystemOneUiProvider,
    _current_handoff,
    _digest,
    _intent_candidates,
    _intent_policy_projection,
)
from qwen_omni_adapters.computer_use import ComputerUseError
from runtime.kev_server import validate_request


def node(identity='target', **changes):
    return {'id': identity, 'generation': '1', 'scope_id': 'window', 'role': 'textbox',
        'name': 'Observed document', 'value': '', 'visible': True, 'enabled': True,
        'sensitive': False, 'focused': True, 'bounds': {'x': 4, 'y': 8, 'width': 400, 'height': 300},
        'operations': ['click', 'double_click', 'right_click'], **changes}


def state(*nodes):
    return {'session_id': 'session', 'revision': 'observed-revision',
        'scope': {'kind': 'desktop', 'id': 'window'}, 'truncated': False, 'nodes': list(nodes)}


def request_for(observation, candidates=None, context=None, *, choice='intent-abstain'):
    candidates = candidates if candidates is not None else _intent_candidates(observation, {})
    context = context if context is not None else {'goal': 'Use the observed application',
        'active_objective': 'Inspect the selected document', 'omission_is_absence': False}
    requests = []

    def respond(request):
        body = json.loads(request.content)
        validate_request(body)
        requests.append(body)
        legal = body['questions']['action']['criteria']
        return httpx.Response(200, json={'model': provider.model, 'answers': {'action': {
            'type': 'choice', 'choice': choice, 'probabilities': {key: float(key == choice) for key in legal}}}})

    provider = SystemOneUiProvider('http://127.0.0.1:1', mode='active',
        client=httpx.Client(transport=httpx.MockTransport(respond)))
    before = copy.deepcopy((observation, candidates, context))
    result = provider.decide_intent(observation, candidates, context=context)
    assert (observation, candidates, context) == before
    return result, requests, candidates


def descriptions(payload):
    return [json.loads(value) for value in payload['questions']['action']['criteria'].values()]


def historical_inputs(name):
    original = json.loads(zlib.decompress(base64.b64decode(_HISTORICAL_REQUESTS[name])))
    observed = json.loads(original['questions']['action']['criteria']['intent-0'])['node']
    current = json.loads(zlib.decompress(base64.b64decode(_HISTORICAL_SOURCE_CONTEXT)))
    current['nodes'] = [copy.deepcopy(observed)]
    context = json.loads(original['state'])['task']
    bank = _intent_candidates(current, {'node_id': observed['id']})
    return original, current, bank, context


@pytest.mark.parametrize('value', ['', 'https://example.test/document', 'é' * 1100,
    'First paragraph.\nSecond paragraph.\n' * 50, '\\"\n' * 350, '本文🧭' * 1365, 'long text ' * 10000])
def test_editable_content_is_shared_once_with_explicit_coverage(value):
    observed = node(value=value)
    result, requests, bank = request_for(state(observed))
    assert len(requests) == 1 and result['receipt']['status'] == 'accepted'
    payload = requests[0]
    assert max(map(len, payload['questions']['action']['criteria'].values())) <= 1000
    compact = json.loads(payload['state'])
    assert compact['schema'] == 'robit.omni.ui-intent-projection.v2'
    assert len(compact['nodes']) == 1
    shared = compact['nodes']['n0']
    assert shared['observed']['id'] == observed['id']
    assert 'source_sha256' not in shared['projection']
    assert shared['projection']['input_authority'] is False
    if len(json.dumps(value, ensure_ascii=False, separators=(',', ':'))) <= 2048:
        assert shared['observed']['value'] == value
        assert not shared['projection']['omitted_fields']
    else:
        assert 'value' not in shared['observed']
        coverage = shared['projection']['omitted_fields']['value']
        assert coverage == {'status': 'partial', 'source_json_characters': len(json.dumps(value,
            ensure_ascii=False, separators=(',', ':'))), 'source_sha256': _digest(value),
            'complete': False, 'omission_is_absence': False, 'preview': value[:256],
            'source_characters': len(value)}
    primitives = [item for item in descriptions(payload) if item['kind'] == 'intent']
    assert {item['node_ref'] for item in primitives} == {'n0'}
    assert all(item['node'] == {'id': observed['id']} and item['args'] == {}
               and item['effect'] == 'unknown' and item['non_authorizing'] for item in primitives)
    assert all(item['node']['value'] == value for item in bank if item['kind'] == 'intent')
    assert result['receipt']['intent_bank_sha256'] == _digest(bank)


@pytest.mark.parametrize('name', ['6b082e49a2bd43e29992bb6baadfad2a', '093df360df424200be2babc6343d01e0'])
def test_exact_actual_qwen_requests_now_fit_original_parser(name):
    original, current, bank, context = historical_inputs(name)
    assert _digest(original) == _HISTORICAL_HASHES[name]['request']
    with pytest.raises(ValueError, match='invalid action ID or description'):
        validate_request(original)
    assert [len(v) for k, v in original['questions']['action']['criteria'].items()
            if k.startswith('intent-') and k not in {'intent-observe', 'intent-abstain'}] == [2918, 2925, 2924]
    assert _digest(bank) == _HISTORICAL_HASHES[name]['bank']
    assert all({key: value for key, value in candidate.items() if key != 'id'} ==
        json.loads(original['questions']['action']['criteria'][candidate['id']]) for candidate in bank)
    result, requests, _ = request_for(current, bank, context)
    assert result['receipt']['status'] == 'accepted' and len(requests) == 1
    rebuilt = requests[0]
    validate_request(rebuilt)
    assert max(map(len, rebuilt['questions']['action']['criteria'].values())) < 1000
    shared = json.loads(rebuilt['state'])['nodes']['n0']
    assert shared['observed'] == bank[0]['node']
    assert shared['observed']['value'] == bank[1]['node']['value'] == bank[2]['node']['value']
    assert not shared['projection']['omitted_fields']
    assert result['receipt']['intent_bank_sha256'] == _HISTORICAL_HASHES[name]['bank']
    assert result['receipt']['context_sha256'] == _HISTORICAL_HASHES[name]['context']


@pytest.mark.parametrize('name', ['6b082e49a2bd43e29992bb6baadfad2a', '093df360df424200be2babc6343d01e0'])
def test_semantic_handoff_keeps_operation_target_effect_and_evidence_without_audit_duplication(name):
    _, current, bank, context = historical_inputs(name)
    result, requests, _ = request_for(current, bank, context)
    compact = json.loads(requests[0]['state'])
    task = compact['task']
    original = context['action_handoff']
    projected = task['action_handoff']
    assert task['goal'] == context['goal'] and task['active_objective'] == context['active_objective']
    assert projected['schema'] == 'robit.omni.ui-intent-handoff-projection.v1'
    for key in ('authority', 'input_authority', 'operation_family', 'next_operation',
                'needed_evidence', 'expected_transition'):
        assert projected[key] == original[key]
    target = compact['nodes'][projected['target']['node_ref']]['observed']
    assert (target['id'], target['generation'], target['scope_id']) == (
        original['target']['node_id'], original['target']['generation'], original['target']['scope_id'])
    assert projected['binding_metadata_retained_privately'] is True
    assert projected['omission_is_absence'] is False
    assert not {'goal_sha256', 'source_epoch', 'source_phase', 'source_observation', 'predecessor'} & projected.keys()
    assert result['receipt']['context_sha256'] == _digest(context)
    assert result['receipt']['intent_bank_sha256'] == _digest(bank)


@pytest.mark.parametrize('input_attempted', [None, False, True])
@pytest.mark.parametrize('effect_state', ['unknown', 'absent'])
def test_historical_feedback_removes_only_audit_hashes_and_preserves_uncertainty(input_attempted, effect_state):
    _, current, bank, context = historical_inputs('6b082e49a2bd43e29992bb6baadfad2a')
    context['previous_result']['facts'] = {'status': 'rejected', 'error': 'native_pointer_target_occluded',
        'reason': 'The actual guarded input did not prove dispatch.',
        'effect_receipt': {'effect_state': effect_state, 'environment_may_have_changed': True},
        'ui_boundary_receipt': {'input_attempted': input_attempted, 'adapter_called': True,
            'observed': False, 'verified': False, 'cached': False, 'task_progress': False,
            'new_effect_count': 0, 'novelty_key': 'a' * 64,
            'before_semantic_revision': 'b' * 64, 'after_semantic_revision': 'c' * 64}}
    original = copy.deepcopy(context)
    result, requests, _ = request_for(current, bank, context)
    previous = json.loads(requests[0]['state'])['task']['previous_result']
    expected = copy.deepcopy(context['previous_result']['facts'])
    for key in ('novelty_key', 'before_semantic_revision', 'after_semantic_revision'):
        expected['ui_boundary_receipt'].pop(key)
    assert previous['facts'] == expected
    assert previous['facts']['ui_boundary_receipt']['input_attempted'] is input_attempted
    assert previous['facts']['effect_receipt']['effect_state'] == effect_state
    assert previous['facts']['status'] == 'rejected' and previous['facts']['error'] == 'native_pointer_target_occluded'
    assert previous['projection']['audit_bindings_retained_privately'] == [
        'novelty_key', 'before_semantic_revision', 'after_semantic_revision']
    assert previous['projection']['omission_is_absence'] is False
    assert context == original and result['receipt']['context_sha256'] == _digest(original)


@pytest.mark.parametrize('fault', ['goal', 'objective', 'session', 'scope', 'generation', 'value', 'authority'])
def test_invalid_handoff_is_not_compacted_or_relabelled_as_current_target(fault):
    _, current, bank, context = historical_inputs('6b082e49a2bd43e29992bb6baadfad2a')
    if fault in {'goal', 'objective'}:
        context['goal' if fault == 'goal' else 'active_objective'] += ' changed'
    elif fault == 'session':
        current['session_id'] = 'different-session'
    elif fault == 'scope':
        current['scope']['id'] = 'different-scope'
    elif fault in {'generation', 'value'}:
        current['nodes'][0][fault] += ' changed'
    else:
        context['action_handoff']['input_authority'] = True
    bank = _intent_candidates(current, {'node_id': current['nodes'][0]['id']})
    before = copy.deepcopy((current, bank, context))
    compact, _ = _intent_policy_projection(current, bank, context)
    assert compact['task']['action_handoff'] == context['action_handoff']
    assert 'node_ref' not in compact['task']['action_handoff']['target']
    with pytest.raises(ComputerUseError):
        _current_handoff(context['action_handoff'], goal=context['goal'], objective=context['active_objective'],
            state=current, target={'node_id': current['nodes'][0]['id']})
    assert (current, bank, context) == before


def test_stale_private_candidate_cannot_acquire_current_handoff_alias_by_same_id():
    _, current, bank, context = historical_inputs('6b082e49a2bd43e29992bb6baadfad2a')
    for candidate in bank:
        if candidate['kind'] == 'intent':
            candidate['node']['value'] = 'different observed contents'
    compact, _ = _intent_policy_projection(current, bank, context)
    assert compact['task']['action_handoff'] == context['action_handoff']
    assert 'node_ref' not in compact['task']['action_handoff']['target']


def test_observed_parent_row_description_and_text_coverage_remain_exact():
    ancestors = [{'id': 'parent', 'generation': '7', 'role': 'row', 'name': 'Input 5 items',
        'visible': True, 'enabled': True}]
    observed = node(description='Observed document body',
        structure={'parent_id': 'parent', 'child_index': 2, 'ancestors': ancestors},
        text_observation={'character_count': 7, 'range_start': 0, 'range_end': 3, 'truncated': True},
        value='abc')
    _, requests, _ = request_for(state(observed))
    shared = json.loads(requests[0]['state'])['nodes']['n0']
    assert shared['observed'] == observed
    assert shared['observed']['text_observation']['truncated'] is True
    assert shared['projection']['all_source_fields_included'] is True
    assert shared['projection']['omission_is_absence'] is False


def test_large_structure_or_other_metadata_is_omitted_explicitly_and_retained_privately():
    ancestors = [{'id': str(index) + 'p' * 511, 'generation': 'g' * 512, 'role': 'row',
        'name': 'n' * 512, 'visible': True, 'enabled': True} for index in range(3)]
    structure = {'parent_id': ancestors[-1]['id'], 'ancestors': ancestors}
    selected = ['observed text ' * 200, 'another observed value ' * 150]
    observed = node(structure=structure, selected=selected, description='details' * 70)
    _, requests, bank = request_for(state(observed))
    shared = json.loads(requests[0]['state'])['nodes']['n0']
    assert 'structure' not in shared['observed'] and 'selected' not in shared['observed']
    assert shared['observed']['description'] == observed['description']
    omissions = shared['projection']['omitted_fields']
    assert omissions['structure']['status'] == 'omitted' and 'preview' not in omissions['structure']
    assert omissions['structure']['source_sha256'] == _digest(structure)
    assert omissions['selected']['status'] == 'omitted'
    assert shared['projection']['all_source_fields_included'] is False
    assert bank[0]['node'] == observed


@pytest.mark.parametrize('identity', ['重复🧭' * 100, '"' * 512, '\x00' * 512, '\\' * 512])
def test_unicode_and_escaped_identities_keep_exact_shared_binding(identity):
    _, requests, _ = request_for(state(node(identity)))
    payload = requests[0]
    assert max(map(len, payload['questions']['action']['criteria'].values())) <= 1000
    assert json.loads(payload['state'])['nodes']['n0']['observed']['id'] == identity
    assert {p['node_ref'] for p in descriptions(payload) if p['kind'] == 'intent'} == {'n0'}


def test_duplicate_labels_never_merge_target_identity_or_generation():
    first, second = node('first', name='Save'), node('second', name='Save', generation='next')
    _, requests, bank = request_for(state(first, second))
    compact = json.loads(requests[0]['state'])
    assert {entry['observed']['id'] for entry in compact['nodes'].values()} == {'first', 'second'}
    assert {entry['observed']['generation'] for entry in compact['nodes'].values()} == {'1', 'next'}
    aliases = {entry['observed']['id']: alias for alias, entry in compact['nodes'].items()}
    for proposal in descriptions(requests[0]):
        if proposal['kind'] == 'intent':
            assert proposal['node_ref'] == aliases[proposal['node']['id']]
    assert len([c for c in bank if c['kind'] == 'intent']) == 6


@pytest.mark.parametrize('changes', [{'visible': False}, {'enabled': False}, {'sensitive': True}])
def test_ineligible_nodes_do_not_enter_descriptive_projection(changes):
    observed = node(value='PRIVATE SENTINEL', **changes)
    _, requests, _ = request_for(state(observed))
    assert json.loads(requests[0]['state'])['nodes'] == {}
    assert 'PRIVATE SENTINEL' not in json.dumps(requests)
    assert {c['kind'] for c in descriptions(requests[0])} == {'observe', 'abstain'}


@pytest.mark.parametrize('drift', ['generation', 'scope', 'missing', 'disabled'])
def test_compacting_long_controls_preserves_actual_fresh_offer_guards(drift):
    adapter, store, cancel, requests = configured()
    original = adapter.observe
    observations = 0

    def observed():
        nonlocal observations
        observations += 1
        current = original()
        current['nodes'][0]['value'] = 'document content ' * 200
        if observations >= 2:
            if drift == 'generation':
                current['nodes'][0]['generation'] = 'changed'
            elif drift == 'scope':
                current['scope']['window_id'] = 'changed'
            elif drift == 'missing':
                current['nodes'].pop(0)
            else:
                current['nodes'][0]['enabled'] = False
        return current

    adapter.observe = observed
    result = run(store, cancel, subgoal())
    assert len(requests) == 1 and not adapter.calls
    assert result['intent_selection_receipt']['fresh_offer_matched'] is False
    assert result['selection_receipts'][0]['selection_status'] == 'intent_choice_stale'


def test_returning_projection_alias_instead_of_offered_candidate_cannot_dispatch():
    def alter(body):
        body['answers']['action']['choice'] = 'n0'
    adapter, store, cancel, requests = configured(alter=alter)
    result = run(store, cancel)
    assert len(requests) == 1 and not adapter.calls
    assert result['policy_receipts'][0]['status'] == 'rejected'


# Frozen actual logical inputs, compressed only to avoid repeating the complete
# document and planner context in every regression case. No oracle is included.
_HISTORICAL_REQUESTS = {
    '6b082e49a2bd43e29992bb6baadfad2a': (
        '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'
    ),
    '093df360df424200be2babc6343d01e0': (
        'eNrtWuuO47YVfhVi/jQFRrZkybpM/jTdXBrkimSDAMkUAkVStnZkUdHFs5NggD5En7BP0u+QlGzPeKdJUBQpYGCwK1Pi0bl+5zu0'
        'f7naaanqq5urN7xTsuX1TnXLO7X3ouIvsSiFWotVyaMiKmQc+as4DGNexFEm1oInabpO5Hp1dX3106j6odJNf3XzyxUXdElXoqsG'
        '1VWcrqtmUM3g+XjXL7dXvNv0t1h+vL69UmWpxIBPt1djc9fo++b2Cst3VSPNot1p1hpoS9turwo9NrK311tVbbYkIF5neOi+ksMW'
        'n9K1j09vcRVFuHjARZKaFza8qBUJH7pRYaHUYuyPFzaqUR0nM4wGgXl5ZdXhQ99WN37gR4koV0kRlJGfBIqXcSiE8P1UlErFVl2+'
        'U2aP+aRbJ5PU/vH2qldDvuf1qMxdo4O5EnUl7syV1CMUzY8WOr7JZadb86kjs4/u9qLTdW2dpx5ysdWdtDdUDQ/ng3pr3Tg8tMp9'
        '+jvJ0bVVk5YK/dYJg775ic1hIGPhr3karHy1CsI4LtYilYkfl2Esk+ldTV8N1Z4klrzuyZ093CqGsXOx441AuujO+OGXM/H4z+4v'
        '/WQVl2ki4kQiNUWZZOtk7cuSp36WxOUZ989WtrxR1kv7qq8Ks0ovfrz+fbog6FFRptwPY+VnMk7jpJRJEsdrXorSF/9LXcokXUdJ'
        'FKFwlSz8VRZJ+CVLMxHxws+Cl3Sx6ZOTSuc0okwR26qWOQpTUVlRebXAjWbIf6MSVIWUbLkuetXtJ5Ngs+AtJYrMO95sXL4oAwRB'
        'Gvgml3hHte4/Gn14B7hRXS6AB8PRUybjIXW+QauwoxF8MC41uUkyXAlC+69RbZVgA+/vWNFVqrxhn//tSw+Fzv71j3+yL4AR445t'
        'eYO6HJiSFXK44jVred/f3jb4e72teoa/fmzbulLSypJ84Nes0QPjDf76e9VdM4uSrO20UBIWM90xAMSu6nu3vmuHBftGcUjZKqbe'
        'VoDYZsMoepJVTQsdjJb9kl7ASgjgdW1fWUJ8v7BKfVWWlSA9ez12QjHjQZLUakBrz95r+L7awC14T6fHzda8r1PYUXT6HgGC0hLy'
        '61rfY71We94MTE9i66q56/98gzd5bKP3490N++4z9oneq67ZITnYh9WmGvDgtwh1JRQeZOy7bz6/YdthaPub5fL+/n6BnYvxbrkZ'
        'K0n4sOyHh1p59FEtuTdo72dPlx6e8sY7z9w0cl7rthI3bLZxjsqyrXnVeDXSaOQbshpb2CTeeMuaaXoXPPpmhH8hxUL0+6wamNSq'
        't3EbB4Bp9bNiXOIFeMC9yvjYY7sKtdPrcrhhX0yXz82sFe+axfzsQujdUjXe2J8Ye69q3FDLP4x5+PtQiZoIgku7sqqx8722U1S9'
        'lLqIMO0pHgY1pcKn9OgSHascvGAxvB2era7OroZPVoXuOlvL/eJNrxur0TcwrCKNSvPilg/bnnKTU+thg2bff/XNZ3/96MtXf7Pa'
        'fARrlHymzsny6vxy+HQZqANs8gwMLkS/N/c+6ACOe7WcI+S1XNwtfq5aq7C7z4RuBsIsUuuHT79mO7UrUGHWAPgYhY679YMJn605'
        'qtXvdXdXqEZsvdlIu+W9RmOblU1xuWYaOzsGiIL38Kl18QRIqYFQZmyMiDmaO/7AChN6JU3szvrqrKfO+undXvquJ4xB0lgjGTG7'
        'bg7fDJFoDB1n9x1vAYjkA6m6BfvadBomK8oG3T0wfAL8WadNVi7YhyOQl0CeTLWSyC3Oz/ZpVEdRwdxmwT4CSj1MQdihRhgSCxoh'
        '95mcst4Bp5FjUpzxwSUeYFIWFGgXZQfAHvBbjsLkJ9UKMPcBjochh8RcsC81K7QG0d3A+Bbv2PJe4XKrocfQGfbXb7UmtL5m/VbV'
        '9fLV559Cn3pPPWTHCUpwB0S7NZAOm62DvFINYuueRD3o2po+l7qEJQgGmpVpIgv2VSdhEZzR2PRy2L+Uqr8bdMs++e5T0wcoLzml'
        'JTlG2W4F/1FE6AV027SgncKbJBDkHLshKt/kkzLQ/EByoG9h2rEjto+YL9zwEFyGh8vwcBkeLsPDZXi4DA+X4eEyPFyGh8vwcBke'
        'LsPD8+HhlP0ezRCrywxxmSEuM8RlhrjMEJcZ4jJDXGaIywxxmSEuM8Rlhng+Q5zQ26MRghfwZtXYQWKeEtyqGxPOyT4SYWmHeiLC'
        'rb4o4pFkWD5rf3B19QoBQkbrRh3avnNh21U7w4WN45x4iZwplcnlCaWB5Q2Y3Z96wwoIUIs3yl7dV4hR4xgA1cxI/XijKdVfb4+f'
        'RCSdHGattPXUqHu7x1kzPCzY57xQlA9QyrAfmxrgSd1oeoUYO6PfETvrUSocCYW8bnVPBUKto/GOfIS85bRouEuFeYrZye7avKiz'
        'gN2zEv9sbYrjStcjiV+wD2z8WFUaY2e7rmd3XR/5E45zREQYy8HrVEf7FwgyjTcIDOYfsI2rR4QMogcX7V5s1Y7bBNNFNSz0rqkW'
        'Y+W51IDyb2y7WeyDF1KB6CTS3DJUiocR+QqsrVaDLYPdS4SRyg8AVz9Q2R5jPrQ4JNJR82Om2HsiaUoxA7kIytdTD1YGyKD9gOex'
        '86dRDyZyxvs9HAcsMyED/dugE2BQw6sEHNxz7Kc8UQa58dK2IhiyYYMKiCTF3iI6sotA2uI6+GmvqP0Y1We1Z254jpbcaws5Rw45'
        'JScm6Y3H3mIYkeSgUy5DG5kZUKktiDsiQKZCjvU3UsyyUZXyCBtMKe1s6quuKtH60C8nP2Mz3Tl4kSIF6XD0h6pG4nU0kfW2UYxP'
        'utlzDrIcdu3yp3twPD6gJIQHJyCNPJLkrfxVHPhBssRQMlQ75RGLX27aMfC4EKodDA2udbPxTOrpxopyI8kkcuITs7zXge+vg+iH'
        '5UwZbs6wn+sz1Of6DO+5fhfpuX6B8SzYZ0q1Zho5BE0eO9C2wj+IfwhYXBXJE6JrIfLAlwyOGiyxxWFPdM7MTVPPpAKfs9rsmJqs'
        '665AVeJADpxpYARlmeVh3h8r9JU79VBojjIEslE1GK2QdzCLuMpO0wdHzWkKHBhGDWQo9aRvUdxL5/sHyz51/T7jqP8dxjjMu6bn'
        'OKioacs8F/RY7UuCJYi0/XzCcRrsFowInet1YKkS42GPqkNakJKQoRQ4JYFEqzsUTAPOVIHGoETdT5PRBxqkR42m9DHxKWAgnQzB'
        'NFJVoIs85RhzVZvyNWBjmO7EIW0t6iczL+YwFwBq02bNM0eIROOkY054Izo7msg0d9sBxPA5S5JQEHPTPgN2T+Zreun1k5TCv3u4'
        'zrAlvN1qO4VnORH5Iyc7AkUdJrcvsL/OsC06p5TMd8qglHnQMoh87p/m6fkUoRwngnowCEAFy8uq21lu53iCDeuhAWHO700PfoqN'
        '18dOtoXxokEYNkptOLtqJmw3iGO5srV31j+feYs9MLMUJz86FLZnPRpu4Ei9w0kQnfTWGEzpCJUOjk/uHFGb/NmxEN13BzF51eeg'
        'ltQoj+4CL/YVqi4Hboz1YEnAUXTm+7Y48paIvT16VTvUAsWS2pd5GByc8FNwexB4au9MynLi8KfnZXT4Rzl14qJJRb7n6PgHzk1H'
        'zHQo5HRFeOm4+UCclT2+BXz17njRujhvNSDsIT99zDI7mC8UJg8n813Myj7suYcnWuVEGDidFJlCqpp91WlzeoSoPuRbgFhuO9DJ'
        '2d1Y5aaKgai/Upfp8UmLCUWOPOd8OlCoTpNCgAWdZomj9Md+l7w1B5IAs5Mbv8YmRypzFNkGqdUfrYOf5c5l85Hm4+Ojq3gkqqGZ'
        'ZfmS/TYbPbvBcxsmT5yvM2BXtUP5DNPLNdXek6R7MREtcm35ah0byYWMpPAzLqJVuFrLUCXBOkyDOCxEkfoBT4NonWR+oFbCV6kK'
        'RCyyUIpoLdJE8DC1NW9qLUdfEVsjdR2IMIjiLItUGYVFka7Q74NSFWUoVnGsjne5aryJD0vPzqF/99cfZt+RueFKhlkBu3y58ksh'
        '4zAKS7UOk7gI0tIPoW22kgkvlQqDtUzSKEyilfR5IsMEyw7iLBgduzEJwiBLRVlmSRiotBCKDPZVJqMsiKNUqEKEURCvizTK4HRV'
        'JkmsytU6zYr1KnbVToO5q3cOlZKiLFYiK5J17MvYj7ALKsVhxhX+gd4qKqJyVaRlss58H/+Tg8NURGnhNLWsIv8vizbfGJjBxcbH'
        'APpv/0bu7DcovzfWU/XnVrHj6CQxOGbJg6gMi0yqOIp5nMbZKuThKpF+mYl1loZC8cIvY8SmEEEI9xTrMCpksUqtwfM3hnnJMb+5'
        'ytTm2IwK8cDcTnvm/2nPb8w3Qu1xgL4130mdEOynlhBhLvRbMslNdPSUmAZxO2oS7zVHCLjXGC3ecbrM3GHLaxL7kZlsJuWIy+bI'
        'aunIwI+3Vx+Ta53sHkTwXUKfuNuyPhr0T+Z7e7hWdWyHucF4Sbs532jwyp3KGIvnd53MdM9O121onh2vk2SyYzDusucBp4PvdDpg'
        'X/3VO+Z5o0rZ6d0pI7bf5RiJ89chloJP5JkOAd41/tsvpTH7G03Azrj5RvmAzC+0tmmXd9g19bed/e3AXLY5vKFcMzv9fv7XVdtp'
        'v6TzrEPu5hgAzZkstlHvnJFi+q3Dix3zLAWwvd6wpVNsjTNQjlWEzojeKf11FpSJz9FssjAA5GaxX/g8LdYEun4So+f6osjCLAIQ'
        'Sz+M6Gz18d+I5poC'
    ),
}

_HISTORICAL_HASHES = {'6b082e49a2bd43e29992bb6baadfad2a': {'request': '2875a523126afe4916d76be6e06b6e96bc5a41eeaacd1df235d6c409d899ab94', 'bank': '553e1163b49ba1a4d5db789310ae38025993409d6c311bd45252ec8df4c81d23', 'context': 'fc80808ec79e1b32b51370c1ab4df6bc5fd0780200d88796c8fac473944a3a59'}, '093df360df424200be2babc6343d01e0': {'request': 'd4806235c4ac8f946f068a6c0f0f63329fde55aecfe6b84dd508308a986ffe83', 'bank': '553e1163b49ba1a4d5db789310ae38025993409d6c311bd45252ec8df4c81d23', 'context': 'ce9d3f2e891c048e6228e8214a0d1924d1b50fd6cd728c90abdb5d6c3c4c0ce2'}}


# Exact archived source identity/scope for both private observations.
_HISTORICAL_SOURCE_CONTEXT = (
    'eNo1j0GOAyEMBP/CeQ8GYxvPZ1aAQUKRMqMhSg5R/r5ko1zb5VL3080259ivv8Pc5qhwpYyG6hVNKnhC37rUkkEgRJGYYqXeOwKi'
    '6oL6uiD0aKlUcT/ubPfxFi4ba7YcYsbkczIg9V0gWwBFX3pRhgI5FQJjAGEMBLUoahRiA4zLNut+NLc93X+9fJvH2NAbV6CcfIAW'
    'PDIXqskEuCOb2Hq7jOubtzYvt/1YwXHu92Ht/ErCyh4L2h+f4clLUgV1r9cf7WNOjA=='
)


@pytest.mark.parametrize('objective', [
    'Open the second Save control without using the first',
    'Inspect the selected row; do not activate the similarly named popup',
    '查看所选文件，不要选择同名的其他行 🧭',
])
def test_private_audit_removal_preserves_controls_relationships_and_decision_semantics(objective):
    ancestor = {'id': 'row-opaque-identity', 'generation': 'parent-generation',
        'role': 'table_row', 'name': 'Observed row', 'visible': True, 'enabled': True}
    first = node('first-control', name='Save', generation='generation-a',
        structure={'parent_id': ancestor['id'], 'child_index': 0, 'ancestors': [ancestor]})
    second = node('second-control', name='Save', generation='generation-b',
        structure={'parent_id': ancestor['id'], 'child_index': 1, 'ancestors': [ancestor]})
    observation = state(first, second)
    preference = {'authority': 'planner_preference', 'input_authority': False, 'effect': 'unknown',
        'intent': {'id': 'step', 'target': {'node_id': second['id']}, 'verb': 'double_click', 'args': {}},
        'semantic_sha256': 's' * 64}
    context = {'goal': 'Exact original task', 'active_objective': objective,
        'needed_evidence': ['Selected row contents', 'Popup remains unmodified'],
        'planner_preference': preference,
        'transition': {'kind': 'popup_opened', 'source_ref': {'session_id': observation['session_id'],
            'scope_id': observation['scope']['id'], 'revision': observation['revision']}},
        'previous_result': {'facts': {'status': 'abstained', 'effect_receipt': {'effect_state': 'unknown'}}}}
    original = copy.deepcopy((observation, context))
    result, requests, bank = request_for(observation, context=context)
    compact = json.loads(requests[0]['state'])
    expected_context = copy.deepcopy(context)
    expected_context['planner_preference'].pop('semantic_sha256')
    assert compact['task'] == expected_context
    assert 'state_revision' not in compact
    assert [entry['observed'] for entry in compact['nodes'].values()] == [first, second]
    assert all('source_sha256' not in entry['projection'] for entry in compact['nodes'].values())
    assert {json.loads(value)['node']['id'] for value in requests[0]['questions']['action']['criteria'].values()
            if json.loads(value)['kind'] == 'intent'} == {first['id'], second['id']}
    assert len([item for item in bank if item['kind'] == 'intent']) == 6
    assert result['receipt']['state_revision'] == observation['revision']
    assert result['receipt']['context_sha256'] == _digest(context)
    assert result['receipt']['intent_bank_sha256'] == _digest(bank)
    assert (observation, context) == original


def test_exact_ornith_088_projection_removes_only_redundant_audit_fields():
    # Exact source-derived request from both actual088 HTTP-error receipts.
    # These inputs exercise projection only; they do not reconstruct an input
    # authorization or claim the original complete native observation.
    original = json.loads(zlib.decompress(base64.b64decode(_ORNITH088_REQUEST)))
    assert _digest(original) == 'f49b0b64e93fb5575195a5e9eb3792705c451310d0079ada5219fe7e20bd16b5'
    old = json.loads(original['state'])
    nodes = [entry['observed'] for entry in old['nodes'].values()]
    observation = {'session_id': 'projection-only-source', 'revision': old['state_revision'],
        'scope': {'kind': 'desktop', 'id': nodes[0]['scope_id']},
        'truncated': old['task']['observation_truncated'], 'nodes': nodes}
    context = old['task']
    bank = _intent_candidates(observation, context['planner_preference']['intent']['target'])
    assert _digest(bank) == '8b18504bb9ab1d06919d0d511893a572ea157a15c2987bd5a884da08dfcd20b1'
    assert _digest(context) == 'b5759d54dfb94910638e9fe68ceb978bff7849bbca344e37440f90e218d2a243'
    expected = copy.deepcopy(old)
    expected.pop('state_revision')
    for entry in expected['nodes'].values():
        entry['projection'].pop('source_sha256')
    expected['task']['planner_preference'].pop('semantic_sha256')
    result, requests, _ = request_for(observation, bank, context)
    assert result['receipt']['status'] == 'accepted' and len(requests) == 1
    actual = requests[0]
    assert json.loads(actual['state']) == expected
    assert actual['questions'] == original['questions']
    assert _digest(actual) == 'f6255179593cc69e31c77198a64118da1c333c844a74d87eb0a04106bf66adca'
    assert result['receipt']['context_sha256'] == _digest(context)
    assert result['receipt']['intent_bank_sha256'] == _digest(bank)


_ORNITH088_REQUEST = (
    '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'
    'QUXy/wBO+MYu'
)
