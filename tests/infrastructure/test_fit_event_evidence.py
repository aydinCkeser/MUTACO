from copy import deepcopy
import pytest
from infrastructure.truba.fit_event_evidence import compare_fit_events,expected_fit_events


def example():
    history={"reconstructed_gp_fits":[{"training_version":0,"status":"success","fit_id":"fit-0","model_id":"model-0"}]}
    events=[{"kind":"transform_fitted","phase":"initial","sequential_round":None}]
    events+=expected_fit_events(history)
    events+=[{"kind":"recommendation_started","phase":"recommendation","sequential_round":None}]
    for event in events:event['elapsed_seconds']=0.1
    events[2]['details']['elapsed_seconds']=0.05
    return history,{"origin":"observed candidate events during strict replay","events":events}


def test_fit_event_comparison_excludes_only_declared_timing():
    history,observed=example()
    result=compare_fit_events(history,observed)
    assert result['events_exact_equal'] and result['event_boundaries_preserved']
    observed['events'][2]['elapsed_seconds']=999.0
    assert compare_fit_events(history,observed)['events_exact_equal']


@pytest.mark.parametrize('change',['model','message','boundary'])
def test_fit_event_comparison_rejects_scientific_metadata_or_order(change):
    history,observed=example()
    if change=='model':observed['events'][2]['details']['model_id']='different'
    if change=='message':observed['events'][2]['message']='different'
    if change=='boundary':observed['events'][0]['kind']='simulator_attempt_started'
    result=compare_fit_events(history,observed)
    assert not (result['events_exact_equal'] and result['event_boundaries_preserved'])
