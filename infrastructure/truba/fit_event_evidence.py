"""Compare observed candidate GP events with explicitly reconstructed expectations.

Saved original fit records and the unchanged baseline emitter determine metadata
and order, not original historical event times. No GP calculations run here.
"""
from infrastructure.truba.refactor_reference import canonical,digest
from infrastructure.truba.b100_regression import first_difference


def expected_fit_events(history):
    result=[]
    for fit in history["reconstructed_gp_fits"]:
        version=fit["training_version"]
        for kind,message,details in (
            ("gp_fit_started",f"GP fit for training version {version} started",{}),
            ("gp_fit_finished",f"GP fit for training version {version}: {fit['status']}",
             {"fit_id":fit["fit_id"],"model_id":fit["model_id"]}),
        ):
            result.append({"kind":kind,"phase":"fit","logical_index":None,
                           "sequential_round":version,"message":message,"details":details})
    return result


def compare_fit_events(history,observed):
    events=observed["events"]
    expected=expected_fit_events(history)
    actual=canonical([event for event in events if event["phase"]=="fit"])
    boundaries=[]
    for i,event in enumerate(events):
        kind,version=event["kind"],event["sequential_round"]
        if kind=="gp_fit_started":
            before="transform_fitted" if version==0 else "simulator_attempt_finished"
            boundaries.append(i>0 and events[i-1]["kind"]==before
                              and i+1<len(events) and events[i+1]["kind"]=="gp_fit_finished")
        elif kind=="gp_fit_finished":
            after="recommendation_started" if version==len(history["reconstructed_gp_fits"])-1 else "proposal_started"
            boundaries.append(i>0 and events[i-1]["kind"]=="gp_fit_started"
                              and i+1<len(events) and events[i+1]["kind"]==after)
    equal=actual==expected
    return {"expected_origin":"reconstructed from saved original fit records and baseline emitter; not archived historical event logs",
            "candidate_origin":observed["origin"],"events_exact_equal":equal,
            "event_boundaries_preserved":len(boundaries)==len(expected) and all(boundaries),
            "expected_event_count":len(expected),"observed_event_count":len(actual),
            "expected_sha256":digest(expected),"observed_sha256":digest(actual),
            "first_difference":first_difference(expected,actual),
            "excluded":"existing predeclared elapsed_seconds fields only; actual historical times unavailable"}
