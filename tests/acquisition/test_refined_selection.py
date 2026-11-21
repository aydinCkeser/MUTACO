"""Observe the executed selector, using real screening and scripted refinements.

Defensive malformed-result fixtures exercise existing guards; they do not claim
that the current numerical optimizers normally produce these results.
"""
import ast
from contextlib import contextmanager
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

from mutaco import (LatentPosteriorSummary, ParameterAxis, ParameterSpec,
                    ProposalConfig, ScientificIdentity, SeedCoordinate,
                    SeedDeriver, StandardizedTarget)
from mutaco.acquisition import proposal as p

# Fixed winner after each restart; initial None is the loop-entry state.
CASES = {
    "improved": (["better", "equal", "worse"], [None, 0, 0, 0]),
    "equal": (["equal", "worse", "worse"], [None, 0, 0, 0]),
    "worse": (["worse", "worse", "worse"], [None, None, None, None]),
    "multiple": (["better", "best", "better"], [None, 0, 1, 1]),
    "tie": (["better", "better", "better"], [None, 0, 0, 0]),
    "late": (["worse", "better", "best"], [None, None, 1, 2]),
    "all_failed": (["failed", "failed", "failed"], [None, None, None, None]),
    "terminated": (["terminated", "better", "worse"], [None, 0, 0, 0]),
    "missing_candidate": (["no_candidate", "better", "worse"], [None, None, 1, 1]),
    "missing_score": (["no_score", "equal", "worse"], [None, None, 1, 1]),
    "nonfinite_score": (["nan_score", "inf_score", "negative_inf_score"], [None, None, None, None]),
    "nonfinite_candidate": (["nan_candidate", "inf_candidate", "negative_inf_candidate"], [None, None, None, None]),
    "bounds": (["below", "above", "boundary"], [None, None, None, 2]),
    "ulp_threshold": (["worse_ulp", "equal", "better_ulp"], [None, None, 1, 2]),
    "duplicate": (["better", "equal", "worse"], [None, 0, 0, 0]),
    "failed_short_circuit": (["failed_nan", "better", "worse"], [None, None, 1, 1]),
    "mixed_invalid": (["no_candidate", "failed_nan", "no_score"], [None, None, None, None]),
}
MODES = ("barrier", "exploration", "targeted_exploration")


@pytest.fixture(autouse=True)
def no_physical_simulator(monkeypatch):
    from mutaco_graph import OverdampedScientificSimulator
    attempts = []
    def forbidden(*args, **kwargs):
        attempts.append(True)
        raise AssertionError("physical simulator forbidden in selection tests")
    for name in ("evaluate", "evaluate_batch", "evaluate_condition_grid"):
        monkeypatch.setattr(OverdampedScientificSimulator, name, forbidden)
    yield
    assert not attempts


def _rng():
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()


def _assert_rng(before):
    after = np.random.get_state()
    assert random.getstate() == before[0]
    assert before[1][0] == after[0] and np.array_equal(before[1][1], after[1])
    assert before[1][2:] == after[2:]
    assert torch.equal(before[2], torch.random.get_rng_state())


def _tensor(value):
    return None if value is None else {
        "shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device),
        "bytes": value.detach().contiguous().cpu().numpy().tobytes().hex(),
    }


def _capture(name, mode, dtype, document):
    if location := os.environ.get("MUTACO_SELECTION_CAPTURE_DIR"):
        output = Path(location).resolve()
        assert output.is_absolute() and not output.is_relative_to(Path(__file__).resolve().parents[2])
        def encode(value):
            if isinstance(value, float) and not math.isfinite(value): return {"nonfinite": value.hex()}
            if isinstance(value, dict): return {k: encode(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)): return [encode(v) for v in value]
            return value
        output.mkdir(parents=True, exist_ok=True)
        (output/f"{name}-{mode}-{dtype}.json").write_text(json.dumps(encode({
            "origin": "controlled selector characterization, not historical optimizer states", **document,
        }), indent=2, sort_keys=True, allow_nan=False)+"\n")


@contextmanager
def _observe_selection(seen):
    # Observe the existing Python statements, without wrapping or reimplementing
    # the selector. Find the same loop before and after moving it to a helper.
    tree = ast.parse(Path(p.__file__).read_text())
    functions = [f for f in tree.body if isinstance(f, ast.FunctionDef) and any(
        isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
        and n.target.id == "selected_outcome" for n in f.body)]
    assert len(functions) == 1
    function = functions[0]
    loop = next(n for n in function.body if isinstance(n, ast.For)
                and isinstance(n.iter, ast.Name) and n.iter.id == "outcomes")
    threshold = next(n for n in loop.body if isinstance(n, ast.If)
                     and ast.unparse(n.test) == "not nonworsening")
    improvement = next(n for n in loop.body if isinstance(n, ast.If)
                       and ast.unparse(n.test) == "strictly_better")
    def index(value):
        return None if value is None else next(i for i, item in enumerate(seen["outcomes"]) if item is value)
    def trace(frame, event, arg):
        if event == "line" and frame.f_code.co_filename == p.__file__ and frame.f_code.co_name == function.name:
            state = frame.f_locals
            if frame.f_lineno == loop.lineno:
                seen["winners"].append(index(state["selected_outcome"]))
            elif frame.f_lineno == threshold.lineno:
                seen["thresholds"].append((index(state["outcome"]), state["nonworsening"]))
            elif frame.f_lineno == improvement.lineno:
                seen["improvements"].append((index(state["outcome"]), index(state["selected_outcome"]), state["strictly_better"]))
        return trace
    before = sys.gettrace()
    sys.settrace(trace)
    try: yield
    finally: sys.settrace(before)


def _setup(monkeypatch, dtype, mode, recipes, *, fault=None):
    minimize = mode != "exploration"
    means = torch.stack((torch.full((4,), 1.0 if mode == "barrier" else 8.0, dtype=dtype), torch.zeros(4, dtype=dtype)), -1)
    covariance = torch.diag_embed(torch.stack((torch.tensor([1.,4.,4.,2.], dtype=dtype), torch.ones(4,dtype=dtype)), -1))
    posterior = LatentPosteriorSummary(means, covariance, "selection-transform", "selection-model")
    target = StandardizedTarget(torch.zeros(2,dtype=dtype), ("a","b"), posterior.transform_id)
    cfg = ProposalConfig.from_seed_deriver(
        SeedDeriver(ScientificIdentity("refined-selection-characterization"),193),
        SeedCoordinate("raw_candidates", ("round",0)), beta=1.,alpha=.5,tau_interior=0.,
        raw_samples=4,restarts=3,max_iterations=1,
        **({"infeasible_fallback_policy":"minimum_normalized_target_miss_diagnostic_v1"} if mode=="targeted_exploration" else {}))
    seen = {"outcomes":[],"starts":[],"winners":[],"thresholds":[],"improvements":[],
            "outcome_snapshots":[],"finite_checks":[],"posterior_calls":0,"duplicates":0,"identities":0,
            "fault":MemoryError("controlled "+str(fault)+" allocation failure")}
    class Provider:
        def __init__(self):
            self.model_id,self.transform_id=posterior.model_id,posterior.transform_id
            self.parameter_dim,self.dtype,self.device=2,dtype,torch.device("cpu")
        def posterior(self, theta):
            assert theta.shape==(4,2)
            seen["posterior_calls"]+=1
            return posterior
    def restart(provider,start,*,raw_index,restart_index,initial_score,**kwargs):
        i=restart_index;kind=recipes[i]
        seen["starts"].append((raw_index,start,initial_score))
        if fault=="optimizer" and i==1: raise seen["fault"]
        if i==0:seen["raw_best_score"]=initial_score.detach().clone()
        base=seen["raw_best_score"];sign=-1 if minimize else 1
        score=base.clone()+sign*(2 if kind in {"best","terminated"} else -1 if kind=="worse" else 0 if kind=="equal" else 1)
        if kind.endswith("_ulp"):
            direction=sign if kind=="better_ulp" else -sign
            score=torch.nextafter(base,torch.full_like(base,math.inf*direction))
        if kind in {"nan_score","inf_score","negative_inf_score","failed_nan"}:
            score=torch.full_like(base,{"nan_score":math.nan,"inf_score":math.inf,"negative_inf_score":-math.inf,"failed_nan":math.nan}[kind])
        candidate=torch.tensor([.15+.1*i,.25+.05*i],dtype=dtype)
        if kind in {"nan_candidate","inf_candidate","negative_inf_candidate","failed_nan"}:
            candidate[0]={"nan_candidate":math.nan,"inf_candidate":math.inf,"negative_inf_candidate":-math.inf,"failed_nan":math.nan}[kind]
        if kind=="below":candidate[0]=-torch.finfo(dtype).eps
        if kind=="above":candidate[0]=torch.nextafter(torch.tensor(1.,dtype=dtype),torch.tensor(math.inf,dtype=dtype))
        if kind=="boundary":candidate=torch.tensor([0.,1.],dtype=dtype)
        if kind=="no_candidate":candidate=None
        if kind=="no_score":score=None
        status=p.RestartStatus.FAILED if kind in {"failed","failed_nan"} else p.RestartStatus.TERMINATED if kind=="terminated" else p.RestartStatus.COMPLETED
        diagnostic=p.RestartDiagnostic(i,raw_index,status,
            p.RestartStopReason.POSTERIOR_ERROR if status is not p.RestartStatus.COMPLETED else p.RestartStopReason.MAX_ITERATIONS,
            1,0,0,0,float(initial_score),None if score is None else float(score),"scripted result")
        result=p._RestartOutcome(candidate,score,diagnostic)
        seen["outcomes"].append(result)
        seen["outcome_snapshots"].append([(_tensor(t), None if t is None else t._version) for t in (candidate,score)])
        return result
    isfinite=p.torch.isfinite
    def finite(value):
        for i,item in enumerate(seen["outcomes"]):
            if value is item.score or value is item.candidate:
                kind="score" if value is item.score else "candidate"
                seen["finite_checks"].append((i,kind))
                if fault=="selection" and i==1 and kind=="score":raise seen["fault"]
        return isfinite(value)
    duplicate,identity=p._duplicate_diagnostic,p._candidate_identity
    def duplicates(*args,**kwargs):seen["duplicates"]+=1;return duplicate(*args,**kwargs)
    def identities(*args,**kwargs):seen["identities"]+=1;return identity(*args,**kwargs)
    monkeypatch.setattr(p,"_barrier_restart",restart);monkeypatch.setattr(p,"_exploration_restart",restart)
    monkeypatch.setattr(p.torch,"isfinite",finite)
    monkeypatch.setattr(p,"_duplicate_diagnostic",duplicates);monkeypatch.setattr(p,"_candidate_identity",identities)
    return Provider(),target,cfg,seen


@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
@pytest.mark.parametrize("mode",MODES)
@pytest.mark.parametrize("case",CASES)
def test_refined_selection_decisions_ownership_identity_and_fallback(monkeypatch,dtype,mode,case):
    recipes,winners=CASES[case]
    provider,target,cfg,seen=_setup(monkeypatch,dtype,mode,recipes)
    evaluated=torch.tensor([[.15,.25] if case=="duplicate" else [.9,.9]],dtype=dtype)
    spec=ParameterSpec((ParameterAxis("x",0.,1.),ParameterAxis("y",0.,1.)))
    config=deepcopy(cfg.scientific_document());before=_rng()
    inputs=[target.values,evaluated]
    input_snapshots=[(_tensor(t),t._version) for t in inputs]
    with _observe_selection(seen):
        result=p.propose_candidate(provider,parameter_spec=spec,target=target,evaluated_theta=evaluated,config=cfg)
    _assert_rng(before)
    assert seen["winners"]==winners
    for value,(bits,version) in zip(inputs,input_snapshots):
        assert _tensor(value)==bits and value._version==version
    for outcome,snapshots in zip(seen["outcomes"],seen["outcome_snapshots"]):
        for value,(bits,version) in zip((outcome.candidate,outcome.score),snapshots):
            assert _tensor(value)==bits and (value is None or value._version==version)
    assert [i for i,_ in seen["thresholds"]]==[i for i,k in enumerate(recipes) if k in {"better","best","equal","worse","terminated","boundary","better_ulp","worse_ulp"}]
    assert [ok for i,ok in seen["thresholds"]]==[recipes[i] not in {"worse","worse_ulp"} for i,_ in seen["thresholds"]]
    expected_improvements={"improved":[(1,0,False)],"multiple":[(1,0,True),(2,1,False)],
        "tie":[(1,0,False),(2,0,False)],"late":[(2,1,True)],"terminated":[(1,0,False)],
        "ulp_threshold":[(2,1,True)],"duplicate":[(1,0,False)]}.get(case,[])
    assert seen["improvements"]==expected_improvements
    expected_checks=[]
    for i,kind in enumerate(recipes):
        if kind in {"failed","failed_nan","no_candidate","no_score"}:continue
        expected_checks.append((i,"score"))
        if kind not in {"nan_score","inf_score","negative_inf_score"}:expected_checks.append((i,"candidate"))
    assert seen["finite_checks"]==expected_checks
    chosen=winners[-1]
    assert result.status is p.ProposalStatus.SUCCESS and result.failure_reason is None
    if chosen is not None:
        assert result.candidate_source is p.ReturnedCandidateSource.REFINED
        assert result.candidate is seen["outcomes"][chosen].candidate
        assert result.fallback_reason is result.fallback_explanation is None
    else:
        assert result.candidate_source is p.ReturnedCandidateSource.RAW_FALLBACK
        assert result.candidate._base is None and result.candidate.is_contiguous()
        assert result.candidate.untyped_storage().data_ptr()!=result.pool.theta.untyped_storage().data_ptr()
        assert torch.equal(result.candidate,result.pool.theta[result.raw_rankings[0].raw_index])
        all_failed=case=="all_failed"
        assert result.fallback_reason is (p.FallbackReason.ALL_RESTARTS_FAILED if all_failed else p.FallbackReason.NO_VALID_NONWORSENING_REFINEMENT)
        assert result.fallback_explanation==("every configured restart failed; returned the best ranked raw candidate" if all_failed else "no restart produced a valid non-worsening refined candidate")
    assert result.candidate.shape==(2,) and result.candidate.dtype is dtype and result.candidate.device.type=="cpu"
    assert seen["posterior_calls"]==1 and result.posterior_calls.total==1 and result.posterior_calls.per_restart==(0,0,0)
    assert result.simulator_calls==0 and seen["duplicates"]==seen["identities"]==1
    assert result.branch is (p.AcquisitionBranch.BARRIER if mode=="barrier" else p.AcquisitionBranch.EXPLORATION)
    assert result.branch_direction==("maximize" if mode=="exploration" else "minimize")
    assert result.model_id==provider.model_id and result.transform_id==provider.transform_id
    assert result.configuration==config and [r.raw_index for r in result.restarts]==[1,2,3]
    assert [r.restart_index for r in result.restarts]==[0,1,2]
    assert tuple(o.diagnostic for o in seen["outcomes"])==result.restarts
    for index,start,score in seen["starts"]:
        assert start.untyped_storage().data_ptr()==result.pool.theta.untyped_storage().data_ptr() and start.storage_offset()==index*2
        assert start.dtype is dtype and score.dtype is dtype
    assert result.duplicate.exact_duplicate==(case=="duplicate") and not result.duplicate.provenance()["selection_changed"]
    assert result.duplicate_recovery is None
    regenerated=p.generate_candidate_pool(parameter_spec=spec,config=cfg,dtype=dtype,device=torch.device("cpu"))
    assert regenerated.pool_id==result.pool.pool_id and torch.equal(regenerated.theta,result.pool.theta)
    assert result.pool.seed_value==cfg.candidate_seed and result.pool.seed_coordinate==cfg.seed_coordinate
    _assert_rng(before)
    document=result.provenance();document.pop("timings");document.pop("execution")
    _capture(case,mode,str(dtype),{"result":document,"candidate":_tensor(result.candidate),
        "outcomes":[{"candidate":_tensor(o.candidate),"score":_tensor(o.score),"diagnostic":o.diagnostic.provenance()} for o in seen["outcomes"]],
        "winner_trace":seen["winners"],"threshold_checks":seen["thresholds"],"improvement_checks":seen["improvements"],"finite_check_order":seen["finite_checks"],"rng_preserved":True})


@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
@pytest.mark.parametrize("mode",MODES)
@pytest.mark.parametrize("fault",["optimizer","selection"])
def test_refined_selection_error_propagation_precedes_identity_and_duplicate(monkeypatch,dtype,mode,fault):
    provider,target,cfg,seen=_setup(monkeypatch,dtype,mode,["better"]*3,fault=fault)
    before=_rng()
    with _observe_selection(seen),pytest.raises(MemoryError) as caught:
        p.propose_candidate(provider,parameter_spec=ParameterSpec((ParameterAxis("x",0.,1.),ParameterAxis("y",0.,1.))),
            target=target,evaluated_theta=torch.tensor([[.9,.9]],dtype=dtype),config=cfg)
    assert caught.value is seen["fault"] and seen["duplicates"]==seen["identities"]==0
    assert seen["posterior_calls"]==1 and len(seen["starts"])==(2 if fault=="optimizer" else 3)
    assert seen["winners"]==([] if fault=="optimizer" else [None,0])
    _assert_rng(before)
    _capture("error-"+fault,mode,str(dtype),{"exception":str(caught.value),"winner_trace":seen["winners"],
        "threshold_checks":seen["thresholds"],"finite_check_order":seen["finite_checks"],
        "completed_outcomes":len(seen["outcomes"]),"duplicate_calls":0,"identity_calls":0,"rng_preserved":True})
