"""Characterize actual fit diagnostics on archived initial B40 observations.

Scripted optimizer reports below test diagnostic contracts, not convergence or
historical fitted states. The real library optimizer is also exercised.
"""
from contextlib import ExitStack
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import random
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from botorch.exceptions.errors import ModelFittingError
from botorch.optim.core import OptimizationStatus
from linear_operator.utils.errors import NotPSDError

from experiments.paper.exploration import scalar_primary_replay as replay
from mutaco.orchestration import runner
from mutaco.surrogate import exact_multitask as gp
from mutaco import GPFitStatus, GPFitFailureKind, InvalidPosteriorError


CASES=("success","iteration_limit","nonfinite","before_callback","after_callback",
       "caller_callback","invalid_parameters","invalid_posterior","warm","fixed",
       "real_optimizer","recovery_success","recovery_exhausted")


def _tensor(value):
    return None if value is None else {"shape":list(value.shape),"dtype":str(value.dtype),
        "device":str(value.device),"bytes":value.detach().contiguous().cpu().numpy().tobytes().hex()}


def _rng():return random.getstate(),np.random.get_state(),torch.random.get_rng_state().clone()


def _assert_rng(before):
    after=np.random.get_state()
    assert random.getstate()==before[0] and before[1][0]==after[0]
    assert np.array_equal(before[1][1],after[1]) and before[1][2:]==after[2:]
    assert torch.equal(before[2],torch.random.get_rng_state())


def _canonical(value):
    if isinstance(value,dict):return {k:_canonical(v) for k,v in value.items() if k not in {"execution","timings"} and not k.endswith("_seconds")}
    if isinstance(value,(tuple,list)):return [_canonical(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return {"nonfinite":value.hex()}
    return value


def _capture(case,value):
    if location:=os.environ.get("MUTACO_GP_DIAGNOSTICS_CAPTURE_DIR"):
        output=Path(location).resolve()
        assert output.is_absolute() and not output.is_relative_to(Path(__file__).resolve().parents[2])
        output.mkdir(parents=True,exist_ok=True)
        (output/(case+'.json')).write_text(json.dumps(_canonical({
            "origin":"reconstructed archived-input characterization; scripted statuses are not historical fits",**value}),sort_keys=True,indent=2,allow_nan=False)+'\n')


@pytest.fixture(autouse=True)
def no_physical_simulator(monkeypatch):
    from mutaco_graph import OverdampedScientificSimulator
    attempts=[]
    def forbidden(*args,**kwargs):attempts.append(True);raise AssertionError('physical simulation forbidden')
    for name in ('evaluate','evaluate_batch','evaluate_condition_grid'):monkeypatch.setattr(OverdampedScientificSimulator,name,forbidden)
    yield
    assert not attempts


@pytest.fixture(scope='module')
def archived_initial_training():
    # Stop before fitting: production collection verifies every initial point
    # and context before returning its archived response. No adaptive response.
    objects={}
    class CaptureComplete(BaseException):pass
    def capture(data,*,config,**kwargs):objects.update(data=data,config=config);raise CaptureComplete
    before=_rng()
    with patch.object(runner,'fit_exact_multitask_gp_with_recovery',capture):
        with pytest.raises(CaptureComplete):replay.replay_reference(40,16)
    _assert_rng(before)
    data,config=objects['data'],objects['config']
    assert data.theta.shape==(16,4) and data.response_standardized.shape==(16,20)
    assert data.theta.dtype is torch.float32 and config.max_iterations==12
    return data,config


def _model_state(model):
    return {name:{"value":_tensor(value),"version":value._version,
                  "gradient":_tensor(value.grad),"gradient_object":None if value.grad is None else id(value.grad)}
            for name,value in model.named_parameters()}


@pytest.mark.parametrize('case',CASES)
def test_fit_optimizer_diagnostic_lifecycle(monkeypatch,archived_initial_training,case):
    data,config=archived_initial_training
    seen={'calls':[],'models':[],'registrations':[],'reports':[],'documents':[],
          'callbacks':[],'covariances':[],'before_documents':None,'optimizer_calls':0}
    build,convert,covariance,posterior=gp._build_model,gp._optimizer_result_provenance,gp._training_covariance_diagnostics,gp.ExactMultiTaskGPSurrogate.posterior
    def building(*args,**kwargs):
        model=build(*args,**kwargs);seen['calls'].append('build');seen['models'].append(model)
        priors=[(n,type(v).__name__) for n,_,v,*_ in model.named_priors()]
        constraints=[(n,type(v).__name__) for n,v in model.named_constraints()]
        assert priors==[('likelihood.noise_covar.noise_prior','LogNormalPrior'),
                        ('covar_module.kernels.0.lengthscale_prior','LogNormalPrior'),
                        ('covar_module.kernels.1.IndexKernelPrior','BetaPrior')]
        assert ('covar_module.kernels.1.raw_var_constraint','_FloorPreservingPositive') in constraints
        assert ('covar_module.kernels.1.raw_covar_factor_constraint','GreaterThan') in constraints
        seen['registrations'].append({'priors':priors,'constraints':constraints,'initial_parameters':{n:_tensor(v) for n,v in model.named_parameters()}})
        return model
    def converting(value):
        seen['calls'].append('document')
        result=convert(value);seen['documents'].append(result)
        seen['before_documents']=[_model_state(m) for m in seen['models']]
        return result
    def covariance_spy(*args,**kwargs):
        seen['calls'].append('covariance');result=covariance(*args,**kwargs);seen['covariances'].append(result);return result
    def posterior_spy(*args,**kwargs):
        seen['calls'].append('posterior')
        if case=='invalid_posterior':raise InvalidPosteriorError('controlled posterior rejection')
        return posterior(*args,**kwargs)
    def caller(parameters,result):
        seen['calls'].append('caller');seen['callbacks'].append(result)
        if case=='caller_callback':raise ValueError('controlled caller failure')
    def optimizer(mll,**kwargs):
        seen['calls'].append('optimizer');index=seen['optimizer_calls'];seen['optimizer_calls']+=1
        assert torch.is_grad_enabled() and kwargs['options']=={'maxiter':12}
        if case=='before_callback' or case=='recovery_exhausted' or (case=='recovery_success' and index==0):
            raise NotPSDError('controlled matrix not positive definite')
        value=gp.get_loss_closure(mll)().sum();value.backward()
        assert torch.isfinite(value) and any(p.grad is not None for p in mll.model.parameters())
        status=OptimizationStatus.STOPPED if case=='iteration_limit' else OptimizationStatus.FAILURE if case=='nonfinite' else OptimizationStatus.SUCCESS
        fval=math.nan if case=='nonfinite' else 1.25
        if case=='recovery_success':fval=(1.25,1.25,2.,3.)[index-1]
        report=SimpleNamespace(status=status,step=12 if case=='iteration_limit' else 1,fval=fval,runtime=.125,
            message='TOTAL NO. OF ITERATIONS REACHED LIMIT' if case=='iteration_limit' else 'scripted optimizer result')
        seen['reports'].append(report)
        kwargs['callback']({},report)
        if case=='after_callback':
            with torch.no_grad():next(mll.model.parameters()).add_(.01)
            raise NotPSDError('controlled matrix not positive definite')
        if case=='invalid_parameters':
            with torch.no_grad():next(mll.model.parameters()).fill_(math.nan)
        return report
    def fitting(mll,**kwargs):
        seen['calls'].append('fit')
        assert kwargs['max_attempts']==1 and kwargs['optimizer_kwargs']['options']=={'maxiter':12}
        try:kwargs['optimizer'](mll,callback=caller,**kwargs['optimizer_kwargs'])
        except Exception as exc:raise ModelFittingError('controlled wrapper rejection') from exc
        if case in {'iteration_limit','nonfinite'}:raise ModelFittingError('controlled wrapper rejection')
        return mll
    monkeypatch.setattr(gp,'_build_model',building);monkeypatch.setattr(gp,'_optimizer_result_provenance',converting)
    monkeypatch.setattr(gp,'_training_covariance_diagnostics',covariance_spy)
    monkeypatch.setattr(gp.ExactMultiTaskGPSurrogate,'posterior',posterior_spy)
    if case!='real_optimizer':
        monkeypatch.setattr(gp,'_FIT_MLL',fitting);monkeypatch.setattr(gp,'fit_gpytorch_mll_scipy',optimizer)
    before=_rng();snapshots=[(_tensor(v),v._version) for v in (data.theta,data.response_standardized)]
    warm=None
    if case in {'warm','fixed'}:
        initial=gp.fit_exact_multitask_gp(data,config=config)
        assert initial.surrogate is not None
        warm=initial.surrogate.warm_start_state()
        warm_bits={n:_tensor(t) for n,t in warm.parameter_state.items()}
        seen['calls'].clear();seen['callbacks'].clear();seen['reports'].clear();seen['documents'].clear();seen['optimizer_calls']=0
    if case.startswith('recovery_'):
        result=gp.fit_exact_multitask_gp_with_recovery(data,config=config,training_version=0)
    else:
        result=gp.fit_exact_multitask_gp(data,config=config,warm_start_state=warm,
            initialization_kind='condition_previous_gp_on_current_data_v1' if case=='fixed' else None,optimize=case!='fixed')
    _assert_rng(before)
    for value,(bits,version) in zip((data.theta,data.response_standardized),snapshots):assert _tensor(value)==bits and value._version==version
    assert seen['before_documents']==[_model_state(m) for m in seen['models']]
    if warm is not None:assert {n:_tensor(t) for n,t in warm.parameter_state.items()}==warm_bits
    diagnostic=result.diagnostic;document=diagnostic.optimizer_result
    assert diagnostic.training_data_id==data.identity() and diagnostic.configuration==config.provenance()
    assert diagnostic.configuration['max_iterations']==12
    assert diagnostic.recovery_attempts
    if not case.startswith('recovery_'):
        attempt=diagnostic.recovery_attempts[0]
        assert attempt['optimizer_result'] is document
        assert attempt['initialization']['attempt_seed']==config.fit_seed
        assert diagnostic.model_id==(None if result.surrogate is None else result.surrogate.model_id)
    expected_failure={'iteration_limit':GPFitFailureKind.ITERATION_EXHAUSTED,'nonfinite':GPFitFailureKind.NONFINITE_OBJECTIVE,
        'before_callback':GPFitFailureKind.COVARIANCE_REJECTED,'after_callback':GPFitFailureKind.COVARIANCE_REJECTED,
        'caller_callback':GPFitFailureKind.OPTIMIZER_FAILURE,'invalid_parameters':GPFitFailureKind.INVALID_PARAMETERS,
        'invalid_posterior':GPFitFailureKind.COVARIANCE_REJECTED,'recovery_exhausted':GPFitFailureKind.INITIAL_GP_FIT_FAILURE}
    assert diagnostic.status is (GPFitStatus.FAILED if case in expected_failure else GPFitStatus.SUCCESS)
    assert diagnostic.failure_kind is expected_failure.get(case)
    if case=='before_callback':assert document is None and diagnostic.validation_posterior_calls==0
    elif document is not None:
        assert document['configured_max_iterations']==12 and isinstance(document['initial_objective'],float)
        if case=='nonfinite':assert not document['objective_finite'] and document['objective_improvement'] is None
        else:assert document['objective_improvement']==document['initial_objective']-document['final_objective']
    if case in {'after_callback','caller_callback'}:
        # Existing nested local shadows the outer diagnostic value; preserve it.
        assert document['last_valid_covariance'] is None
        assert len(seen['covariances'])==2 and document['failure_covariance'] is seen['covariances'][0]
        assert seen['calls'].index('caller')<seen['calls'].index('covariance')<seen['calls'].index('document')
    if case=='fixed':
        assert seen['optimizer_calls']==0 and not seen['callbacks'] and document['status']=='not_run_fixed_conditioning'
        assert document['actual_iterations']==0 and document['objective_improvement']==0 and diagnostic.validation_posterior_calls==1
    if case=='iteration_limit':assert document['actual_iterations']==12 and document['stopped']
    if case=='nonfinite':assert diagnostic.recovery_attempts[0]['marginal_likelihood'] is None
    if case in {'recovery_success','recovery_exhausted'}:
        attempts=diagnostic.recovery_attempts;assert len(attempts)==seen['optimizer_calls']==5
        assert [a['initialization']['kind'] for a in attempts]==['deterministic_cold']+['coordinate_seeded_restart']*4
        if case=='recovery_success':assert [a['selected'] for a in attempts]==[False,True,False,False,False]
        else:assert diagnostic.failure_stage=='initial_gp_fit_recovery'
    if case not in {'real_optimizer','before_callback','recovery_exhausted','fixed'}:
        for report in seen['reports']:assert set(report.__dict__)=={'status','step','fval','runtime','message'}
    posterior_document=None
    if result.surrogate is not None:
        points=data.theta[:2].detach().clone().requires_grad_(True)
        summary=result.surrogate.posterior(points)
        assert summary.mean.shape==(2,20) and summary.covariance.shape==(2,20,20)
        assert summary.mean.dtype is torch.float32 and summary.mean.device.type=='cpu'
        gradient,=torch.autograd.grad(summary.mean.sum(),points)
        assert torch.isfinite(gradient).all()
        posterior_document={'mean':_tensor(summary.mean),'covariance':_tensor(summary.covariance),'gradient':_tensor(gradient)}
    # Summarization returns the same freshly converted dict, preserving aliases.
    if document is not None and case not in {'fixed','recovery_success','recovery_exhausted'}:assert document is seen['documents'][-1]
    _assert_rng(before)
    states=[{name:{k:v for k,v in entry.items() if k!='gradient_object'} for name,entry in _model_state(model).items()} for model in seen['models']]
    _capture(case,{'diagnostic':runner.FitRecord(0,diagnostic).provenance(),'registrations':seen['registrations'],'model_states':states,
        'training_data_id':data.identity(),'theta':_tensor(data.theta),'standardized_responses':_tensor(data.response_standardized),
        'posterior':posterior_document,'calls':seen['calls'],'optimizer_calls':seen['optimizer_calls'],
        'callback_reports':[{'status':r.status.name,'step':r.step,'fval':r.fval} for r in seen['callbacks']],
        'rng_preserved':True,'training_and_warm_state_unmodified':True})
