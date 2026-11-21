from __future__ import annotations

from dataclasses import dataclass, replace
import random

import numpy as np
import pytest
import torch

from mutaco import (
    GPFitDiagnostic,
    GPFitFailureKind,
    GPFitResult,
    GPFitStatus,
    JointChiSquareResponseDimensionBeta,
    RetryableSimulationError,
    RetryableSimulationReason,
    RunnerStatus,
    ScientificIdentity,
    SeedDeriver,
    SequentialRunnerConfig,
    SequentialScientificIntent,
    SimulationResult,
    TerminalRecommendationContext,
    run_sequential,
    strict_metadata_document,
)
import mutaco.orchestration.runner as runner_module
import mutaco.recommendation.physical_l2 as physical_l2_module
import mutaco.recommendation.physical_mae as physical_mae_module
from experiments.benchmarking.methods.mutaco_projection import scientific_projection
from mutaco_analytic import HandwrittenAnalyticSimulator


def _intent(
    simulator: HandwrittenAnalyticSimulator,
    *,
    rounds: int = 1,
    root_seed: int = 4117,
) -> SequentialScientificIntent:
    capabilities = simulator.capabilities
    return SequentialScientificIntent(
        simulator_identity=capabilities.simulator_identity,
        simulator_definition_metadata=strict_metadata_document(
            capabilities.definition_metadata
        ),
        parameter_spec=capabilities.parameter_spec,
        response_spec=capabilities.response_spec,
        physical_target=(-0.1, 0.2, -0.15),
        initial_design_policy="scrambled_sobol",
        initial_point_count=4,
        caller_initial_points=None,
        sequential_rounds=rounds,
        transform_settings={"ddof": 1, "scale_floor": 1e-12},
        gp_settings={
            "likelihood": "task_diagonal_gaussian",
            "max_iterations": 2,
        },
        acquisition_settings={
            "raw_policy": "scrambled_sobol",
            "raw_samples": 8,
            "restarts": 1,
            "max_iterations": 1,
        },
        recommendation_policy="final_posterior_standardized_l1",
        dtype=torch.float64,
        root_seed=root_seed,
        seed_coordinate_schema={
            "initial_design": ["phase", "repeat"],
            "simulator_evaluation": ["purpose", "logical_index", "repeat"],
            "gp_fit": ["training_version"],
            "raw_candidates": ["sequential_round"],
        },
    )


def _science_projection(result) -> dict[str, object]:
    return {
        "status": result.status.value,
        "scientific_identity": result.scientific_identity.value,
        "initial_observation_ids": result.initial_design.observation_ids,
        "observations": tuple(
            {
                "id": item.observation_id,
                "theta": item.theta.tolist(),
                "response": item.response_physical.tolist(),
                "standardized": item.response_standardized.tolist(),
                "seed": item.context.evaluation_seed,
            }
            for item in result.observations
        ),
        "fit_ids": tuple(item.diagnostic.fit_id for item in result.fits),
        "model_ids": tuple(item.diagnostic.model_id for item in result.fits),
        "candidates": tuple(item.candidate.tolist() for item in result.proposals),
        "candidate_ids": tuple(item.candidate_id for item in result.proposals),
        "recommendation": result.recommendation.selected_observation_id,
        "logical": result.logical_accounting,
    }


def _assert_pre_run_stop(result, events, intent, *, status, explanation):
    assert result.status is status
    assert result.failure_explanation == explanation
    assert result.scientific_identity == intent.identity()
    assert result.configuration["scientific_intent"] == intent.scientific_document()
    assert result.capability_snapshot is not None
    assert result.initial_design is result.transform is result.standardized_target is None
    assert result.recommendation is result.terminal_surrogate is None
    assert result.observations == result.fits == result.proposals == ()
    assert result.observation_snapshots == {}
    assert result.progress_callback_failures == ()
    assert result.logical_accounting == {
        "planned_observations": intent.initial_point_count + intent.sequential_rounds,
        "committed_observations": 0, "initial_observations": 0,
        "sequential_observations": 0, "training_data_versions": 0,
        "committed_fit_events": 0, "committed_proposal_events": 0,
        "committed_recommendation_events": 0,
    }
    assert all(value == 0 for value in result.physical_accounting.values())
    assert all(value == 0 for key, value in result.timings.items() if key != "total_seconds")
    assert [(e.kind.value, e.phase, e.message, e.logical_index, e.sequential_round)
            for e in events] == [
        ("run_started", "setup", "sequential run started", None, None),
        ("run_finished", "terminal", f"run finished with status {status.value}", None, None),
    ]
    assert dict(events[0].details) == {
        "scientific_identity": intent.identity().value, "device": "cpu",
        "planned_observations": intent.initial_point_count + intent.sequential_rounds,
    }
    assert dict(events[1].details) == {"status": status.value}


def _pre_run_rng_state():
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()


def _assert_pre_run_rng_preserved(before):
    assert random.getstate() == before[0]
    after = np.random.get_state()
    assert after[0] == before[1][0] and np.array_equal(after[1], before[1][1])
    assert after[2:] == before[1][2:]
    assert torch.equal(torch.random.get_rng_state(), before[2])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("diagnostic_family", [None, "validation-diagnostic-family"])
def test_pre_run_validation_success_order_and_seeds(monkeypatch, dtype, diagnostic_family):
    """Stop at the initial-design boundary, after executing the real validators."""
    simulator = HandwrittenAnalyticSimulator(dtype=dtype)
    beta = JointChiSquareResponseDimensionBeta.resolve(coverage=0.95, response_dimension=3)
    intent = replace(_intent(simulator), dtype=dtype,
                     diagnostic_internal_seed_family_id=diagnostic_family)
    intent = replace(intent, acquisition_settings={
        **strict_metadata_document(intent.acquisition_settings),
        "beta": beta.value, "beta_policy": beta.document(),
    })
    order, configs, events = [], {}, []
    metadata = runner_module.strict_metadata_document
    def record_metadata(value):
        if value is intent.gp_settings:
            order.append("gp_metadata")
        elif value is intent.acquisition_settings:
            order.append("acquisition_metadata")
        return metadata(value)
    monkeypatch.setattr(runner_module, "strict_metadata_document", record_metadata)
    beta_validator = runner_module.JointChiSquareResponseDimensionBeta.from_document
    def record_beta(*args, **kwargs):
        order.append("beta")
        return beta_validator(*args, **kwargs)
    monkeypatch.setattr(runner_module.JointChiSquareResponseDimensionBeta, "from_document", record_beta)
    for name in ("GPFitConfig", "ProposalConfig"):
        cls = getattr(runner_module, name)
        constructor = cls.from_seed_deriver
        def record_config(*args, _name=name, _constructor=constructor, **kwargs):
            order.append(_name)
            configs[_name] = _constructor(*args, **kwargs)
            return configs[_name]
        monkeypatch.setattr(cls, "from_seed_deriver", record_config)
    policy = runner_module.RecommendationPolicy
    def record_policy(value):
        order.append("recommendation")
        return policy(value)
    monkeypatch.setattr(runner_module, "RecommendationPolicy", record_policy)
    def stop(config, deriver):
        order.append("initial_design")
        expected = SeedDeriver(ScientificIdentity(diagnostic_family) if diagnostic_family else intent.identity(), intent.root_seed)
        gp, proposal = configs["GPFitConfig"], configs["ProposalConfig"]
        assert gp.fit_seed_coordinate == runner_module.SeedCoordinate("gp_fit", ("training_version", 0))
        assert proposal.seed_coordinate == runner_module.SeedCoordinate("raw_candidates", ("sequential_round", 0))
        assert gp.fit_seed == expected.seed_for(gp.fit_seed_coordinate) == deriver.seed_for(gp.fit_seed_coordinate)
        assert proposal.candidate_seed == expected.seed_for(proposal.seed_coordinate)
        raise RuntimeError("characterization boundary")
    monkeypatch.setattr(runner_module, "_initial_design", stop)
    before = _pre_run_rng_state()
    result = run_sequential(simulator, config=SequentialRunnerConfig(intent), progress_callback=events.append)
    # ProposalConfig also resolves beta metadata internally; keep that second call.
    assert order == ["gp_metadata", "acquisition_metadata", "beta", "GPFitConfig",
                     "ProposalConfig", "beta", "recommendation", "initial_design"]
    _assert_pre_run_stop(result, events, intent, status=RunnerStatus.INITIAL_DESIGN_FAILED,
                         explanation="RuntimeError: characterization boundary")
    _assert_pre_run_rng_preserved(before)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("invalid,expected", [
    ("beta_mapping", "ValueError: beta_policy must be a strict mapping"),
    ("beta_dimension", "ValueError: beta response dimension does not match simulator response dimension"),
    ("gp", "ValueError: max_iterations must be a positive integer"),
    ("proposal", "ValueError: raw_samples must be a positive integer"),
    ("gp_type", "TypeError: GPFitConfig.__init__() got an unexpected keyword argument 'unknown_option'"),
])
def test_pre_run_validation_first_failure(monkeypatch, dtype, invalid, expected):
    """Earlier invalid options win even when later validators would also fail."""
    simulator = HandwrittenAnalyticSimulator(dtype=dtype)
    intent = replace(_intent(simulator), dtype=dtype)
    gp = strict_metadata_document(intent.gp_settings)
    acquisition = strict_metadata_document(intent.acquisition_settings)
    acquisition["raw_samples"] = 0
    if invalid != "proposal":
        gp["max_iterations"] = 0
    if invalid == "gp_type":
        gp["unknown_option"] = 1
    if invalid == "beta_mapping":
        acquisition["beta_policy"] = []
    if invalid == "beta_dimension":
        acquisition["beta_policy"] = JointChiSquareResponseDimensionBeta.resolve(
            coverage=0.95, response_dimension=20).document()
    intent = replace(intent, gp_settings=gp, acquisition_settings=acquisition)
    def forbidden(*args, **kwargs):
        pytest.fail("invalid options must stop before initial design or simulation")
    monkeypatch.setattr(runner_module, "_initial_design", forbidden)
    monkeypatch.setattr(type(simulator), "evaluate", forbidden)
    events, before = [], _pre_run_rng_state()
    result = run_sequential(simulator, config=SequentialRunnerConfig(intent), progress_callback=events.append)
    _assert_pre_run_stop(result, events, intent, status=RunnerStatus.CONFIGURATION_INVALID, explanation=expected)
    _assert_pre_run_rng_preserved(before)


@pytest.mark.parametrize("stage", ["gp_metadata", "acquisition_metadata", "GPFitConfig", "ProposalConfig", "RecommendationPolicy"])
@pytest.mark.parametrize("error", [ValueError, TypeError, runner_module.SimulatorMetadataContractError])
def test_pre_run_validation_exception_boundary(monkeypatch, stage, error):
    """All three existing caught exception classes keep the committed snapshot."""
    simulator = HandwrittenAnalyticSimulator()
    intent = _intent(simulator)
    def fail(*args, **kwargs):
        raise error("characterized validation failure")
    if stage.endswith("metadata"):
        original = runner_module.strict_metadata_document
        selected = intent.gp_settings if stage == "gp_metadata" else intent.acquisition_settings
        def metadata(value):
            return fail() if value is selected else original(value)
        monkeypatch.setattr(runner_module, "strict_metadata_document", metadata)
    elif stage in {"GPFitConfig", "ProposalConfig"}:
        monkeypatch.setattr(getattr(runner_module, stage), "from_seed_deriver", fail)
    else:
        monkeypatch.setattr(runner_module, stage, fail)
    events, before = [], _pre_run_rng_state()
    result = run_sequential(simulator, config=SequentialRunnerConfig(intent), progress_callback=events.append)
    _assert_pre_run_stop(result, events, intent, status=RunnerStatus.CONFIGURATION_INVALID,
                         explanation=f"{error.__name__}: characterized validation failure")
    _assert_pre_run_rng_preserved(before)


@pytest.fixture
def initial_collection_tripwire():
    from contextlib import ExitStack
    from infrastructure.truba.refactor_reference import simulator_guard
    with ExitStack() as stack:
        attempts = simulator_guard(stack)
        yield
        assert attempts == []


class _InitialCollectionAdapter:
    """Deterministic test responses; no physical backend is called."""
    def __init__(self, dtype, *, fail_at=None, failure=None, retries=0):
        self.delegate = HandwrittenAnalyticSimulator(dtype=dtype)
        self.fail_at, self.failure, self.retries = fail_at, failure, retries
        self.calls, self.responses = [], []

    @property
    def capabilities(self):
        return self.delegate.capabilities

    def evaluate(self, theta, context=None):
        assert context is not None
        self.calls.append((theta.detach().clone(), context))
        if context.logical_index == self.fail_at:
            if self.failure == "runtime":
                raise RuntimeError("initial adapter failure")
            if self.failure == "metadata":
                raise runner_module.SimulatorMetadataContractError("initial metadata failure")
            if self.failure == "tensor":
                raise runner_module.TensorContractError("initial tensor failure")
            if self.failure == "retry" and context.attempt_index < self.retries:
                raise RetryableSimulationError(RetryableSimulationReason.TRANSIENT_RESOURCE, "initial retry")
        index = context.logical_index
        response = torch.tensor([index + 0.25, -index - 0.5, index + 0.75],
                                dtype=theta.dtype, device=theta.device)
        self.responses.append(response)
        return SimulationResult(response, {"stored_row": index})


def _stop_after_initial_collection(monkeypatch, *, reached):
    def stop(*args, **kwargs):
        assert reached, "failed collection must stop before frozen response preparation"
        return runner_module._FrozenResponsePreparation(None, None, "", "collection boundary")
    monkeypatch.setattr(runner_module, "_prepare_frozen_responses", stop)


def _assert_initial_collection_state(result, adapter, events, intent, *, committed,
                                     status, message, failures, retries, total_budget):
    assert result.status is status and result.failure_explanation == message
    assert result.scientific_identity == intent.identity()
    assert result.capability_snapshot is not None and result.initial_design is not None
    assert result.transform is result.standardized_target is result.terminal_surrogate is None
    assert result.recommendation is None and result.fits == result.proposals == ()
    assert result.observation_snapshots == {}
    assert result.logical_accounting == {
        "planned_observations": 4, "committed_observations": committed,
        "initial_observations": committed, "sequential_observations": 0,
        "training_data_versions": 0, "committed_fit_events": 0,
        "committed_proposal_events": 0, "committed_recommendation_events": 0,
    }
    expected = {key: 0 for key in result.physical_accounting if not key.endswith("seconds")}
    expected.update(simulator_attempts=len(adapter.calls), simulator_successes=committed,
                    simulator_failures=failures, simulator_retry_attempts=retries)
    assert {k:v for k,v in result.physical_accounting.items() if not k.endswith("seconds")} == expected
    assert result.physical_accounting["simulator_seconds"] >= 0
    for index, observation in enumerate(result.observations):
        assert observation.observation_id == result.initial_design.observation_ids[index]
        assert observation.phase.value == "initial"
        assert observation.logical_index == observation.phase_index == index
        assert observation.response_standardized is None
        assert torch.equal(observation.theta, result.initial_design.theta[index])
        assert torch.equal(observation.response_physical, adapter.responses[index])
        assert observation.response_physical.data_ptr() != adapter.responses[index].data_ptr()
        assert observation.theta.is_contiguous() and observation.response_physical.is_contiguous()
        assert observation.theta.dtype == observation.response_physical.dtype == intent.dtype
    deriver = SeedDeriver(intent.identity(), intent.root_seed)
    if intent.evaluation_seed_family_id is not None:
        deriver = SeedDeriver(ScientificIdentity(intent.evaluation_seed_family_id), intent.root_seed)
    for theta, context in adapter.calls:
        coordinate = runner_module.SeedCoordinate("simulator_evaluation", ("initial", context.logical_index, 0))
        seed = intent.fixed_evaluation_seed if intent.fixed_evaluation_seed is not None else deriver.seed_for(coordinate)
        assert context.evaluation_seed == seed and context.repeat_index == 0
        assert context.scientific_run_id == intent.identity().value and context.purpose == "initial"
        assert torch.equal(theta, result.initial_design.theta[context.logical_index])
    expected_events = [("run_started", "setup", None, "sequential run started"),
                       ("initial_design_ready", "initial", None, "initial design contains 4 points")]
    for call, (_, context) in enumerate(adapter.calls):
        index, attempt = context.logical_index, context.attempt_index
        expected_events.append(("simulator_attempt_started", "initial", index,
                                f"initial evaluation {index} attempt {attempt} started"))
        starts = [e for e in events if e.kind.value == "simulator_attempt_started"]
        assert dict(starts[call].details) == {"attempt_index": attempt, "total_attempts": call + 1,
                                            "total_budget": total_budget}
        retry = (index == adapter.fail_at and adapter.failure == "retry" and attempt < adapter.retries)
        failed = (index == adapter.fail_at and adapter.failure in {"runtime", "metadata", "tensor"})
        if retry or not failed:
            outcome = "requested a retry" if retry else "succeeded"
            expected_events.append(("simulator_attempt_finished", "initial", index,
                                    f"initial evaluation {index} {outcome}"))
    expected_events.append(("run_finished", "terminal", None, f"run finished with status {status.value}"))
    assert [(e.kind.value,e.phase,e.logical_index,e.message) for e in events] == expected_events
    assert all(e.sequential_round is None for e in events)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("policy", ["default", "paired", "fixed"])
def test_initial_collection_success_context_order(monkeypatch, initial_collection_tripwire, dtype, policy):
    adapter = _InitialCollectionAdapter(dtype)
    intent = replace(_intent(adapter.delegate, rounds=0), dtype=dtype)
    if policy == "paired":
        intent = replace(intent, evaluation_seed_policy="paired_coordinate_diagnostic_v1",
                         evaluation_seed_family_id="initial-collection-paired")
    if policy == "fixed":
        intent = replace(intent, evaluation_seed_policy="fixed_full_stream_crn_diagnostic_v1", fixed_evaluation_seed=731)
    _stop_after_initial_collection(monkeypatch, reached=True)
    events, before = [], _pre_run_rng_state()
    result = run_sequential(adapter, config=SequentialRunnerConfig(intent), progress_callback=events.append)
    _assert_initial_collection_state(result, adapter, events, intent, committed=4,
        status=RunnerStatus.RESPONSE_TRANSFORM_FAILED, message="collection boundary",
        failures=0, retries=0, total_budget=4)
    assert [item.context.logical_index for item in result.observations] == list(range(4))
    assert all([a.status.value for a in item.attempts] == ["success"] for item in result.observations)
    _assert_pre_run_rng_preserved(before)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_initial_collection_retry_success(monkeypatch, initial_collection_tripwire, dtype):
    adapter = _InitialCollectionAdapter(dtype, fail_at=2, failure="retry", retries=1)
    intent = replace(_intent(adapter.delegate, rounds=0), dtype=dtype)
    _stop_after_initial_collection(monkeypatch, reached=True)
    events, before = [], _pre_run_rng_state()
    result = run_sequential(adapter, config=SequentialRunnerConfig(intent, max_attempts_per_logical_evaluation=2),
                            progress_callback=events.append)
    _assert_initial_collection_state(result, adapter, events, intent, committed=4,
        status=RunnerStatus.RESPONSE_TRANSFORM_FAILED, message="collection boundary",
        failures=1, retries=1, total_budget=8)
    first, second = adapter.calls[2][1], adapter.calls[3][1]
    assert first.randomness_address() == second.randomness_address()
    assert first.evaluation_seed == second.evaluation_seed and (first.attempt_index,second.attempt_index) == (0,1)
    assert [a.status.value for a in result.observations[2].attempts] == ["retryable_failure","success"]
    _assert_pre_run_rng_preserved(before)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scenario,at,failure,limit,budget,committed,status,message,failed,retried", [
    ("first_runtime",0,"runtime",2,None,0,RunnerStatus.SIMULATOR_RUNTIME_FAILED,"RuntimeError: initial adapter failure",1,0),
    ("partial_runtime",2,"runtime",2,None,2,RunnerStatus.SIMULATOR_RUNTIME_FAILED,"RuntimeError: initial adapter failure",1,0),
    ("metadata",2,"metadata",2,None,2,RunnerStatus.SIMULATOR_METADATA_INVALID,"initial metadata failure",1,0),
    ("tensor",2,"tensor",2,None,2,RunnerStatus.SIMULATOR_CONTRACT_FAILED,"initial tensor failure",1,0),
    ("retry_exhausted",2,"retry",2,None,2,RunnerStatus.SIMULATOR_ATTEMPTS_EXHAUSTED,"transient_resource: initial retry",2,1),
    ("total_budget",None,None,2,2,2,RunnerStatus.SIMULATOR_TOTAL_ATTEMPT_BUDGET_EXHAUSTED,"the total checked-evaluation attempt budget is exhausted",0,0),
    ("budget_before_retry",0,"retry",2,1,0,RunnerStatus.SIMULATOR_TOTAL_ATTEMPT_BUDGET_EXHAUSTED,"the total checked-evaluation attempt budget is exhausted",1,0),
    ("attempt_limit_before_budget",0,"retry",1,1,0,RunnerStatus.SIMULATOR_ATTEMPTS_EXHAUSTED,"transient_resource: initial retry",1,0),
])
def test_initial_collection_failure_state(monkeypatch, initial_collection_tripwire, dtype,
        scenario, at, failure, limit, budget, committed, status, message, failed, retried):
    adapter = _InitialCollectionAdapter(dtype, fail_at=at, failure=failure, retries=99)
    intent = replace(_intent(adapter.delegate, rounds=0), dtype=dtype)
    config = SequentialRunnerConfig(intent,max_attempts_per_logical_evaluation=limit,max_total_simulator_attempts=budget)
    _stop_after_initial_collection(monkeypatch, reached=False)
    events, before = [], _pre_run_rng_state()
    result = run_sequential(adapter, config=config, progress_callback=events.append)
    _assert_initial_collection_state(result, adapter, events, intent, committed=committed,
        status=status, message=message, failures=failed, retries=retried, total_budget=config.effective_total_attempt_budget)
    assert len(result.observations) == committed
    assert result.progress_callback_failures == ()
    _assert_pre_run_rng_preserved(before)


def test_initial_collection_capability_failure_precedes_budget(monkeypatch, initial_collection_tripwire):
    adapter = _InitialCollectionAdapter(torch.float64)
    intent = _intent(adapter.delegate, rounds=0)
    original = runner_module.SimulatorCapabilitySnapshot.validate_current
    def validate(snapshot, capabilities):
        if len(adapter.calls) == 2:
            raise runner_module.SimulatorCapabilityDriftError("initial capability drift")
        return original(snapshot, capabilities)
    monkeypatch.setattr(runner_module.SimulatorCapabilitySnapshot, "validate_current", validate)
    _stop_after_initial_collection(monkeypatch, reached=False)
    events, before = [], _pre_run_rng_state()
    result = run_sequential(adapter,config=SequentialRunnerConfig(intent,max_total_simulator_attempts=2),progress_callback=events.append)
    _assert_initial_collection_state(result, adapter, events, intent, committed=2,
        status=RunnerStatus.SIMULATOR_CAPABILITY_DRIFT,message="initial capability drift",failures=0,retries=0,total_budget=2)
    _assert_pre_run_rng_preserved(before)


def test_initial_collection_callback_failure_is_isolated(monkeypatch, initial_collection_tripwire):
    adapter = _InitialCollectionAdapter(torch.float64)
    intent = _intent(adapter.delegate, rounds=0)
    _stop_after_initial_collection(monkeypatch,reached=True)
    events, before = [], _pre_run_rng_state()
    def progress(event):
        events.append(event)
        if event.kind.value == "simulator_attempt_started" and event.logical_index == 1:
            raise RuntimeError("initial display failure")
    result=run_sequential(adapter,config=SequentialRunnerConfig(intent),progress_callback=progress)
    _assert_initial_collection_state(result,adapter,events,intent,committed=4,
        status=RunnerStatus.RESPONSE_TRANSFORM_FAILED,message="collection boundary",failures=0,retries=0,total_budget=4)
    assert result.progress_callback_failures == ("RuntimeError: initial display failure",)
    _assert_pre_run_rng_preserved(before)


def _record_real_gp_inputs(monkeypatch):
    """Observe real preparation/model building without replacing their mathematics."""
    import mutaco.surrogate.exact_multitask as gp_module
    captures, builds = [], []
    fit = runner_module.fit_exact_multitask_gp_with_recovery
    build = gp_module._build_model
    def record_fit(data, *, config, warm_start_states, training_version):
        captures.append((training_version, data, config))
        return fit(data, config=config, warm_start_states=warm_start_states, training_version=training_version)
    def record_build(data, config):
        model = build(data, config)
        n, m = data.theta.shape[0], data.response_dim
        actual = model.train_inputs[0]
        lower = torch.tensor(data.parameter_spec.lower, dtype=data.theta.dtype, device=data.theta.device)
        upper = torch.tensor(data.parameter_spec.upper, dtype=data.theta.dtype, device=data.theta.device)
        coordinates = (data.theta - lower) / (upper - lower) if config.normalize_parameters else data.theta
        assert actual.shape == (n*m, data.parameter_dim + 1)
        assert torch.equal(actual[:,:-1], coordinates.repeat_interleave(m,dim=0))
        assert torch.equal(actual[:,-1],torch.arange(m,dtype=data.theta.dtype,device=data.theta.device).repeat(n))
        assert torch.equal(model.train_targets, data.response_standardized.reshape(-1))
        assert actual.dtype == model.train_targets.dtype == data.theta.dtype
        assert actual.device == model.train_targets.device == data.theta.device
        builds.append((n,m))
        return model
    monkeypatch.setattr(runner_module,"fit_exact_multitask_gp_with_recovery",record_fit)
    monkeypatch.setattr(gp_module,"_build_model",record_build)
    return captures, builds


@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
def test_gp_training_preparation_initial_and_updates(monkeypatch,initial_collection_tripwire,dtype):
    simulator=HandwrittenAnalyticSimulator(dtype=dtype)
    intent=replace(_intent(simulator,rounds=2),dtype=dtype,initial_point_count=16)
    captures,builds=_record_real_gp_inputs(monkeypatch)
    events,before=[],_pre_run_rng_state()
    result=run_sequential(simulator,config=SequentialRunnerConfig(intent),progress_callback=events.append)
    assert result.status is RunnerStatus.COMPLETED
    assert [v for v,_,_ in captures]==[0,1,2]
    assert [data.theta.shape[0] for _,data,_ in captures]==[16,17,18]
    assert len(builds)>=3
    assert result.transform.metadata.fitted_observation_ids==result.initial_design.observation_ids
    names=tuple(axis.name for axis in intent.response_spec.axes)
    for version,data,config in captures:
        rows=result.observations[:16+version]
        assert data.theta.shape==(16+version,intent.parameter_spec.dimension)
        assert data.response_standardized.shape==(16+version,intent.response_spec.dimension)
        assert data.observation_ids==tuple(item.observation_id for item in rows)
        assert data.parameter_spec is intent.parameter_spec
        assert data.transform_id==result.transform.metadata.transform_id
        assert result.fits[version].diagnostic.training_data_id==data.identity()
        assert data.theta.dtype==data.response_standardized.dtype==dtype
        assert data.theta.device==data.response_standardized.device==torch.device("cpu")
        assert data.theta.is_contiguous() and data.response_standardized.is_contiguous()
        assert not data.theta.requires_grad and not data.response_standardized.requires_grad
        assert torch.equal(data.theta,torch.stack([item.theta for item in rows]))
        assert torch.equal(data.response_standardized,torch.stack([item.response_standardized for item in rows]))
        assert torch.equal(data.response_standardized,result.transform.to_standardized(
            torch.stack([item.response_physical for item in rows]),response_names=names))
        assert config.fit_seed_coordinate==runner_module.SeedCoordinate("gp_fit",("training_version",version))
        assert config.fit_seed==SeedDeriver(intent.identity(),intent.root_seed).seed_for(config.fit_seed_coordinate)
        for item in rows:
            assert data.theta.untyped_storage().data_ptr()!=item.theta.untyped_storage().data_ptr()
            assert data.response_standardized.untyped_storage().data_ptr()!=item.response_standardized.untyped_storage().data_ptr()
        kinds=[e.kind.value for e in events if e.sequential_round==version and e.phase=="fit"]
        assert kinds==["gp_fit_started","gp_fit_finished"]
    first,second=captures[0][1],captures[1][1]
    saved_theta=result.observations[0].theta.clone()
    saved_response=result.observations[0].response_standardized.clone()
    first.theta[0,0]+=1
    first.response_standardized[0,0]+=1
    assert torch.equal(result.observations[0].theta,saved_theta)
    assert torch.equal(result.observations[0].response_standardized,saved_response)
    assert torch.equal(second.theta[0],saved_theta) and torch.equal(second.response_standardized[0],saved_response)
    assert result.logical_accounting["committed_observations"]==18
    assert result.logical_accounting["committed_fit_events"]==3
    assert result.physical_accounting["simulator_attempts"]==18
    _assert_pre_run_rng_preserved(before)


@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
def test_gp_training_preparation_keeps_duplicate_parameter_rows(monkeypatch,initial_collection_tripwire,dtype):
    simulator=HandwrittenAnalyticSimulator(dtype=dtype)
    intent=replace(_intent(simulator,rounds=0),dtype=dtype,initial_point_count=16)
    design=runner_module._initial_design(SequentialRunnerConfig(intent),SeedDeriver(intent.identity(),intent.root_seed))
    points=design.theta.tolist();points[-1]=points[0].copy()
    intent=replace(intent,initial_design_policy="caller_supplied",caller_initial_points=tuple(tuple(row) for row in points))
    captures,_=_record_real_gp_inputs(monkeypatch)
    before=_pre_run_rng_state()
    result=run_sequential(simulator,config=SequentialRunnerConfig(intent))
    assert result.status is RunnerStatus.COMPLETED
    assert len(captures)==1
    data=captures[0][1]
    assert data.theta.shape[0]==16 and len(set(data.observation_ids))==16
    assert torch.equal(data.theta[0],data.theta[-1])
    assert torch.equal(data.response_standardized[0],data.response_standardized[-1])
    assert data.observation_ids[0]!=data.observation_ids[-1]
    assert result.logical_accounting["committed_observations"]==16
    _assert_pre_run_rng_preserved(before)


@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
@pytest.mark.parametrize("fail_at",[7,16])
def test_gp_training_preparation_excludes_partial_evaluations(monkeypatch,initial_collection_tripwire,dtype,fail_at):
    adapter=_InitialCollectionAdapter(dtype,fail_at=fail_at,failure="runtime")
    intent=replace(_intent(adapter.delegate,rounds=1),dtype=dtype,initial_point_count=16)
    captures,_=_record_real_gp_inputs(monkeypatch)
    events,before=[],_pre_run_rng_state()
    result=run_sequential(adapter,config=SequentialRunnerConfig(intent),progress_callback=events.append)
    assert result.status is RunnerStatus.SIMULATOR_RUNTIME_FAILED
    assert result.failure_explanation=="RuntimeError: initial adapter failure"
    assert len(result.observations)==fail_at
    assert result.logical_accounting["committed_observations"]==fail_at
    assert result.physical_accounting["simulator_attempts"]==fail_at+1
    assert [data.theta.shape[0] for _,data,_ in captures]==([] if fail_at==7 else [16])
    assert result.logical_accounting["committed_fit_events"]==(0 if fail_at==7 else 1)
    assert all(e.sequential_round!=1 for e in events if e.kind.value=="gp_fit_started")
    assert events[-1].kind.value=="run_finished"
    _assert_pre_run_rng_preserved(before)


@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
def test_gp_training_preparation_reachable_standardization_overflow(monkeypatch,initial_collection_tripwire,dtype):
    """A finite accepted response can overflow the frozen division on an update."""
    class TinyThenLarge:
        def __init__(self):
            self.delegate=HandwrittenAnalyticSimulator(dtype=dtype)
        @property
        def capabilities(self):
            return self.delegate.capabilities
        def evaluate(self,theta,context=None):
            values=(self.delegate.evaluate(theta,context).values*1e-6 if context.logical_index<16
                    else torch.full((3,),torch.finfo(dtype).max/4,dtype=dtype))
            assert bool(torch.isfinite(values).all())
            return SimulationResult(values,{"fixture":"finite-response-frozen-scale-overflow"})
    adapter=TinyThenLarge()
    intent=replace(_intent(adapter.delegate,rounds=1),dtype=dtype,initial_point_count=16)
    captures,_=_record_real_gp_inputs(monkeypatch)
    events,before=[],_pre_run_rng_state()
    result=run_sequential(adapter,config=SequentialRunnerConfig(intent),progress_callback=events.append)
    assert result.status is RunnerStatus.GP_FIT_FAILED
    assert result.failure_explanation=="ValueError: surrogate training tensors must be finite"
    assert len(result.observations)==17 and len(result.fits)==1 and len(captures)==1
    assert torch.isfinite(result.observations[-1].response_physical).all()
    assert not torch.isfinite(result.observations[-1].response_standardized).all()
    assert result.transform is not None and result.standardized_target is not None
    assert result.logical_accounting["committed_observations"]==17
    assert result.logical_accounting["committed_fit_events"]==1
    assert result.physical_accounting["simulator_attempts"]==17
    assert [(e.kind.value,e.sequential_round) for e in events if e.phase=="fit"]==[
        ("gp_fit_started",0),("gp_fit_finished",0),("gp_fit_started",1)]
    assert events[-1].kind.value=="run_finished"
    _assert_pre_run_rng_preserved(before)


class _SequentialCommitAdapter:
    def __init__(self,dtype,*,failure=None,fail_at=5,retry=False):
        self.delegate=HandwrittenAnalyticSimulator(dtype=dtype)
        self.failure,self.fail_at,self.retry=failure,fail_at,retry
        self.calls,self.returned=[],[]
    @property
    def capabilities(self):
        return self.delegate.capabilities
    def evaluate(self,theta,context=None):
        self.calls.append(context)
        if context.logical_index==self.fail_at and self.failure:
            cls={"runtime":RuntimeError,"metadata":runner_module.SimulatorMetadataContractError,
                 "tensor":runner_module.TensorContractError}[self.failure]
            raise cls("sequential evaluation failure")
        if self.retry and context.logical_index==4 and context.attempt_index==0:
            raise RetryableSimulationError(RetryableSimulationReason.TRANSIENT_RESOURCE,"sequential retry")
        result=self.delegate.evaluate(theta,context)
        self.returned.append((context,result.values))
        return result


def _trace_sequential_commit(monkeypatch,events,*,allocation_fault=None):
    raw,standardized,state=[],[],{}
    evaluate=runner_module._evaluate
    standardize=runner_module.FrozenResponseTransform.to_standardized
    collect=runner_module._collect_initial_observations
    def record_evaluate(*args,**kwargs):
        outcome=evaluate(*args,**kwargs)
        if kwargs["phase"] is runner_module.ObservationPhase.SEQUENTIAL:
            raw.append(outcome)
        return outcome
    def record_collect(*args,**kwargs):
        state.update(observations=kwargs["observations"],logical=kwargs["logical"],physical=kwargs["physical"])
        return collect(*args,**kwargs)
    def record_standardize(transform,values,**kwargs):
        in_sequential=bool(events and events[-1].phase=="sequential")
        if in_sequential and allocation_fault=="standardize":
            raise MemoryError("standardization allocation fault")
        output=standardize(transform,values,**kwargs)
        if in_sequential:standardized.append(output)
        return output
    monkeypatch.setattr(runner_module,"_evaluate",record_evaluate)
    monkeypatch.setattr(runner_module,"_collect_initial_observations",record_collect)
    monkeypatch.setattr(runner_module.FrozenResponseTransform,"to_standardized",record_standardize)
    return raw,standardized,state,standardize


@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
@pytest.mark.parametrize("policy",["default","paired","fixed"])
def test_sequential_commit_order_ownership_and_context(monkeypatch,initial_collection_tripwire,dtype,policy):
    adapter=_SequentialCommitAdapter(dtype)
    intent=replace(_intent(adapter.delegate,rounds=3),dtype=dtype)
    if policy=="paired":intent=replace(intent,evaluation_seed_policy="paired_coordinate_diagnostic_v1",evaluation_seed_family_id="sequential-commit-paired")
    if policy=="fixed":intent=replace(intent,evaluation_seed_policy="fixed_full_stream_crn_diagnostic_v1",fixed_evaluation_seed=193)
    events=[];raw,outputs,state,standardize=_trace_sequential_commit(monkeypatch,events)
    before=_pre_run_rng_state()
    result=run_sequential(adapter,config=SequentialRunnerConfig(intent),progress_callback=events.append)
    assert result.status is RunnerStatus.COMPLETED
    assert len(raw)==len(outputs)==3
    assert result.transform.metadata.fitted_observation_ids==result.initial_design.observation_ids
    names=tuple(axis.name for axis in intent.response_spec.axes)
    deriver=SeedDeriver(ScientificIdentity(intent.evaluation_seed_family_id) if intent.evaluation_seed_family_id else intent.identity(),intent.root_seed)
    expected_events=[]
    for round_index,(outcome,output,committed) in enumerate(zip(raw,outputs,result.observations[4:])):
        original=outcome.observation
        assert original.response_standardized is None and committed is not original
        assert committed.theta is original.theta and committed.response_physical is original.response_physical
        assert committed.context is original.context and committed.attempts is original.attempts
        assert torch.equal(committed.response_standardized,output)
        assert committed.response_standardized.untyped_storage().data_ptr()!=output.untyped_storage().data_ptr()
        assert torch.equal(committed.response_standardized,standardize(result.transform,original.response_physical,response_names=names))
        assert committed.response_standardized.dtype==committed.response_physical.dtype==dtype
        assert committed.response_standardized.device==torch.device("cpu")
        assert committed.response_standardized.is_contiguous() and not committed.response_standardized.requires_grad
        proposal=result.proposals[round_index]
        assert torch.equal(committed.theta,proposal.candidate)
        assert committed.observation_id==runner_module.sequential_observation_identity(
            scientific_identity=intent.identity(),global_logical_index=4+round_index,
            sequential_round_index=round_index,candidate_id=proposal.candidate_id,theta=proposal.candidate)
        context=committed.context
        coordinate=runner_module.SeedCoordinate("simulator_evaluation",("sequential",4+round_index,0))
        assert context.evaluation_seed==(intent.fixed_evaluation_seed if policy=="fixed" else deriver.seed_for(coordinate))
        assert context.logical_index==4+round_index and context.purpose=="sequential" and context.repeat_index==0
        expected_events.extend([("proposal_started",round_index),("proposal_finished",round_index),
            ("simulator_attempt_started",round_index),("simulator_attempt_finished",round_index),
            ("gp_fit_started",round_index+1),("gp_fit_finished",round_index+1)])
    start=next(i for i,e in enumerate(events) if e.kind.value=="proposal_started")
    end=next(i for i,e in enumerate(events) if e.kind.value=="recommendation_started")
    assert [(e.kind.value,e.sequential_round) for e in events[start:end]]==expected_events
    assert result.logical_accounting["committed_observations"]==7 and result.logical_accounting["sequential_observations"]==3
    assert result.logical_accounting["committed_fit_events"]==4 and result.logical_accounting["committed_proposal_events"]==3
    assert result.physical_accounting["simulator_attempts"]==result.physical_accounting["simulator_successes"]==7
    assert result.physical_accounting["simulator_failures"]==0 and result.progress_callback_failures==()
    saved=result.observations[4].response_standardized.clone();outputs[0].add_(1)
    assert torch.equal(result.observations[4].response_standardized,saved)
    assert state["observations"][4].observation_id==result.observations[4].observation_id
    _assert_pre_run_rng_preserved(before)


@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
def test_sequential_commit_retry_reuses_realization(monkeypatch,initial_collection_tripwire,dtype):
    adapter=_SequentialCommitAdapter(dtype,retry=True)
    intent=replace(_intent(adapter.delegate,rounds=2),dtype=dtype)
    events=[];raw,outputs,_,_=_trace_sequential_commit(monkeypatch,events)
    before=_pre_run_rng_state()
    result=run_sequential(adapter,config=SequentialRunnerConfig(intent,max_attempts_per_logical_evaluation=2),progress_callback=events.append)
    assert result.status is RunnerStatus.COMPLETED and len(outputs)==2
    attempts=[c for c in adapter.calls if c.logical_index==4]
    assert len(attempts)==2 and attempts[0].randomness_address()==attempts[1].randomness_address()
    assert attempts[0].evaluation_seed==attempts[1].evaluation_seed
    assert [a.status.value for a in result.observations[4].attempts]==["retryable_failure","success"]
    assert result.observations[4].context.attempt_index==1
    assert result.physical_accounting["simulator_attempts"]==7 and result.physical_accounting["simulator_retry_attempts"]==1
    assert result.logical_accounting["committed_observations"]==6 and result.logical_accounting["committed_fit_events"]==3
    _assert_pre_run_rng_preserved(before)


@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
@pytest.mark.parametrize("failure,status,message",[
    ("runtime",RunnerStatus.SIMULATOR_RUNTIME_FAILED,"RuntimeError: sequential evaluation failure"),
    ("metadata",RunnerStatus.SIMULATOR_METADATA_INVALID,"sequential evaluation failure"),
    ("tensor",RunnerStatus.SIMULATOR_CONTRACT_FAILED,"sequential evaluation failure"),
])
def test_sequential_commit_evaluation_failure_precedes_transform(monkeypatch,initial_collection_tripwire,dtype,failure,status,message):
    adapter=_SequentialCommitAdapter(dtype,failure=failure)
    intent=replace(_intent(adapter.delegate,rounds=3),dtype=dtype)
    events=[];raw,outputs,_,standardize=_trace_sequential_commit(monkeypatch,events)
    trace=runner_module.FrozenResponseTransform.to_standardized
    def pending_fault(transform,values,**kwargs):
        if events and events[-1].phase=="sequential" and events[-1].logical_index==5:
            raise MemoryError("later standardization must not execute")
        return trace(transform,values,**kwargs)
    monkeypatch.setattr(runner_module.FrozenResponseTransform,"to_standardized",pending_fault)
    before=_pre_run_rng_state()
    result=run_sequential(adapter,config=SequentialRunnerConfig(intent),progress_callback=events.append)
    assert result.status is status and result.failure_explanation==message
    assert len(result.observations)==5 and len(outputs)==1 and raw[-1].observation is None
    assert len(result.fits)==2 and len(result.proposals)==2 and result.recommendation is None
    assert result.logical_accounting["committed_observations"]==5 and result.logical_accounting["sequential_observations"]==1
    assert result.physical_accounting["simulator_attempts"]==6 and result.physical_accounting["simulator_successes"]==5
    assert result.physical_accounting["simulator_failures"]==1
    assert events[-2].kind.value=="simulator_attempt_started" and events[-1].kind.value=="run_finished"
    assert not any(e.kind.value=="gp_fit_started" and e.sequential_round==2 for e in events)
    _assert_pre_run_rng_preserved(before)


@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
def test_sequential_commit_standardization_allocation_failure(monkeypatch,initial_collection_tripwire,dtype):
    adapter=_SequentialCommitAdapter(dtype)
    intent=replace(_intent(adapter.delegate),dtype=dtype)
    events=[];raw,outputs,state,_=_trace_sequential_commit(monkeypatch,events,allocation_fault="standardize")
    before=_pre_run_rng_state()
    result=run_sequential(adapter,config=SequentialRunnerConfig(intent),progress_callback=events.append)
    assert result.status is RunnerStatus.RESPONSE_TRANSFORM_FAILED
    assert result.failure_explanation=="MemoryError: standardization allocation fault"
    assert raw[0].observation.response_standardized is None and outputs==[]
    assert len(result.observations)==4 and len(result.fits)==1 and len(result.proposals)==1
    assert result.physical_accounting["simulator_attempts"]==result.physical_accounting["simulator_successes"]==5
    assert result.logical_accounting["committed_observations"]==4 and result.logical_accounting["sequential_observations"]==0
    assert events[-2].kind.value=="simulator_attempt_finished" and events[-1].kind.value=="run_finished"
    assert not any(e.kind.value=="gp_fit_started" and e.sequential_round==1 for e in events)
    _assert_pre_run_rng_preserved(before)


@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
def test_sequential_commit_clone_allocation_failure_stays_uncaught(monkeypatch,initial_collection_tripwire,dtype):
    """Inject a resource failure at the actual commit clone, outside its catch."""
    adapter=_SequentialCommitAdapter(dtype)
    intent=replace(_intent(adapter.delegate),dtype=dtype)
    events=[];raw,outputs,state,_=_trace_sequential_commit(monkeypatch,events)
    original_clone=torch.Tensor.clone
    def clone(tensor,*args,**kwargs):
        if outputs and tensor.untyped_storage().data_ptr()==outputs[-1].untyped_storage().data_ptr():
            raise MemoryError("commit clone allocation fault")
        return original_clone(tensor,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,"clone",clone)
    before=_pre_run_rng_state()
    with pytest.raises(MemoryError,match="commit clone allocation fault"):
        run_sequential(adapter,config=SequentialRunnerConfig(intent),progress_callback=events.append)
    assert len(raw)==len(outputs)==1 and raw[0].observation.response_standardized is None
    assert len(state["observations"])==4 and state["logical"]["committed_observations"]==4
    assert state["logical"]["sequential_observations"]==0 and state["logical"]["committed_fit_events"]==1
    assert state["physical"]["simulator_attempts"]==state["physical"]["simulator_successes"]==5
    assert events[-1].kind.value=="simulator_attempt_finished"
    assert not any(e.kind.value=="run_finished" for e in events)
    _assert_pre_run_rng_preserved(before)


@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
def test_sequential_commit_after_production_duplicate_recovery(monkeypatch,initial_collection_tripwire,dtype):
    """A controlled duplicate proposal exercises real recovery on a real raw pool."""
    import mutaco.acquisition.proposal as proposal_module
    adapter=_SequentialCommitAdapter(dtype)
    intent=replace(_intent(adapter.delegate),dtype=dtype,duplicate_policy="best_ranked_unevaluated_raw_v1")
    original=runner_module.propose_candidate
    def duplicate(*args,**kwargs):
        result=original(*args,**kwargs)
        candidate=kwargs["evaluated_theta"][0].clone()
        identity=proposal_module._candidate_identity(candidate,source=result.candidate_source,branch=result.branch,
            pool=result.pool,target_id=result.target_id,model_id=result.model_id,transform_id=result.transform_id,
            parameter_spec=kwargs["parameter_spec"],config=kwargs["config"])
        return replace(result,candidate=candidate,candidate_id=identity,
            duplicate=proposal_module._duplicate_diagnostic(candidate,kwargs["evaluated_theta"]))
    monkeypatch.setattr(runner_module,"propose_candidate",duplicate)
    before=_pre_run_rng_state()
    result=run_sequential(adapter,config=SequentialRunnerConfig(intent))
    assert result.status is RunnerStatus.COMPLETED
    proposal=result.proposals[0]
    assert proposal.candidate_source.value=="raw_duplicate_recovery" and proposal.duplicate_recovery is not None
    assert torch.equal(result.observations[4].theta,proposal.candidate)
    assert result.logical_accounting["committed_observations"]==5 and result.logical_accounting["committed_fit_events"]==2
    assert result.physical_accounting["simulator_attempts"]==5
    recovery=proposal.duplicate_recovery
    assert recovery.additional_simulator_calls==recovery.additional_gp_fits==recovery.additional_random_draws==0
    _assert_pre_run_rng_preserved(before)


def test_sequential_commit_callback_failure_remains_observational(monkeypatch,initial_collection_tripwire):
    adapter=_SequentialCommitAdapter(torch.float64)
    intent=_intent(adapter.delegate,rounds=2)
    events=[];raw,outputs,_,_=_trace_sequential_commit(monkeypatch,events)
    def progress(event):
        events.append(event)
        if event.kind.value=="simulator_attempt_finished" and event.phase=="sequential":
            raise RuntimeError("sequential display failure")
    before=_pre_run_rng_state()
    result=run_sequential(adapter,config=SequentialRunnerConfig(intent),progress_callback=progress)
    assert result.status is RunnerStatus.COMPLETED and len(outputs)==2
    assert result.progress_callback_failures==("RuntimeError: sequential display failure",)*2
    assert len(result.observations)==6 and len(result.fits)==3
    assert result.physical_accounting["simulator_successes"]==6
    _assert_pre_run_rng_preserved(before)


def _fit_result_state_trace(monkeypatch):
    state={}
    original=runner_module._collect_initial_observations
    def collect(*args,**kwargs):
        state.update(observations=kwargs["observations"],logical=kwargs["logical"],physical=kwargs["physical"])
        return original(*args,**kwargs)
    monkeypatch.setattr(runner_module,"_collect_initial_observations",collect)
    return state


def _fit_result_intent(simulator,*,rounds=3,recovery=True):
    intent=replace(_intent(simulator,rounds=rounds),dtype=simulator.dtype)
    if recovery:
        from mutaco import GRAPH_M20_TASK_VARIANCE_FLOOR,GRAPH_M20_TASK_VARIANCE_REFERENCE_SCALE,GRAPH_M20_TASK_VARIANCE_EPSILON_MULTIPLIER
        intent=replace(intent,gp_settings={**strict_metadata_document(intent.gp_settings),
            "recovery_policy":"graph_m20_sequential_recovery_v1","max_warm_start_states":2,
            "task_variance_floor_policy":"standardized_absolute_v1","task_variance_floor":GRAPH_M20_TASK_VARIANCE_FLOOR,
            "task_variance_floor_reference_scale":GRAPH_M20_TASK_VARIANCE_REFERENCE_SCALE,
            "task_variance_floor_epsilon_multiplier":GRAPH_M20_TASK_VARIANCE_EPSILON_MULTIPLIER})
    return intent


def _controlled_fit_diagnostic(data,config,version,*,success=False):
    attempts=tuple({"fit_executed":True,"status":"failed","fit_id":f"controlled-{version}-{i}"} for i in range(5))
    if version>0:
        attempts=({"fit_executed":False,"status":"rejected_incompatible"},*attempts,
                  {"fit_executed":False,"status":"fixed_conditioning_failed"})
    return GPFitDiagnostic(status=GPFitStatus.SUCCESS if success else GPFitStatus.FAILED,
        fit_id=f"controlled-fit-{version}",training_data_id=data.identity(),model_id=None,
        elapsed_seconds=0.125*(version+1),warnings=("controlled-result-double",),
        failure_type=None if success else "ControlledRecoveryFailure",
        failure_message=None if success else "controlled recovery exhausted",
        validation_posterior_calls=3+version,posterior_validation_diagnostics=None,optimizer_result=None,
        failure_kind=None if success else (GPFitFailureKind.INITIAL_GP_FIT_FAILURE if version==0 else GPFitFailureKind.RECOVERY_EXHAUSTED),
        failure_stage=None if success else "controlled_recovery",exception_chain=(),
        configuration=config.provenance(),execution={},recovery_attempts=attempts)


@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
@pytest.mark.parametrize("broken_callback",[None,"gp_fit_started","gp_fit_finished"])
def test_fit_result_real_success_warm_history_and_events(monkeypatch,initial_collection_tripwire,dtype,broken_callback):
    simulator=HandwrittenAnalyticSimulator(dtype=dtype)
    intent=_fit_result_intent(simulator)
    state=_fit_result_state_trace(monkeypatch)
    original=runner_module.fit_exact_multitask_gp_with_recovery
    returned,inputs,events,snapshots=[],[],[],[]
    def fit(data,*,config,warm_start_states,training_version):
        inputs.append((data,config,training_version,tuple(s.source_fit_id for s in warm_start_states)))
        result=original(data,config=config,warm_start_states=warm_start_states,training_version=training_version)
        returned.append(result)
        return result
    monkeypatch.setattr(runner_module,"fit_exact_multitask_gp_with_recovery",fit)
    def progress(event):
        events.append(event)
        if event.kind.value=="gp_fit_finished":
            version=event.sequential_round;diag=returned[-1].diagnostic
            assert event.message==f"GP fit for training version {version}: {diag.status.value}"
            assert dict(event.details)=={"fit_id":diag.fit_id,"model_id":diag.model_id,"elapsed_seconds":diag.elapsed_seconds}
            snapshots.append((dict(state["physical"]),dict(state["logical"])))
        if event.kind.value==broken_callback:raise RuntimeError("fit progress unavailable")
    before=_pre_run_rng_state()
    result=run_sequential(simulator,config=SequentialRunnerConfig(intent),progress_callback=progress)
    assert result.status is RunnerStatus.COMPLETED and len(returned)==4
    assert [item[2] for item in inputs]==[0,1,2,3]
    fit_ids=[r.diagnostic.fit_id for r in returned]
    assert [item[3] for item in inputs]==[(),(fit_ids[0],),(fit_ids[1],fit_ids[0]),(fit_ids[2],fit_ids[1])]
    seconds=0.0;attempts=posterior_calls=0
    for version,(record,fit_result,(data,config,_,_),snapshot) in enumerate(zip(result.fits,returned,inputs,snapshots)):
        diag=fit_result.diagnostic;seconds+=diag.elapsed_seconds
        attempts+=sum(bool(a.get("fit_executed",False)) for a in diag.recovery_attempts)
        posterior_calls+=diag.validation_posterior_calls
        physical,logical=snapshot
        assert record.training_version==version and record.diagnostic is diag
        assert diag.training_data_id==data.identity() and diag.model_id==fit_result.surrogate.model_id
        assert config.fit_seed==SeedDeriver(intent.identity(),intent.root_seed).seed_for(
            runner_module.SeedCoordinate("gp_fit",("training_version",version)))
        assert physical["gp_fit_attempts"]==attempts and physical["gp_fit_seconds"]==seconds
        assert physical["gp_fit_validation_posterior_calls"]==posterior_calls
        assert physical["gp_fit_successes"]==logical["committed_fit_events"]==logical["training_data_versions"]==version
    assert result.terminal_surrogate is returned[-1].surrogate
    assert result.logical_accounting["committed_fit_events"]==result.physical_accounting["gp_fit_successes"]==4
    assert result.logical_accounting["committed_observations"]==result.physical_accounting["simulator_attempts"]==7
    assert result.logical_accounting["committed_recommendation_events"]==1
    assert [(e.kind.value,e.sequential_round) for e in events if e.phase=="fit"]==[
        pair for version in range(4) for pair in (("gp_fit_started",version),("gp_fit_finished",version))]
    assert result.progress_callback_failures==(("RuntimeError: fit progress unavailable",)*4 if broken_callback else ())
    _assert_pre_run_rng_preserved(before)


@pytest.mark.parametrize("version",[0,1])
@pytest.mark.parametrize("mode",["raised","typed_failure","success_without_model","failed_with_model"])
def test_fit_result_controlled_failures_keep_partial_state(monkeypatch,initial_collection_tripwire,version,mode):
    simulator=HandwrittenAnalyticSimulator()
    intent=_fit_result_intent(simulator,rounds=2)
    state=_fit_result_state_trace(monkeypatch)
    original=runner_module.fit_exact_multitask_gp_with_recovery
    returned,events,finish_observed=[],[],[]
    class UnusableModel:
        def warm_start_state(self):raise AssertionError("a failed fit must not update warm history")
    def fit(data,*,config,warm_start_states,training_version):
        if training_version!=version:
            result=original(data,config=config,warm_start_states=warm_start_states,training_version=training_version)
        elif mode=="raised":raise RuntimeError("controlled numerical fit failure")
        else:
            diag=_controlled_fit_diagnostic(data,config,training_version,success=mode=="success_without_model")
            result=GPFitResult(UnusableModel() if mode=="failed_with_model" else None,diag)
        returned.append(result)
        return result
    monkeypatch.setattr(runner_module,"fit_exact_multitask_gp_with_recovery",fit)
    def progress(event):
        events.append(event)
        if event.kind.value=="gp_fit_finished":
            finish_observed.append((event.sequential_round,dict(state["physical"]),dict(state["logical"])))
            if event.sequential_round==version:raise RuntimeError("failed-fit display unavailable")
    before=_pre_run_rng_state()
    result=run_sequential(simulator,config=SequentialRunnerConfig(intent),progress_callback=progress)
    typed=mode in {"typed_failure","failed_with_model"}
    assert result.status is (RunnerStatus.INITIAL_GP_FIT_FAILED if typed and version==0 else RunnerStatus.GP_FIT_FAILED)
    assert result.failure_explanation==("RuntimeError: controlled numerical fit failure" if mode=="raised" else
        None if mode=="success_without_model" else "controlled recovery exhausted")
    assert len(result.observations)==4+version and result.logical_accounting["committed_observations"]==4+version
    assert result.logical_accounting["committed_fit_events"]==result.logical_accounting["training_data_versions"]==version
    assert result.physical_accounting["gp_fit_successes"]==version
    assert len(result.fits)==len(returned)==version+(mode!="raised")
    assert all(record.diagnostic is reported.diagnostic for record,reported in zip(result.fits,returned))
    assert result.physical_accounting["gp_fit_attempts"]==sum(bool(a.get("fit_executed",False)) for item in returned for a in item.diagnostic.recovery_attempts)
    assert result.physical_accounting["gp_fit_validation_posterior_calls"]==sum(item.diagnostic.validation_posterior_calls for item in returned)
    assert result.physical_accounting["simulator_attempts"]==4+version
    assert result.logical_accounting["committed_proposal_events"]==version
    assert result.recommendation is None and result.logical_accounting["committed_recommendation_events"]==0
    assert result.terminal_surrogate is None
    assert [(e.kind.value,e.sequential_round) for e in events if e.phase=="fit"]==[
        ("gp_fit_started",0),*([("gp_fit_finished",0),("gp_fit_started",1)] if version else []),
        *([] if mode=="raised" else [("gp_fit_finished",version)])]
    assert events[-1].kind.value=="run_finished"
    assert result.progress_callback_failures==(() if mode=="raised" else ("RuntimeError: failed-fit display unavailable",))
    if mode!="raised":
        _,physical,logical=finish_observed[-1]
        assert logical["committed_fit_events"]==physical["gp_fit_successes"]==version
        assert result.fits[-1].diagnostic.status is (GPFitStatus.SUCCESS if mode=="success_without_model" else GPFitStatus.FAILED)
    _assert_pre_run_rng_preserved(before)


@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
def test_fit_result_controlled_recovery_counts_only_executed_attempts(monkeypatch,initial_collection_tripwire,dtype):
    simulator=HandwrittenAnalyticSimulator(dtype=dtype)
    intent=replace(_intent(simulator,rounds=0),dtype=dtype)
    original=runner_module.fit_exact_multitask_gp_with_recovery
    selected=[]
    def fit(*args,**kwargs):
        result=original(*args,**kwargs)
        attempts=({"fit_executed":False,"status":"rejected_incompatible"},
                  {"fit_executed":True,"status":"failed","fit_id":"controlled-earlier-failure"},
                  {"fit_executed":True,"status":"success","fit_id":result.diagnostic.fit_id,"selected":True},
                  {"fit_executed":False,"status":"conditioned_without_optimization"})
        diag=replace(result.diagnostic,recovery_attempts=attempts,elapsed_seconds=0.375,validation_posterior_calls=9)
        selected.append(GPFitResult(result.surrogate,diag));return selected[-1]
    monkeypatch.setattr(runner_module,"fit_exact_multitask_gp_with_recovery",fit)
    before=_pre_run_rng_state()
    result=run_sequential(simulator,config=SequentialRunnerConfig(intent))
    assert result.status is RunnerStatus.COMPLETED and result.terminal_surrogate is selected[0].surrogate
    assert result.fits[0].diagnostic is selected[0].diagnostic
    assert result.physical_accounting["gp_fit_attempts"]==2 and result.physical_accounting["gp_fit_successes"]==1
    assert result.physical_accounting["gp_fit_seconds"]==0.375
    assert result.physical_accounting["gp_fit_validation_posterior_calls"]==9
    assert result.logical_accounting["committed_fit_events"]==1 and result.logical_accounting["committed_recommendation_events"]==1
    _assert_pre_run_rng_preserved(before)


@pytest.mark.parametrize("version",[0,1])
def test_fit_result_record_allocation_precedes_failed_status_handling(monkeypatch,initial_collection_tripwire,version):
    simulator=HandwrittenAnalyticSimulator()
    intent=_fit_result_intent(simulator,rounds=2)
    state=_fit_result_state_trace(monkeypatch)
    original=runner_module.fit_exact_multitask_gp_with_recovery
    record=runner_module.FitRecord
    returned,events=[],[]
    def fit(data,*,config,warm_start_states,training_version):
        if training_version==version:result=GPFitResult(None,_controlled_fit_diagnostic(data,config,training_version))
        else:result=original(data,config=config,warm_start_states=warm_start_states,training_version=training_version)
        returned.append(result);return result
    def allocate(training_version,diagnostic):
        if training_version==version:raise MemoryError("fit record allocation fault")
        return record(training_version,diagnostic)
    monkeypatch.setattr(runner_module,"fit_exact_multitask_gp_with_recovery",fit)
    monkeypatch.setattr(runner_module,"FitRecord",allocate)
    before=_pre_run_rng_state()
    with pytest.raises(MemoryError,match="fit record allocation fault"):
        run_sequential(simulator,config=SequentialRunnerConfig(intent),progress_callback=events.append)
    assert state["logical"]["committed_fit_events"]==state["physical"]["gp_fit_successes"]==version
    assert state["logical"]["committed_observations"]==4+version
    assert state["physical"]["gp_fit_attempts"]==sum(bool(a.get("fit_executed",False)) for item in returned[:version] for a in item.diagnostic.recovery_attempts)
    assert events[-1].kind.value=="gp_fit_started" and events[-1].sequential_round==version
    assert not any(e.kind.value=="run_finished" for e in events)
    _assert_pre_run_rng_preserved(before)


def test_complete_runner_enforces_lifecycle_and_accounting() -> None:
    simulator = HandwrittenAnalyticSimulator()
    events = []
    config = SequentialRunnerConfig(_intent(simulator))
    assert "terminal_recommendation_context" not in config.execution_document()
    result = run_sequential(
        simulator,
        config=config,
        progress_callback=events.append,
    )

    assert result.status is RunnerStatus.COMPLETED
    assert len(result.observations) == 5
    assert len(result.fits) == 2
    assert len(result.proposals) == 1
    assert result.recommendation is not None
    assert result.recommendation.simulator_calls == 0
    assert result.logical_accounting == {
        "planned_observations": 5,
        "committed_observations": 5,
        "initial_observations": 4,
        "sequential_observations": 1,
        "training_data_versions": 2,
        "committed_fit_events": 2,
        "committed_proposal_events": 1,
        "committed_recommendation_events": 1,
    }
    assert result.physical_accounting["simulator_attempts"] == 5
    assert result.physical_accounting["gp_fit_attempts"] == 2
    assert result.physical_accounting["proposal_calls"] == 1
    assert result.physical_accounting["recommendation_posterior_calls"] == 1
    assert result.initial_design is not None
    assert tuple(item.observation_id for item in result.observations[:4]) == (
        result.initial_design.observation_ids
    )
    assert events[0].kind.value == "run_started"
    assert events[-1].kind.value == "run_finished"


def test_zero_round_runner_fits_once_and_recommends_without_simulation() -> None:
    simulator = HandwrittenAnalyticSimulator()
    result = run_sequential(
        simulator,
        config=SequentialRunnerConfig(_intent(simulator, rounds=0)),
    )
    assert result.status is RunnerStatus.COMPLETED
    assert len(result.observations) == 4
    assert len(result.fits) == 1
    assert not result.proposals
    assert result.physical_accounting["simulator_attempts"] == 4
    assert result.logical_accounting["committed_recommendation_events"] == 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("stage", ["fit", "to_standardized", "standardize_target"])
def test_response_preparation_failure_preserves_committed_state(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype, stage: str,
) -> None:
    """A later preparation failure retains earlier transform/observation work."""
    simulator = HandwrittenAnalyticSimulator(dtype=dtype)
    intent = replace(_intent(simulator, rounds=0), dtype=dtype)
    events = []

    def fail(*args, **kwargs):
        raise ValueError(f"forced {stage} failure")

    monkeypatch.setattr(runner_module.FrozenResponseTransform, stage, fail)
    before = (random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone())
    result = run_sequential(
        simulator, config=SequentialRunnerConfig(intent), progress_callback=events.append,
    )

    assert result.status is RunnerStatus.RESPONSE_TRANSFORM_FAILED
    assert result.failure_explanation == f"ValueError: forced {stage} failure"
    assert (result.transform is None) == (stage == "fit")
    assert result.standardized_target is None
    assert tuple(item.observation_id for item in result.observations) == result.initial_design.observation_ids
    assert len(result.observations) == result.logical_accounting["initial_observations"] == 4
    for item in result.observations:
        assert item.response_physical.dtype is dtype
        assert (item.response_standardized is not None) == (stage == "standardize_target")
    assert result.physical_accounting["simulator_attempts"] == 4
    assert result.physical_accounting["gp_fit_attempts"] == 0
    assert not result.fits and not result.proposals and result.recommendation is None
    assert "transform_fitted" not in [event.kind.value for event in events]
    assert events[-1].kind.value == "run_finished"
    assert random.getstate() == before[0]
    assert all(np.array_equal(a, b) for a, b in zip(before[1], np.random.get_state()))
    assert torch.equal(before[2], torch.random.get_rng_state())


def test_runner_dispatches_evaluated_physical_l2_in_core() -> None:
    simulator = HandwrittenAnalyticSimulator()
    intent = replace(
        _intent(simulator, rounds=0),
        recommendation_policy="evaluated_posterior_mean_physical_l2",
    )
    result = run_sequential(simulator, config=SequentialRunnerConfig(intent))

    assert result.status is RunnerStatus.COMPLETED
    assert result.recommendation is not None
    assert result.recommendation.policy_id == "evaluated_posterior_mean_physical_l2"
    assert result.recommendation.selected_observation_id in {
        item.observation_id for item in result.observations
    }
    assert result.recommendation.simulator_calls == 0
    assert result.recommendation.gp_fits == 0
    assert result.physical_accounting["recommendation_posterior_calls"] == 1


def test_runner_dispatches_continuous_physical_l2_in_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key, value in {
        "raw_samples": 16,
        "restarts": 2,
        "max_iterations": 2,
    }.items():
        monkeypatch.setitem(
            physical_l2_module.CONTINUOUS_PHYSICAL_L2_SETTINGS,
            key,
            value,
        )
    simulator = HandwrittenAnalyticSimulator()
    intent = replace(
        _intent(simulator, rounds=0),
        recommendation_policy="continuous_posterior_mean_physical_l2",
    )
    result = run_sequential(simulator, config=SequentialRunnerConfig(intent))

    assert result.status is RunnerStatus.COMPLETED
    assert result.recommendation is not None
    assert result.recommendation.policy_id == "continuous_posterior_mean_physical_l2"
    assert result.recommendation.selected_observation_id is None
    assert result.recommendation.simulator_calls == 0
    assert result.recommendation.gp_fits == 0
    assert result.physical_accounting["recommendation_posterior_calls"] >= 1


def test_runner_dispatches_continuous_physical_mae_in_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key, value in {
        "raw_samples": 16,
        "restarts": 2,
        "max_iterations": 2,
    }.items():
        monkeypatch.setitem(
            physical_mae_module.CONTINUOUS_PHYSICAL_MAE_SETTINGS,
            key,
            value,
        )
    simulator = HandwrittenAnalyticSimulator()
    intent = replace(
        _intent(simulator, rounds=0),
        recommendation_policy="continuous_posterior_mean_physical_l1_mae",
    )
    result = run_sequential(simulator, config=SequentialRunnerConfig(intent))

    assert result.status is RunnerStatus.COMPLETED
    assert result.recommendation is not None
    assert result.recommendation.policy_id == (
        "continuous_posterior_mean_physical_l1_mae"
    )
    assert result.recommendation.selected_observation_id is None
    assert result.recommendation.simulator_calls == 0
    assert result.recommendation.gp_fits == 0
    assert result.physical_accounting["recommendation_posterior_calls"] >= 1


@pytest.mark.parametrize(
    ("policy_id", "function_name", "settings"),
    (
        (
            "continuous_posterior_mean_physical_l2",
            "recommend_continuous_physical_l2",
            physical_l2_module.CONTINUOUS_PHYSICAL_L2_SETTINGS,
        ),
        (
            "continuous_posterior_mean_physical_l1_mae",
            "recommend_continuous_physical_mae",
            physical_mae_module.CONTINUOUS_PHYSICAL_MAE_SETTINGS,
        ),
    ),
)
def test_explicit_terminal_context_preserves_intent_identity_and_real_target_seed(
    monkeypatch: pytest.MonkeyPatch,
    policy_id: str,
    function_name: str,
    settings: dict[str, object],
) -> None:
    for key, value in {
        "raw_samples": 16,
        "restarts": 2,
        "max_iterations": 2,
    }.items():
        monkeypatch.setitem(settings, key, value)
    calls = 0
    original = getattr(runner_module, function_name)

    def counted(**kwargs):
        nonlocal calls
        calls += 1
        return original(**kwargs)

    monkeypatch.setattr(runner_module, function_name, counted)
    simulator = HandwrittenAnalyticSimulator()
    intent = _intent(simulator, rounds=0)
    family = ScientificIdentity.create(
        {"schema": "test-graph-terminal-family-v1", "target": 7}
    )
    deriver = SeedDeriver(family, 60101)
    context = TerminalRecommendationContext(
        policy_id=policy_id,
        target_id="graph-target-7",
        target_index=7,
        seed_deriver=deriver,
    )
    result = run_sequential(
        simulator,
        config=SequentialRunnerConfig(
            intent,
            terminal_recommendation_context=context,
        ),
    )

    assert result.status is RunnerStatus.COMPLETED
    assert result.scientific_identity == intent.identity()
    assert result.recommendation is not None
    assert result.recommendation.policy_id == policy_id
    assert result.recommendation.target_id == "graph-target-7"
    document = result.recommendation.document()
    assert document["seed_coordinates"] == [7, policy_id]
    assert document["derived_seed"] == deriver.seed_for(
        runner_module.SeedCoordinate(
            "terminal_recommendation", (7, policy_id)
        )
    )
    assert calls == 1
    assert result.logical_accounting["committed_recommendation_events"] == 1
    assert result.recommendation.simulator_calls == 0
    assert result.recommendation.gp_fits == 0
    projection = scientific_projection(result)
    assert projection["recommendation"]["selected_index"] is None
    assert projection["recommendation"]["posterior_model_id"] == (
        result.recommendation.model_id
    )


def test_runner_rejects_beta_response_dimension_drift_before_simulation() -> None:
    simulator = HandwrittenAnalyticSimulator()
    beta = JointChiSquareResponseDimensionBeta.resolve(
        coverage=0.95, response_dimension=20
    )
    intent = replace(
        _intent(simulator, rounds=0),
        acquisition_settings={
            "raw_policy": "scrambled_sobol",
            "raw_samples": 8,
            "restarts": 1,
            "max_iterations": 1,
            "beta": beta.value,
            "beta_policy": beta.document(),
        },
    )
    result = run_sequential(simulator, config=SequentialRunnerConfig(intent))

    assert result.status is RunnerStatus.CONFIGURATION_INVALID
    assert "response dimension" in str(result.failure_explanation)
    assert result.physical_accounting["simulator_attempts"] == 0


def test_initial_gp_failure_is_isolated_without_proposal_or_recommendation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostic = GPFitDiagnostic(
        status=GPFitStatus.FAILED,
        fit_id="failed-initial-fit",
        training_data_id="initial-training",
        model_id=None,
        elapsed_seconds=0.0,
        warnings=("campaign_quality_warning_required_if_initial_failures_recur",),
        failure_type="InitialGPFitFailure",
        failure_message="initial_gp_fit_failure",
        validation_posterior_calls=0,
        posterior_validation_diagnostics=None,
        optimizer_result=None,
        failure_kind=GPFitFailureKind.INITIAL_GP_FIT_FAILURE,
        failure_stage="initial_gp_fit_recovery",
        exception_chain=(),
        configuration={},
        execution={},
        recovery_attempts=tuple(
            {"fit_executed": True, "status": "failed"} for _ in range(5)
        ),
    )
    monkeypatch.setattr(
        runner_module,
        "fit_exact_multitask_gp_with_recovery",
        lambda *args, **kwargs: GPFitResult(None, diagnostic),
    )
    simulator = HandwrittenAnalyticSimulator()
    result = run_sequential(
        simulator,
        config=SequentialRunnerConfig(_intent(simulator, rounds=1)),
    )
    assert result.status is RunnerStatus.INITIAL_GP_FIT_FAILED
    assert result.physical_accounting["simulator_attempts"] == 4
    assert result.physical_accounting["gp_fit_attempts"] == 5
    assert result.proposals == ()
    assert result.recommendation is None


def test_repeated_execution_is_scientifically_deterministic() -> None:
    simulator = HandwrittenAnalyticSimulator()
    config = SequentialRunnerConfig(_intent(simulator))
    first = run_sequential(simulator, config=config)
    second = run_sequential(simulator, config=config)
    assert first.status is RunnerStatus.COMPLETED
    assert _science_projection(first) == _science_projection(second)


@dataclass
class _RetryOnceSimulator:
    delegate: HandwrittenAnalyticSimulator
    calls: int = 0

    @property
    def capabilities(self):
        return self.delegate.capabilities

    def evaluate(self, theta: torch.Tensor, context=None) -> SimulationResult:
        self.calls += 1
        if self.calls == 1:
            raise RetryableSimulationError(
                RetryableSimulationReason.TRANSIENT_RESOURCE,
                "test-only transient",
            )
        return self.delegate.evaluate(theta, context)


def test_retryable_failure_reuses_scientific_context_and_counts_attempts() -> None:
    delegate = HandwrittenAnalyticSimulator()
    simulator = _RetryOnceSimulator(delegate)
    result = run_sequential(
        simulator,
        config=SequentialRunnerConfig(
            _intent(delegate, rounds=0),
            max_attempts_per_logical_evaluation=2,
        ),
    )
    assert result.status is RunnerStatus.COMPLETED
    assert result.physical_accounting["simulator_attempts"] == 5
    assert result.physical_accounting["simulator_retry_attempts"] == 1
    attempts = result.observations[0].attempts
    assert [item.status.value for item in attempts] == [
        "retryable_failure",
        "success",
    ]
    assert result.observations[0].context.attempt_index == 1


def test_unclassified_simulator_error_is_not_retried() -> None:
    class Broken(_RetryOnceSimulator):
        def evaluate(self, theta: torch.Tensor, context=None) -> SimulationResult:
            raise RuntimeError("ordinary failure")

    delegate = HandwrittenAnalyticSimulator()
    result = run_sequential(
        Broken(delegate),
        config=SequentialRunnerConfig(
            _intent(delegate, rounds=0),
            max_attempts_per_logical_evaluation=3,
        ),
    )
    assert result.status is RunnerStatus.SIMULATOR_RUNTIME_FAILED
    assert result.physical_accounting["simulator_attempts"] == 1


def test_total_attempt_budget_and_progress_failures_are_operational_only() -> None:
    simulator = HandwrittenAnalyticSimulator()
    config = SequentialRunnerConfig(
        _intent(simulator, rounds=0),
        max_total_simulator_attempts=2,
    )
    failed_events = 0

    def broken_progress(_event) -> None:
        nonlocal failed_events
        failed_events += 1
        raise RuntimeError("display unavailable")

    result = run_sequential(
        simulator,
        config=config,
        progress_callback=broken_progress,
    )
    assert result.status is RunnerStatus.SIMULATOR_TOTAL_ATTEMPT_BUDGET_EXHAUSTED
    assert result.logical_accounting["committed_observations"] == 2
    assert result.physical_accounting["simulator_attempts"] == 2
    assert len(result.progress_callback_failures) == failed_events
    assert all("display unavailable" in item for item in result.progress_callback_failures)


def test_reliability_controls_change_execution_but_not_scientific_identity() -> None:
    simulator = HandwrittenAnalyticSimulator()
    intent = _intent(simulator, rounds=0)
    default = run_sequential(
        simulator,
        config=SequentialRunnerConfig(intent),
    )
    retry_enabled = run_sequential(
        simulator,
        config=SequentialRunnerConfig(
            intent,
            max_attempts_per_logical_evaluation=2,
            max_total_simulator_attempts=8,
        ),
    )
    assert default.status is RunnerStatus.COMPLETED
    assert retry_enabled.status is RunnerStatus.COMPLETED
    assert default.scientific_identity == retry_enabled.scientific_identity
    assert default.execution_identity != retry_enabled.execution_identity
    assert _science_projection(default) == _science_projection(retry_enabled)


def test_observation_snapshot_is_neutral_and_contains_no_new_work() -> None:
    simulator = HandwrittenAnalyticSimulator()
    intent = _intent(simulator, rounds=1)
    baseline = run_sequential(simulator, config=SequentialRunnerConfig(intent))
    replay = run_sequential(
        HandwrittenAnalyticSimulator(),
        config=SequentialRunnerConfig(
            intent,
            snapshot_observation_counts=(intent.initial_point_count,),
        ),
    )
    assert replay.observation_snapshots is not None
    snapshot = replay.observation_snapshots[intent.initial_point_count]
    assert snapshot.provenance()["simulator_calls"] == 0
    assert snapshot.provenance()["gp_fits"] == 0
    assert [item.observation_id for item in baseline.observations] == [
        item.observation_id for item in replay.observations
    ]
    assert [item.candidate_id for item in baseline.proposals] == [
        item.candidate_id for item in replay.proposals
    ]


@pytest.mark.cuda
def test_runner_cuda_contract_when_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable on this machine")
    simulator = HandwrittenAnalyticSimulator(device=torch.device("cuda"))
    result = run_sequential(
        simulator,
        config=SequentialRunnerConfig(
            _intent(simulator, rounds=0),
            device=torch.device("cuda"),
        ),
    )
    assert result.status is RunnerStatus.COMPLETED
    assert all(item.theta.device.type == "cuda" for item in result.observations)
