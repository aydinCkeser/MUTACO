"""Pinned single-round NLE/NRE training and physical sample-MAP plumbing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from experiments.contracts import PhysicalCallCounts
from .direct_posterior_decision import (
    SBI_ZOO_PHYSICAL_SCORE_API,
    SBI_ZOO_SCORE_IDENTITIES,
    PosteriorSampleMapResult,
    posterior_sample_map_decision,
)
from .sbi_zoo_contract import (
    METHODS,
    POSTERIOR_SAMPLE_COUNT,
    PRIOR_LAW,
    estimator_contract,
    mcmc_contract,
    require_pinned_runtime,
)
from .sbi_zoo_dataset import SharedTrainingDataset
from .sbi_zoo_failures import (
    InsufficientPosteriorSamples,
    MCMCInitializationFailure,
    MCMCSamplingFailure,
    NonfinitePosteriorScore,
    PosteriorConstructionFailure,
    PosteriorSupportViolation,
    TrainingFailure,
)


@dataclass(frozen=True)
class TrainedSbiZooEstimator:
    method: str
    budget: int
    inference: Any
    estimator: Any
    prior: Any
    dataset_receipt: dict[str, Any]
    effective_configuration: dict[str, Any]
    training_seed: int


@dataclass(frozen=True)
class SbiZooRecommendation:
    method: str
    samples: torch.Tensor
    sample_map: PosteriorSampleMapResult
    recommendation_seed: int
    mcmc_configuration: dict[str, Any]


def _lazy_sbi() -> tuple[Any, Any, Any, Any, Any, Any]:
    require_pinned_runtime()
    from sbi.inference import NLE, NRE
    from sbi.inference.posteriors.posterior_parameters import MCMCPosteriorParameters
    from sbi.neural_nets import classifier_nn, likelihood_nn
    from sbi.utils import BoxUniform

    return NLE, NRE, MCMCPosteriorParameters, classifier_nn, likelihood_nn, BoxUniform


def authoritative_prior() -> Any:
    """Return the physical independent uniform prior used by historical SBI."""

    *_, BoxUniform = _lazy_sbi()
    lower, upper = PRIOR_LAW.bounds()
    return BoxUniform(lower, upper, device="cpu")


def _builders(method: str) -> tuple[Any, str, Any]:
    NLE, NRE, _, classifier_nn, likelihood_nn, _ = _lazy_sbi()
    if method == "sbi_nle_single_round":
        builder = likelihood_nn(
            model="maf",
            z_score_theta="independent",
            z_score_x="independent",
            hidden_features=50,
            num_transforms=5,
            embedding_net=nn.Identity(),
            num_blocks=2,
            dropout_probability=0.0,
            use_batch_norm=False,
        )
        return NLE, "density_estimator", builder
    if method == "sbi_nre_single_round":
        builder = classifier_nn(
            model="mlp",
            z_score_theta="independent",
            z_score_x="independent",
            hidden_features=50,
            embedding_net_theta=nn.Identity(),
            embedding_net_x=nn.Identity(),
            norm_layer=nn.LayerNorm,
        )
        return NRE, "classifier", builder
    raise ValueError("unknown SBI-zoo method")


def build_inference(
    method: str, *, summary_writer: Any, prior: Any | None = None
) -> tuple[Any, Any]:
    """Construct one explicit pinned inference object without training it."""

    inference_class, estimator_key, builder = _builders(method)
    resolved_prior = authoritative_prior() if prior is None else prior
    inference = inference_class(
        prior=resolved_prior,
        device="cpu",
        logging_level="WARNING",
        show_progress_bars=False,
        summary_writer=summary_writer,
        tracker=None,
        **{estimator_key: builder},
    )
    return inference, resolved_prior


def train_single_round(
    dataset: SharedTrainingDataset,
    *,
    method: str,
    training_seed: int,
    training_log_dir: Path,
) -> TrainedSbiZooEstimator:
    """Fit once on the exact persisted shared bytes; no simulator is reachable."""

    if method not in METHODS:
        raise ValueError("unknown SBI-zoo method")
    budget = int(dataset.metadata["budget"])
    config = estimator_contract(method, budget)
    from torch.utils.tensorboard import SummaryWriter

    if Path(training_log_dir).exists():
        raise TrainingFailure("training log directory must be a new append-only path")
    writer = SummaryWriter(log_dir=str(training_log_dir))
    try:
        inference, prior = build_inference(method, summary_writer=writer)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(training_seed)
            inference.append_simulations(dataset.theta, dataset.response)
            train_kwargs = {
                "training_batch_size": min(64, budget),
                "learning_rate": 5e-4,
                "validation_fraction": 0.2,
                "stop_after_epochs": 20,
                # sbi 0.26.1 loops while epoch <= max_num_epochs. Passing 199
                # enforces the frozen scientific cap of 200 completed epochs.
                "max_num_epochs": 199,
                "clip_max_norm": 5.0,
                "resume_training": False,
                "discard_prior_samples": False,
                "retrain_from_scratch": False,
                "show_train_summary": False,
                "dataloader_kwargs": {
                    "num_workers": 0,
                    "pin_memory": False,
                    "timeout": 0,
                    "persistent_workers": False,
                    "collate_fn": None,
                    "worker_init_fn": None,
                    "multiprocessing_context": None,
                    "generator": None,
                    "prefetch_factor": None,
                    "pin_memory_device": "",
                    "in_order": True,
                },
            }
            if method == "sbi_nre_single_round":
                train_kwargs["num_atoms"] = 2
            estimator = inference.train(**train_kwargs)
    except Exception as exc:
        raise TrainingFailure(f"{method} training failed: {exc}") from exc
    finally:
        writer.close()
    return TrainedSbiZooEstimator(
        method=method,
        budget=budget,
        inference=inference,
        estimator=estimator,
        prior=prior,
        dataset_receipt=dataset.consumer_receipt(method),
        effective_configuration=config,
        training_seed=training_seed,
    )


def build_mcmc_posterior(trained: TrainedSbiZooEstimator, target_response: torch.Tensor) -> Any:
    """Build the frozen MCMC posterior and condition it on one physical response."""

    if target_response.shape != (20,) or target_response.dtype is not torch.float32:
        raise ValueError("target response must be one CPU float32 20-vector")
    if target_response.device.type != "cpu" or not bool(torch.isfinite(target_response).all()):
        raise ValueError("target response must be finite on CPU")
    _, _, MCMCPosteriorParameters, *_ = _lazy_sbi()
    controls = mcmc_contract()
    parameters = MCMCPosteriorParameters(
        method=controls["method"],
        thin=controls["thin"],
        warmup_steps=controls["warmup_steps"],
        num_chains=controls["num_chains"],
        init_strategy=controls["init_strategy"],
        init_strategy_parameters={},
        num_workers=controls["num_workers"],
        mp_context=controls["mp_context"],
    )
    try:
        posterior = trained.inference.build_posterior(
            trained.estimator,
            prior=trained.prior,
            sample_with="mcmc",
            posterior_parameters=parameters,
        )
        return posterior.set_default_x(target_response.contiguous())
    except Exception as exc:
        raise PosteriorConstructionFailure(
            f"{trained.method} posterior construction failed: {exc}"
        ) from exc


def _install_initialization_guard(posterior: Any) -> None:
    """Classify pinned private initialization failures without changing draws."""

    original = posterior._get_initial_params

    def guarded(*args: Any, **kwargs: Any) -> torch.Tensor:
        try:
            values = original(*args, **kwargs)
        except Exception as exc:
            raise MCMCInitializationFailure(str(exc)) from exc
        if not isinstance(values, torch.Tensor) or values.shape != (20, 4):
            raise MCMCInitializationFailure("expected one four-vector for each of 20 chains")
        if not bool(torch.isfinite(values).all()):
            raise MCMCInitializationFailure("proposal initialization is nonfinite")
        return values

    posterior._get_initial_params = guarded


def sample_exact_posterior_candidates(
    posterior: Any,
    *,
    recommendation_seed: int,
) -> torch.Tensor:
    """Draw exactly 10,000 total physical candidates in pinned chain-major order."""

    _install_initialization_guard(posterior)
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(recommendation_seed)
            np.random.seed(recommendation_seed % (1 << 32))
            samples = posterior.sample(
                (POSTERIOR_SAMPLE_COUNT,),
                show_progress_bars=False,
            )
    except MCMCInitializationFailure:
        raise
    except Exception as exc:
        raise MCMCSamplingFailure(str(exc)) from exc
    finally:
        np.random.set_state(numpy_state)
    if not isinstance(samples, torch.Tensor) or samples.shape != (POSTERIOR_SAMPLE_COUNT, 4):
        raise InsufficientPosteriorSamples(
            "sampler did not return exactly 10,000 total four-dimensional samples"
        )
    samples = samples.to(device="cpu", dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(samples).all()):
        raise InsufficientPosteriorSamples("retained posterior samples contain nonfinite values")
    lower, upper = PRIOR_LAW.bounds()
    if not bool(((samples >= lower) & (samples <= upper)).all()):
        raise PosteriorSupportViolation("retained posterior samples violate physical support")
    return samples


def physical_posterior_score(posterior: Any, theta: torch.Tensor) -> torch.Tensor:
    """Evaluate NLE likelihood+prior or NRE ratio+prior in physical theta.

    ``MCMCPosterior.potential`` calls the underlying physical potential directly.
    It does not use ``_prepare_potential`` and therefore does not add the
    unconstraining transform Jacobian used internally by the sampler.
    """

    scores = posterior.potential(theta, track_gradients=False)
    if not isinstance(scores, torch.Tensor):
        raise NonfinitePosteriorScore("physical posterior potential did not return a tensor")
    flattened = scores.reshape(-1)
    if flattened.numel() != theta.shape[0] or not bool(torch.isfinite(flattened).any()):
        raise NonfinitePosteriorScore("no finite physical-coordinate posterior score")
    return flattened


def recommend_sample_map(
    trained: TrainedSbiZooEstimator,
    posterior: Any,
    *,
    recommendation_seed: int,
    method_configuration_identity: str,
) -> SbiZooRecommendation:
    samples = sample_exact_posterior_candidates(
        posterior, recommendation_seed=recommendation_seed
    )
    lower, upper = PRIOR_LAW.bounds()
    result = posterior_sample_map_decision(
        method=trained.method,
        posterior=posterior,
        samples=samples,
        lower=lower,
        upper=upper,
        method_configuration_identity=method_configuration_identity,
        inference_budget=trained.budget,
        inference_calls=PhysicalCallCounts(trained.budget, trained.budget),
        physical_score=lambda theta: physical_posterior_score(posterior, theta),
        physical_score_identity=SBI_ZOO_SCORE_IDENTITIES[trained.method],
        physical_score_api=SBI_ZOO_PHYSICAL_SCORE_API,
    )
    return SbiZooRecommendation(
        method=trained.method,
        samples=samples,
        sample_map=result,
        recommendation_seed=recommendation_seed,
        mcmc_configuration=mcmc_contract(),
    )
