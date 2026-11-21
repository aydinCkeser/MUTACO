"""Resolved scientific controls for single-round NLE/NRE.

All effective settings are explicit for pinned ``sbi==0.26.1``.  Nothing in
this module constructs a simulator, trains an estimator, or samples a
posterior.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

import torch

from mutaco import ScientificIdentity, SeedCoordinate, SeedDeriver


SBI_VERSION = "0.26.1"
METHODS = ("sbi_nle_single_round", "sbi_nre_single_round")
BUDGETS = (40, 100)
PARAMETER_NAMES = (
    "weight_mean",
    "pairwise_coupling_mean",
    "bias_scale_mean",
    "pairwise_coupling_CV",
)
PARAMETER_LOWER = (0.5, 0.5, 0.5, 0.0)
PARAMETER_UPPER = (2.0, 2.0, 2.0, 1.0)
RESPONSE_DIMENSION = 20
POSTERIOR_SAMPLE_COUNT = 10_000
SLOT_OUTCOME_SCHEMA = "mutaco-sbi-zoo-slot-outcome-v1"
SBI_ZOO_PHYSICAL_SCORE_API = (
    "sbi_mcmc_posterior_potential_physical_no_transform_jacobian_v1"
)
SBI_ZOO_SCORE_IDENTITIES = {
    "sbi_nle_single_round": "sbi-0.26.1-likelihood_plus_prior-physical-theta-v1",
    "sbi_nre_single_round": "sbi-0.26.1-ratio_plus_prior-physical-theta-v1",
}


def require_pinned_runtime() -> dict[str, str]:
    """Reject an unpinned SBI runtime and return identity-bearing versions."""

    actual = version("sbi")
    if actual != SBI_VERSION:
        raise RuntimeError(f"SBI-zoo requires sbi=={SBI_VERSION}, found {actual}")
    return {
        "python_package_sbi": actual,
        "torch": torch.__version__,
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "nflows": version("nflows"),
    }


@dataclass(frozen=True)
class PriorPredictiveThetaLaw:
    """Authoritative independent physical-box prior and byte-generation law."""

    identifier: str = "paper120_sbi_zoo_physical_box_uniform_float32_v1"
    distribution: str = "independent_continuous_uniform_physical_box"
    parameter_names: tuple[str, ...] = PARAMETER_NAMES
    lower: tuple[float, ...] = PARAMETER_LOWER
    upper: tuple[float, ...] = PARAMETER_UPPER
    support: str = "closed_box_for_density_and_validation"
    sampling_interval: str = "torch_rand_half_open_unit_interval_scaled_to_box"
    sampling_implementation: str = "torch.Generator(cpu).manual_seed_then_torch.rand"
    source_dtype: str = "float32"
    conversion_to_float32: str = "bounds_created_float32_and_rand_generated_float32"
    invalid_sample_policy: str = "typed_failure_no_rejection_no_resampling"
    nonfinite_policy: str = "forbidden_typed_failure"
    cross_budget_policy: str = "budget_is_seed_coordinate_b40_b100_independent_not_nested"

    def document(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "distribution": self.distribution,
            "parameter_names": list(self.parameter_names),
            "lower": list(self.lower),
            "upper": list(self.upper),
            "support": self.support,
            "sampling_interval": self.sampling_interval,
            "sampling_implementation": self.sampling_implementation,
            "source_dtype": self.source_dtype,
            "conversion_to_float32": self.conversion_to_float32,
            "invalid_sample_policy": self.invalid_sample_policy,
            "nonfinite_policy": self.nonfinite_policy,
            "cross_budget_policy": self.cross_budget_policy,
            "seed_policy": "mutaco-coordinate-sha256-v1",
            "seed_namespace": "sbi_zoo_shared_dataset_theta",
            "seed_coordinates": ["target_record_id", "budget"],
        }

    def bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.lower, dtype=torch.float32, device="cpu"),
            torch.tensor(self.upper, dtype=torch.float32, device="cpu"),
        )

    def sample(
        self,
        count: int,
        *,
        deriver: SeedDeriver,
        target_record_id: str,
        budget: int,
    ) -> tuple[torch.Tensor, int]:
        if budget not in BUDGETS or count != budget:
            raise ValueError("shared prior-predictive count must equal B40 or B100")
        lower, upper = self.bounds()
        seed = deriver.seed_for(
            SeedCoordinate(
                "sbi_zoo_shared_dataset_theta", (target_record_id, budget)
            )
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        theta = lower + torch.rand(
            (count, len(self.parameter_names)),
            dtype=torch.float32,
            device="cpu",
            generator=generator,
        ) * (upper - lower)
        theta = theta.contiguous()
        if not bool(torch.isfinite(theta).all()):
            raise ValueError("prior-predictive theta contains a nonfinite value")
        if not bool(((theta >= lower) & (theta <= upper)).all()):
            raise ValueError("prior-predictive theta lies outside closed support")
        return theta, seed


PRIOR_LAW = PriorPredictiveThetaLaw()


def dataset_seed_deriver(scientific_identity: str, root_seed: int) -> SeedDeriver:
    """Create the future declaration-owned dataset seed deriver."""

    return SeedDeriver(ScientificIdentity(scientific_identity), root_seed)


def randomness_contract() -> dict[str, Any]:
    """Return disjoint coordinate-addressed randomness namespaces."""

    return {
        "policy": "mutaco-coordinate-sha256-v1",
        "dataset": {
            "theta": ["sbi_zoo_shared_dataset_theta", "target_record_id", "budget"],
            "simulation_row": [
                "sbi_zoo_shared_dataset_simulation_row",
                "target_record_id",
                "budget",
                "row_index",
            ],
        },
        "training": ["sbi_zoo_estimator_training", "method", "target_record_id", "budget"],
        "recommendation": [
            "sbi_zoo_posterior_sampling",
            "method",
            "target_record_id",
            "budget",
        ],
        "heldout": "separate_assessment_declaration_only_never_a_training_seed",
    }


def estimator_seed(
    deriver: SeedDeriver, *, method: str, target_record_id: str, budget: int
) -> int:
    if method not in METHODS or budget not in BUDGETS:
        raise ValueError("unknown estimator seed coordinate")
    return deriver.seed_for(
        SeedCoordinate(
            "sbi_zoo_estimator_training", (method, target_record_id, budget)
        )
    )


def recommendation_seed(
    deriver: SeedDeriver, *, method: str, target_record_id: str, budget: int
) -> int:
    if method not in METHODS or budget not in BUDGETS:
        raise ValueError("unknown recommendation seed coordinate")
    return deriver.seed_for(
        SeedCoordinate(
            "sbi_zoo_posterior_sampling", (method, target_record_id, budget)
        )
    )


def estimator_contract(method: str, budget: int) -> dict[str, Any]:
    """Complete effective estimator and training identity for sbi 0.26.1."""

    if method not in METHODS or budget not in BUDGETS:
        raise ValueError("unknown frozen SBI-zoo method or budget")
    shared = {
        "sbi_version": SBI_VERSION,
        "executed_design": "single_round_one_fit_on_shared_prior_predictive_dataset",
        "device": "cpu",
        "dtype": "float32",
        "inference_constructor": {
            "logging_level": "WARNING",
            "summary_writer": "explicit_artifact_path",
            "tracker": None,
            "show_progress_bars": False,
        },
        "learning_rate": 5e-4,
        "validation_fraction": 0.2,
        "requested_training_batch_size": min(64, budget),
        "training_rows": int(0.8 * budget),
        "validation_rows": budget - int(0.8 * budget),
        "effective_training_batch_size": min(min(64, budget), int(0.8 * budget)),
        "effective_validation_batch_size": min(
            min(64, budget), budget - int(0.8 * budget)
        ),
        "validation_split": "torch.randperm_under_method_training_seed_first_floor_0p8B_rows",
        "train_sampler": "torch.utils.data.SubsetRandomSampler_seeded_ambient_fork_rng",
        "validation_sampler": "torch.utils.data.SubsetRandomSampler_seeded_ambient_fork_rng",
        "train_shuffle": "sampler_random_order_each_epoch",
        "validation_shuffle": "sampler_random_order_each_epoch_despite_shuffle_false",
        "drop_last": True,
        "dataloader": {
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
        "optimizer": {
            "class": "torch.optim.Adam",
            "lr": 5e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0,
            "amsgrad": False,
            "foreach": None,
            "maximize": False,
            "capturable": False,
            "differentiable": False,
            "fused": None,
            "decoupled_weight_decay": False,
        },
        "scheduler": None,
        "gradient_clip_max_norm": 5.0,
        "stop_after_epochs": 20,
        "early_stopping": "strict_validation_loss_improvement_restore_best_after_20_nonimproving_epochs",
        "scientific_maximum_epochs": 200,
        "sbi_api_max_num_epochs_argument": 199,
        "max_epoch_translation": "sbi_0p26p1_loop_is_inclusive_epoch_le_argument_so_199_caps_at_200_epochs",
        "resume_training": False,
        "discard_prior_samples": False,
        "retrain_from_scratch": False,
        "show_train_summary": False,
        "show_progress_bars": False,
        "summary_writer": "explicit_torch_utils_tensorboard_SummaryWriter_at_training_artifact_path",
        "automatic_retry": False,
        "fallback": None,
    }
    if method == "sbi_nle_single_round":
        model = {
            "family": "maf_likelihood",
            "trainer_class": "sbi.inference.trainers.nle.nle_a.NLE_A",
            "objective": "mean_negative_conditional_log_likelihood_x_given_theta",
            "builder": "sbi.neural_nets.likelihood_nn",
            "model": "maf",
            "z_score_theta": "independent",
            "z_score_x": "independent",
            "hidden_features": 50,
            "num_transforms": 5,
            "num_blocks_per_autoregressive_transform": 2,
            "activation": "torch.tanh",
            "random_mask": False,
            "use_residual_blocks": False,
            "dropout_probability": 0.0,
            "use_batch_norm": False,
            "between_transform_permutation": "nflows.RandomPermutation",
            "embedding_net": "torch.nn.Identity",
            "base_distribution": "nflows.StandardNormal",
            "ignored_public_builder_defaults": {
                "num_bins_not_used_by_maf": 10,
                "num_components_not_used_by_maf": 10,
            },
        }
    else:
        model = {
            "family": "mlp_ratio_classifier",
            "trainer_class": "sbi.inference.trainers.nre.nre_b.NRE_B",
            "objective": "atomic_ratio_classification_with_two_atoms",
            "builder": "sbi.neural_nets.classifier_nn",
            "model": "mlp",
            "z_score_theta": "independent",
            "z_score_x": "independent",
            "hidden_features": 50,
            "architecture": [
                "Linear(24,50)",
                "LayerNorm(50)",
                "ReLU",
                "Linear(50,50)",
                "LayerNorm(50)",
                "ReLU",
                "Linear(50,1)",
            ],
            "embedding_net_theta": "torch.nn.Identity",
            "embedding_net_x": "torch.nn.Identity",
            "num_atoms": 2,
            "ratio_orientation": "log_p_theta_x_minus_log_p_theta_minus_log_p_x",
        }
    return {"method": method, "budget": budget, "model": model, "training": shared}


def mcmc_contract() -> dict[str, Any]:
    """Fully resolved retained-sample behavior of pinned sbi 0.26.1."""

    return {
        "sbi_version": SBI_VERSION,
        "method": "slice_np_vectorized",
        "num_chains": 20,
        "warmup_steps": 100,
        "thin": 1,
        "num_workers": 1,
        "mp_context": "spawn",
        "init_strategy": "proposal",
        "initialization_distribution": "authoritative_physical_BoxUniform_prior",
        "initialization_draws_per_chain": 1,
        "initialization_candidate_count": None,
        "initialization_order": "20_sequential_proposal_draws_in_chain_index_order",
        "initialization_seed": "single_method_target_budget_recommendation_seed_controls_all_draws",
        "torch_seed_mapping": "full_63_bit_named_recommendation_seed",
        "numpy_seed_mapping": "named_recommendation_seed_modulo_2_pow_32",
        "ambient_rng_isolation": "restore_torch_and_numpy_states_after_sampling",
        "invalid_initial_state": "typed_MCMCInitializationFailure_no_retry_no_fallback",
        "internal_coordinate_system": "unconstrained_by_sbi_theta_transform",
        "internal_potential_jacobian": True,
        "returned_coordinate_system": "physical_theta_after_theta_transform_inverse",
        "requested_total_samples": POSTERIOR_SAMPLE_COUNT,
        "samples_per_chain": 500,
        "aggregation": "sbi_chain_major_C_order_reshape_then_first_10000",
        "ordered_candidate_index": "chain_index_times_500_plus_within_chain_retained_index",
        "insufficient_or_nonfinite_samples": "typed_failure_no_retry_no_fallback",
        "chain_failure": "aggregate_recommendation_failure_no_partial_chain_acceptance",
        "final_score_coordinate_system": "physical_theta",
        "final_score_api": "MCMCPosterior.potential_physical_without_transform_jacobian",
    }


def durable_outcome_contract() -> dict[str, Any]:
    return {
        "schema_version": SLOT_OUTCOME_SCHEMA,
        "publication": "append_only_complete_sibling_directory_atomic_rename",
        "success_gate": "persist_reload_rehash_validate_then_append_success",
        "post_invocation_failure": "attempted_outcome_unknown_never_failed",
        "implicit_unknown": "durable_attempted_without_terminal_event",
        "orphan_policy": "retain_for_audit_never_reconcile_reuse_or_assemble",
        "dataset_assembly": "exactly_one_valid_ledger_success_artifact_per_ordered_row",
        "hash": "sha256_of_u64be_length_prefixed_metadata_then_response",
    }


def complete_implementation_contract() -> dict[str, Any]:
    return {
        "runtime": require_pinned_runtime(),
        "prior_predictive_theta_law": PRIOR_LAW.document(),
        "randomness": randomness_contract(),
        "estimators": [
            estimator_contract(method, budget)
            for budget in BUDGETS
            for method in METHODS
        ],
        "mcmc": mcmc_contract(),
        "durable_outcomes": durable_outcome_contract(),
    }
