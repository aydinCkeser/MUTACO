"""D=4 adapter for the approved scientific graph-oscillator forward model.

This adapter deliberately exposes only the first Prompt 8 campaign surface:
the scientific P=4/M=20 model, independently keyed evaluations, and a
scrambled-Sobol hidden-target ledger.  It changes no graph physics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from mutaco import ScientificIdentity, SeedCoordinate, SeedDeriver
from mutaco_graph import (
    SCIENTIFIC_IDENTITY,
    SCIENTIFIC_M20,
    GraphSimulationConfig,
    OverdampedScientificSimulator,
    parameter_profile,
)

from ..contracts import AdapterTargetBank, MutacoBudget


class GraphScientificD4Adapter:
    """Benchmark adapter for the approved P=4 scientific graph model."""

    identifier = "graph_scientific_d4"
    _PARAMETER_PROFILE = "p4-paper-baseline"

    def target_bank(
        self,
        config: Mapping[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> AdapterTargetBank:
        graph_config = self._graph_config(config, device=device, dtype=dtype)
        campaign = self._campaign(config)
        count = self._positive_int(campaign.get("target_count"), "target_count")
        if campaign.get("target_bank_policy") != "scrambled_sobol":
            raise ValueError("graph d=4 target bank requires scrambled_sobol")

        profile = parameter_profile(graph_config.parameter_profile_id)
        lower = torch.tensor(
            profile.parameter_spec.lower, dtype=dtype, device=device
        )
        upper = torch.tensor(
            profile.parameter_spec.upper, dtype=dtype, device=device
        )
        randomness = config.get("randomness")
        assert isinstance(randomness, Mapping)
        family = ScientificIdentity.create(
            {
                "schema": "graph-d4-target-family-v1",
                "seed_derivation_policy": randomness.get("seed_policy"),
                "target_bank": {
                    "policy": "scrambled_sobol",
                    "family_seed": campaign["campaign_seed"],
                    "parameter_profile": profile.identifier,
                    "parameter_names": profile.parameter_names,
                    "bounds": [
                        list(profile.parameter_spec.lower),
                        list(profile.parameter_spec.upper),
                    ],
                },
                "simulator": graph_config.identity_document(),
                "target_response_seed_namespace": randomness.get(
                    "target_seed_namespace"
                ),
                "target_construction_policy": "one_context_evaluation_v1",
                "dtype": "float32",
            }
        )
        seed = SeedDeriver(family, int(campaign["campaign_seed"])).seed_for(
            SeedCoordinate("target_bank", ("scrambled_sobol",))
        )
        engine = torch.quasirandom.SobolEngine(
            profile.dimension, scramble=True, seed=seed
        )
        unit_cube = engine.draw(count, dtype=dtype).to(device).contiguous()
        theta = (lower + (upper - lower) * unit_cube).contiguous()
        if torch.unique(theta, dim=0).shape[0] != count:
            raise RuntimeError("graph d=4 target bank contains duplicate theta values")
        declaration = config.get("target_declaration")
        if declaration is not None:
            if not isinstance(declaration, Mapping):
                raise ValueError("target_declaration must be a mapping")
            root = Path(__file__).resolve().parents[3]
            path = (root / str(declaration.get("path", ""))).resolve()
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise ValueError("target declaration must remain inside the repository") from exc
            sidecar = path.with_suffix(".sha256")
            if not path.is_file() or not sidecar.is_file():
                raise ValueError("target declaration and hash sidecar must exist")
            if hashlib.sha256(path.read_bytes()).hexdigest() != sidecar.read_text(
                encoding="ascii"
            ).strip():
                raise ValueError("target declaration hash mismatch")
            ledger = json.loads(path.read_text(encoding="utf-8"))
            declared = torch.tensor(
                [item["theta_star"] for item in ledger.get("records", ())],
                device=device,
                dtype=dtype,
            ).contiguous()
            if (
                ledger.get("target_family_id") != family.value
                or ledger.get("target_count") != count
                or not torch.equal(theta, declared)
            ):
                raise ValueError("frozen target declaration does not match generated bank")

        return AdapterTargetBank(
            theta=theta,
            seed=seed,
            science=family,
            parameter_names=profile.parameter_names,
            bounds=(
                tuple(float(value) for value in profile.parameter_spec.lower),
                tuple(float(value) for value in profile.parameter_spec.upper),
            ),
        )

    def build_simulator(
        self,
        config: Mapping[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> OverdampedScientificSimulator:
        return OverdampedScientificSimulator(
            self._graph_config(config, device=device, dtype=dtype)
        )

    def build_target_simulator(
        self,
        config: Mapping[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> OverdampedScientificSimulator:
        """Construct only the declared forward model for target freezing.

        Target construction has no method, comparison, or inference-budget
        policy.  The ordinary campaign constructor remains stricter and
        validates those independent campaign concerns below.
        """

        return OverdampedScientificSimulator(
            self.target_graph_config(config, device=device, dtype=dtype)
        )

    def latent_response(self, theta: torch.Tensor) -> None:
        """The physical stochastic graph response has no separate latent oracle."""

        del theta
        return None

    @classmethod
    def _campaign(cls, config: Mapping[str, Any]) -> Mapping[str, Any]:
        campaign = config.get("campaign")
        if not isinstance(campaign, Mapping):
            raise ValueError("graph d=4 campaign requires a campaign mapping")
        cls._positive_int(campaign.get("target_count"), "target_count")
        cls._nonnegative_int(campaign.get("campaign_seed"), "campaign_seed")
        return campaign

    @classmethod
    def _graph_config(
        cls,
        config: Mapping[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> GraphSimulationConfig:
        randomness = config.get("randomness")
        if not isinstance(randomness, Mapping) or (
            randomness.get("no_cross_theta_common_random_numbers") is not True
        ):
            raise ValueError(
                "graph d=4 campaign must declare no cross-theta common random numbers"
            )
        comparison = config.get("comparison")
        if not isinstance(comparison, Mapping):
            raise ValueError("graph d=4 campaign requires a comparison mapping")
        if comparison.get("primary_basis") != "equal_simulator_calls":
            raise ValueError("graph d=4 campaign requires equal simulator calls")
        if comparison.get("include_target_generation_in_budget") is not False:
            raise ValueError("target generation must remain outside inference budgets")
        if comparison.get("held_out_assessment_calls") != 1:
            raise ValueError("graph d=4 campaign requires one held-out assessment")

        MutacoBudget.resolve(config)
        return cls.target_graph_config(config, device=device, dtype=dtype)

    @classmethod
    def target_graph_config(
        cls,
        config: Mapping[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> GraphSimulationConfig:
        """Resolve graph physics without importing method/campaign policy."""

        if config.get("dtype") != "float32" or dtype is not torch.float32:
            raise ValueError("the approved graph d=4 benchmark requires float32")
        simulator = config.get("simulator")
        if not isinstance(simulator, Mapping):
            raise ValueError("graph d=4 target construction requires a simulator mapping")
        if simulator.get("adapter") != cls.identifier:
            raise ValueError("graph d=4 target construction uses the wrong simulator adapter")
        if simulator.get("identifier") != SCIENTIFIC_IDENTITY:
            raise ValueError("graph d=4 target construction requires the scientific simulator identity")
        if simulator.get("parameter_profile") != cls._PARAMETER_PROFILE:
            raise ValueError("graph d=4 target construction requires p4-paper-baseline")
        if simulator.get("response_profile") != SCIENTIFIC_M20.identifier:
            raise ValueError("graph d=4 target construction requires the canonical M=20 response")
        return GraphSimulationConfig(
            model_identity=str(simulator["identifier"]),
            parameter_profile_id=str(simulator["parameter_profile"]),
            response_profile_id=str(simulator["response_profile"]),
            node_count=cls._positive_int(simulator.get("node_count"), "node_count"),
            batch_size=cls._positive_int(simulator.get("batch_size"), "batch_size"),
            repeat_size=cls._positive_int(
                simulator.get("repeat_size"), "repeat_size"
            ),
            temperature=float(simulator["temperature"]),
            dt=float(simulator["dt"]),
            burn_in_steps=cls._nonnegative_int(
                simulator.get("burn_in_steps"), "burn_in_steps"
            ),
            sample_steps=cls._positive_int(
                simulator.get("sample_steps"), "sample_steps"
            ),
            sample_stride=cls._positive_int(
                simulator.get("sample_stride"), "sample_stride"
            ),
            device=str(device),
            dtype=dtype,
        )

    @staticmethod
    def _positive_int(value: object, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        return value

    @staticmethod
    def _nonnegative_int(value: object, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
        return value
