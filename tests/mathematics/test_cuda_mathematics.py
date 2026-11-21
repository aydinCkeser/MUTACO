from __future__ import annotations

import pytest
import torch

from mutaco import (
    LatentPosteriorSummary,
    MatchedBarrierObjective,
    StandardizedTarget,
    evaluate_exploration,
)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="compatible Torch CUDA is unavailable")
def test_mathematical_kernel_remains_on_cuda() -> None:
    device = torch.device("cuda")
    mean = torch.tensor([0.2, 0.1], dtype=torch.float64, device=device)
    covariance = torch.eye(2, dtype=torch.float64, device=device)
    posterior = LatentPosteriorSummary(mean, covariance, "tx", "cuda-model")
    target = StandardizedTarget(
        torch.zeros(2, dtype=torch.float64, device=device), ("a", "b"), "tx"
    )

    objective = MatchedBarrierObjective(alpha=1.0, tau_interior=0.0).evaluate(
        posterior, target
    )
    exploration = evaluate_exploration(posterior)
    assert objective.values.device.type == "cuda"
    assert exploration.values.device.type == "cuda"
