from pathlib import Path

import numpy as np
import pytest
import torch

from autoencoder.demo import classify, outcome, reconstruct_image

ROOT = Path(__file__).resolve().parents[1]


class Identity(torch.nn.Module):
    def forward(self, x):
        return x


class Zeros(torch.nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


def test_perfect_reconstruction_has_zero_error():
    image = np.random.default_rng(0).random((28, 28), dtype=np.float32)
    result = reconstruct_image(Identity(), image)
    assert result["mse"] == 0.0
    assert result["error_map"].shape == (28, 28)
    assert np.all(result["error_map"] == 0)


def test_error_matches_mse_definition():
    image = np.random.default_rng(1).random((1, 28, 28), dtype=np.float32)
    result = reconstruct_image(Zeros(), image)
    assert result["mse"] == pytest.approx(float((image**2).mean()), rel=1e-6)
    np.testing.assert_allclose(result["error_map"], image.reshape(28, 28))


def test_rejects_wrong_image_size():
    with pytest.raises(ValueError):
        reconstruct_image(Identity(), np.zeros((10, 10), dtype=np.float32))


def test_decision_rule_uses_strictly_greater_than_threshold():
    assert classify(0.03, 0.02) is True
    assert classify(0.02, 0.02) is False
    assert classify(0.01, 0.02) is False


@pytest.mark.parametrize(
    ("predicted", "actual", "expected"),
    [(True, True, "TP"), (True, False, "FP"), (False, True, "FN"), (False, False, "TN")],
)
def test_outcome(predicted, actual, expected):
    assert outcome(predicted, actual) == expected


@pytest.mark.integration
def test_streamlit_app_runs_end_to_end():
    """Roda o app de verdade (sem navegador). Requer dados e checkpoints."""
    missing = [p for p in ("models/ae_bag.pt", "models/ae_sandal.pt", "data") if not (ROOT / p).exists()]
    if missing:
        pytest.skip(f"faltam {missing}; treine os cenários antes")
    testing = pytest.importorskip("streamlit.testing.v1")
    app = testing.AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=120)
    app.run()
    assert not app.exception, app.exception
    assert len(app.metric) == 4
    assert "Bag (anomalia)" in app.sidebar.selectbox[0].options
    app.sidebar.radio[0].set_value("sandal").run()
    assert not app.exception, app.exception
    assert "Sandal (anomalia)" in app.sidebar.selectbox[0].options


@pytest.mark.integration
@pytest.mark.parametrize("scenario", ["bag", "sandal"])
def test_demo_counts_match_official_evaluation(scenario, monkeypatch):
    """O app tem que tomar as mesmas decisões que a avaliação da #6."""
    if not (ROOT / f"models/ae_{scenario}.pt").exists() or not (ROOT / "data").exists():
        pytest.skip("faltam dados ou checkpoints; treine os cenários antes")
    from autoencoder.demo import load_scenario, test_outcomes
    from autoencoder.evaluate import evaluate_scenario

    monkeypatch.chdir(ROOT)
    data = load_scenario("configs/default.yaml", scenario)
    is_anomaly = data.test_labels == data.anomaly_class
    assert data.val_mse.size > 0
    for row in evaluate_scenario(data.val_mse, data.test_mse, is_anomaly):
        outcomes = test_outcomes(data, row["limiar"])
        assert (outcomes == "FP").sum() == row["fp"]
        assert (outcomes == "FN").sum() == row["fn"]
        assert (outcomes == "TP").sum() == row["tp"]
