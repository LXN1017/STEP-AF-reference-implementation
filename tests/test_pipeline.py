"""Artificial test fixtures only; no clinical data or clinical performance claims."""
import csv
from dataclasses import replace
import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from step_af.features import FEATURES, features_from_ppi, quality_check
from step_af.data import Cohort, TrainingPreprocessor, load_cohort, extract_first_valid
from step_af.model import STEP_AF, ModelConfig
from step_af.statistics import repeated_metrics, holm_adjust, paired_cluster_bootstrap
from step_af.training import TrainConfig, fit_model, run_nested_cv


def test_segment_means_preserve_chronology():
    model = STEP_AF()
    x = torch.arange(96.).reshape(1, 96, 1).expand(-1, -1, 10)
    assert model.segment_means(x, 2).shape == (1, 12, 4, 10)
    assert model.segment_means(x, 4).shape == (1, 12, 2, 10)
    assert model.segment_means(x, 8).shape == (1, 12, 1, 10)
    torch.testing.assert_close(model.segment_means(x, 2)[0, 0, :, 0], torch.tensor([.5, 2.5, 4.5, 6.5]))
    torch.testing.assert_close(model.segment_means(x, 8)[0, -1, 0, 0], torch.tensor(91.5))


@pytest.mark.parametrize("scales,cycle", [((2,4,8),8), ((1,2,4,8),8), ((2,4,8,16),16), ((2,4),8)])
def test_model_conditions_and_gradient(scales, cycle):
    torch.manual_seed(7)
    model = STEP_AF(ModelConfig(scales=scales, cycle_days=cycle))
    x = torch.randn(3, 96, 10)
    logits = model(x)
    assert logits.shape == (3, 2)
    fused, debug = model.local_representation(x)
    assert fused.shape == (3, 96 // cycle, 32)
    assert debug["slope"] > 0
    torch.testing.assert_close(debug["weights"].sum(-1), torch.ones(3, 96 // cycle))
    assert torch.all((debug["gate"] > 0) & (debug["gate"] < 1))
    nn_loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0]))
    nn_loss.backward()
    assert model.raw_a.grad is not None and torch.isfinite(model.raw_a.grad)
    assert model.encoders[0].weight.grad is not None


def test_permutation_moves_only_stages_and_rejects_invalid_order():
    model = STEP_AF(ModelConfig(dropout=0.))
    model.eval()
    x = torch.randn(2, 96, 10)
    identity = torch.arange(12).expand(2, -1)
    torch.testing.assert_close(model(x), model(x, identity))
    reversed_order = torch.arange(11, -1, -1).expand(2, -1)
    assert not torch.allclose(model(x), model(x, reversed_order))
    with pytest.raises(ValueError, match="exactly once"):
        model(x, torch.zeros(2, 12, dtype=torch.int64))
    shortened = STEP_AF(ModelConfig(stage_indices=(11,)))
    assert shortened(x).shape == (2, 2)


def test_preprocessor_uses_only_training_values_and_roundtrip():
    train = np.array([[[1., np.nan], [3., 10.]], [[5., 20.], [7., 30.]]])
    prep = TrainingPreprocessor().fit(train)
    np.testing.assert_allclose(prep.median, [4, 20])
    state_before = prep.state()
    unseen = np.array([[[100000., np.nan]]])
    transformed = prep.transform(unseen)
    assert transformed[0, 0, 1] == 0
    assert prep.state() == state_before
    np.testing.assert_allclose(TrainingPreprocessor.from_state(state_before).transform(unseen), transformed)
    with pytest.raises(ValueError, match="undefined"):
        TrainingPreprocessor().fit(np.full((2, 96, 10), np.nan))


def test_invalid_ppi_gaps_are_not_bridged():
    # [800,810] and [1000,1020] are separate original valid runs.
    feature = features_from_ppi([800,810,1000,1020], [0,1,4,5], [1,2,5,6])
    assert feature[0] == pytest.approx(np.sqrt((10**2 + 20**2)/2))
    assert feature[2] == 0
    assert np.isnan(feature[8])  # No uninterrupted length-three template.
    assert np.isnan(feature[4])  # Short spectral observation span.


def test_spectral_bands_and_entropy_constant_sequence():
    t = np.arange(50, dtype=float)
    pp = 800 + 100*np.sin(2*np.pi*0.1*t)
    features = features_from_ppi(pp, np.arange(50), t)
    assert features[4] > features[5] * 10  # LF oscillation, artificial fixture.
    constant = features_from_ppi(np.full(50, 800.), np.arange(50), t)
    assert constant[0] == 0 and constant[7] == 0 and constant[8] == 0
    assert np.isnan(constant[6]) and np.isnan(constant[9])


def test_quality_and_first_valid_selection(tmp_path):
    # Artificial pulse-like periodic waveform, not a participant recording.
    time = np.arange(12000) / 200
    good = np.sin(2*np.pi*1.2*time) + .25*np.sin(2*np.pi*2.4*time)
    qc = quality_check(good)
    assert qc.valid and len(qc.values) >= 15
    assert not quality_check(np.zeros(12000)).valid
    for name, array in (("invalid", np.zeros(12000)), ("first", good), ("later", good)):
        np.save(tmp_path / f"{name}.npy", array)
    with (tmp_path / "manifest.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id","day","started_at","recording_path"])
        writer.writerow(["test-only",1,"2026-01-01T08:00:00+08:00","invalid.npy"])
        writer.writerow(["test-only",1,"2026-01-01T12:00:00+08:00","first.npy"])
        writer.writerow(["test-only",1,"2026-01-01T20:00:00+08:00","later.npy"])
    result = extract_first_valid(tmp_path/"manifest.csv", tmp_path/"features.csv", tmp_path/"audit.csv")
    assert result["selected_patient_days"] == 1
    with (tmp_path/"audit.csv").open() as f:
        audited = list(csv.DictReader(f))
    assert [row["selected"] for row in audited] == ["False","True","False"]


def test_patient_loader_strict_labels_and_missing_days(tmp_path):
    with (tmp_path/"labels.csv").open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["patient_id","baseline_burden_percent","followup_burden_percent"])
        writer.writerows([["a",0,10],["b",0,10.01]])
    with (tmp_path/"features.csv").open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["patient_id","day",*FEATURES])
        for pid in ("a","b"):
            for day in range(1,97):
                writer.writerow([pid,day,*([1]*10)])
    cohort=load_cohort(tmp_path/"features.csv",tmp_path/"labels.csv")
    np.testing.assert_array_equal(cohort.labels(),[0,1])
    lines=(tmp_path/"features.csv").read_text().splitlines()
    (tmp_path/"features.csv").write_text("\n".join(lines[:-1])+"\n")
    with pytest.raises(ValueError,match="day-level imputation"):
        load_cohort(tmp_path/"features.csv",tmp_path/"labels.csv")


def test_repetition_metrics_do_not_flatten_auc():
    y=np.array([0,0,1,1])
    p=np.array([[.1,.8],[.2,.9],[.3,.1],[.4,.2]])
    result=repeated_metrics(y,p,p>=.5)
    assert result["AUROC"] == pytest.approx(np.mean([roc_auc_score(y,p[:,r]) for r in range(2)]))
    assert result["AUROC"] != roc_auc_score(np.repeat(y,2),p.ravel())
    assert result["AUPRC"] == pytest.approx(np.mean([average_precision_score(y,p[:,r]) for r in range(2)]))
    assert sum(result["confusion"].values()) == 8
    np.testing.assert_allclose(holm_adjust([.01,.03,.04]),[.03,.06,.06])
    same=paired_cluster_bootstrap(y,p,np.array([p]),samples=10,seed=7)
    assert same["delta_auroc"] == [0.] and same["p_unadjusted"] == [1.]


def test_fit_and_nested_cv_end_to_end(tmp_path):
    # Minimal computational smoke test, not a clinical experiment.
    torch.set_num_threads(1)
    rng=np.random.default_rng(8)
    x=rng.normal(size=(30,96,10))
    x[0,0,0]=np.nan
    y=np.tile([0,1],15)
    t=TrainConfig(repetitions=1,outer_folds=3,inner_folds=3,epochs=1,patience=1,batch_size=16)
    config=ModelConfig(dropout=0.)
    model,prep,info=fit_model(x,y,np.arange(20),np.arange(20,30),config,t,3)
    np.testing.assert_allclose(prep.median,np.nanmedian(x[:20].reshape(-1,10),axis=0))
    assert info["best_epoch"] == 1
    cohort=Cohort(tuple(f"artificial-{i}" for i in range(30)),x,np.zeros(30),y*20)
    summary=run_nested_cv(cohort,tmp_path/"run",config,t)
    assert summary["patients"] == 30 and summary["repetitions"] == 1
    with np.load(tmp_path/"run/oof_predictions.npz") as saved:
        assert np.isfinite(saved["probabilities"]).all()
        assert set(saved["fold_ids"].ravel()) == {0,1,2}
    checkpoint=torch.load(tmp_path/"run/repeat_1_fold_1.pt",weights_only=True)
    restored=STEP_AF(ModelConfig(**checkpoint["model_config"]))
    restored.load_state_dict(checkpoint["state_dict"])
    assert not set(checkpoint["training"]["train_indices"]) & set(checkpoint["training"]["test_indices"])
    assert not set(checkpoint["training"]["validation_indices"]) & set(checkpoint["training"]["test_indices"])
