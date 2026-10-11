"""Nested patient-level CV; no outer-test data enter fitting or model selection."""
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import copy
import json
import random
import platform
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, f1_score
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from .data import TrainingPreprocessor
from .model import STEP_AF, ModelConfig
from .statistics import repeated_metrics


@dataclass
class TrainConfig:
    repetitions: int = 5
    outer_folds: int = 3
    inner_folds: int = 3
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    batch_size: int = 8
    epochs: int = 120
    patience: int = 15
    seed: int = 20261011  # Reference choice, not claimed to be an original run seed.
    validation_fraction: float = 0.20
    device: str = "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _predict(model, x, permutation, device):
    model.eval()
    with torch.no_grad():
        xt = torch.as_tensor(x, dtype=torch.float32, device=device)
        perm = None if permutation is None else torch.as_tensor(permutation, device=device)
        return model(xt, perm).softmax(dim=-1)[:, 1].cpu().numpy()


def fit_model(x, y, fit_ids, val_ids, model_config, train_config, seed, permutation=None):
    """Training indices and validation indices must be disjoint patient indices."""
    fit_ids, val_ids = np.asarray(fit_ids), np.asarray(val_ids)
    if np.intersect1d(fit_ids, val_ids).size or not len(fit_ids) or not len(val_ids):
        raise ValueError("Training and validation patients must be nonempty and disjoint.")
    set_seed(seed)
    preprocessor = TrainingPreprocessor().fit(x[fit_ids])
    all_x = preprocessor.transform(x)
    model = STEP_AF(model_config).to(train_config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate,
                                  weight_decay=train_config.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    nstage = len(model_config.stage_indices) if model_config.stage_indices is not None \
        else model_config.days // model_config.cycle_days
    perms = np.tile(np.arange(nstage), (len(x), 1)) if permutation is None else permutation
    dataset = TensorDataset(torch.as_tensor(all_x[fit_ids]), torch.as_tensor(y[fit_ids]),
                            torch.as_tensor(perms[fit_ids], dtype=torch.int64))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=train_config.batch_size, shuffle=True,
                        generator=generator, num_workers=0)
    best_loss, bad_epochs, best_state, best_epoch = float("inf"), 0, None, 0
    for epoch in range(train_config.epochs):
        model.train()
        for xb, yb, perm in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb.to(train_config.device),
                           perm.to(train_config.device) if permutation is not None else None)
            loss = loss_fn(logits, yb.to(train_config.device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss.")
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            xv = torch.as_tensor(all_x[val_ids], device=train_config.device)
            yv = torch.as_tensor(y[val_ids], device=train_config.device)
            pv = None if permutation is None else torch.as_tensor(perms[val_ids], device=train_config.device)
            val_loss = float(loss_fn(model(xv, pv), yv))
        if not np.isfinite(val_loss):
            raise FloatingPointError("Non-finite validation loss.")
        if val_loss < best_loss:
            best_loss, bad_epochs, best_epoch = val_loss, 0, epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= train_config.patience:
                break
    if best_state is None:
        raise RuntimeError("No valid checkpoint was produced.")
    model.load_state_dict(best_state)
    return model, preprocessor, {"best_epoch": best_epoch, "validation_loss": best_loss}


def _decision_threshold(labels, probabilities):
    """Maximize F1 on INNER out-of-fold probabilities; deterministic larger-threshold tie-break.

    This criterion is a reference-code choice; the manuscript does not specify it.
    It must never be optimized on outer-test predictions.
    """
    candidates = np.unique(np.r_[0., probabilities, 1.])
    scores = [f1_score(labels, probabilities >= t, zero_division=0) for t in candidates]
    best = np.flatnonzero(np.isclose(scores, max(scores), atol=1e-12, rtol=0))[-1]
    return float(candidates[best])


def run_nested_cv(cohort, output_directory, model_config=None, train_config=None,
                  learning_rates=None, threshold_pp=10., removed_features=(),
                  permutation_seed=None):
    c, t = model_config or ModelConfig(), train_config or TrainConfig()
    if t.epochs < 1 or t.repetitions < 1 or t.outer_folds < 2 or t.inner_folds < 2:
        raise ValueError("Positive epochs/repetitions and >=2 folds are required.")
    out = Path(output_directory)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "oof_predictions.npz").exists():
        raise FileExistsError("Choose a new output directory to preserve previous runs.")
    kept_features = tuple(j for j in range(cohort.x.shape[-1]) if j not in removed_features)
    if not kept_features or any(j < 0 or j >= cohort.x.shape[-1] for j in removed_features):
        raise ValueError("Invalid feature-removal configuration.")
    x = cohort.x[:, :, kept_features]
    c = replace(c, input_features=len(kept_features))
    c.validate()
    y = cohort.labels(threshold_pp)
    nstage = len(c.stage_indices) if c.stage_indices is not None else c.days // c.cycle_days
    permutations = None
    if permutation_seed is not None:
        rng = np.random.default_rng(permutation_seed)
        permutations = np.stack([rng.permutation(nstage) for _ in cohort.patient_ids])
    probabilities = np.full((len(y), t.repetitions), np.nan)
    predictions = np.full_like(probabilities, -1, dtype=int)
    fold_ids = np.full_like(predictions, -1)
    records = []
    learning_rates = learning_rates or [t.learning_rate]
    if not all(lr > 0 for lr in learning_rates):
        raise ValueError("Learning rates must be positive.")
    for repetition in range(t.repetitions):
        outer = StratifiedKFold(t.outer_folds, shuffle=True, random_state=t.seed + repetition)
        for fold, (outer_train, outer_test) in enumerate(outer.split(x, y)):
            inner_splits = list(StratifiedKFold(
                t.inner_folds, shuffle=True, random_state=t.seed + 100 + repetition * 10 + fold
            ).split(x[outer_train], y[outer_train]))
            candidates = []
            for lr in learning_rates:
                candidate_t = replace(t, learning_rate=lr)
                inner_p = np.full(len(outer_train), np.nan)
                scores = []
                for inner_fold, (train_relative, val_relative) in enumerate(inner_splits):
                    seed = t.seed + 1000 * repetition + 100 * fold + inner_fold
                    fit_ids, val_ids = outer_train[train_relative], outer_train[val_relative]
                    model, prep, _ = fit_model(x, y, fit_ids, val_ids, c, candidate_t, seed, permutations)
                    pv = _predict(model, prep.transform(x[val_ids]),
                                  None if permutations is None else permutations[val_ids], t.device)
                    inner_p[val_relative] = pv
                    scores.append(roc_auc_score(y[val_ids], pv))
                candidates.append((float(np.mean(scores)), lr, inner_p))
            selected = max(candidates, key=lambda item: item[0])
            decision_threshold = _decision_threshold(y[outer_train], selected[2])
            # Fresh holdout drawn strictly inside outer training. No test patient is consulted.
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=t.validation_fraction,
                                              random_state=t.seed + 9000 + repetition * 10 + fold)
            fit_relative, val_relative = next(splitter.split(x[outer_train], y[outer_train]))
            fit_ids, val_ids = outer_train[fit_relative], outer_train[val_relative]
            selected_t = replace(t, learning_rate=selected[1])
            model, prep, fitting = fit_model(x, y, fit_ids, val_ids, c, selected_t,
                                             t.seed + 10000 + repetition * 10 + fold, permutations)
            test_p = _predict(model, prep.transform(x[outer_test]),
                              None if permutations is None else permutations[outer_test], t.device)
            probabilities[outer_test, repetition] = test_p
            predictions[outer_test, repetition] = test_p >= decision_threshold
            fold_ids[outer_test, repetition] = fold
            record = {"repetition": repetition, "fold": fold, "learning_rate": selected[1],
                      "decision_threshold": decision_threshold, "inner_auroc": selected[0],
                      "train_indices": fit_ids.tolist(), "validation_indices": val_ids.tolist(),
                      "test_indices": outer_test.tolist(), **fitting}
            records.append(record)
            torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                        "model_config": asdict(c), "preprocessing": prep.state(),
                        "kept_features": list(kept_features), "decision_threshold": decision_threshold,
                        "permutation_seed": permutation_seed, "training": record},
                       out / f"repeat_{repetition + 1}_fold_{fold + 1}.pt")
            print(f"Completed repetition {repetition + 1}/{t.repetitions}, fold {fold + 1}/{t.outer_folds}", flush=True)
    summary = repeated_metrics(y, probabilities, predictions)
    # Identifiers remain in private local output; gitignore excludes these files.
    np.savez_compressed(out / "oof_predictions.npz", patient_ids=np.asarray(cohort.patient_ids),
                        labels=y, probabilities=probabilities, predictions=predictions, fold_ids=fold_ids)
    metadata = {"provenance": "New manuscript-derived reference implementation; not original run code",
                "model_config": asdict(c), "train_config": asdict(t), "threshold_pp": threshold_pp,
                "learning_rate_candidates": list(learning_rates), "kept_features": list(kept_features),
                "permutation_seed": permutation_seed,
                "versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "torch": torch.__version__}, "metrics": summary, "folds": records}
    (out / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return summary
