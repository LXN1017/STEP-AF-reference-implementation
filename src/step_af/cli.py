"""Command-line entry points for private extraction, nested CV and evaluation."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .data import load_cohort, extract_first_valid, TrainingPreprocessor
from .model import STEP_AF, ModelConfig
from .training import run_nested_cv, TrainConfig
from .statistics import repeated_metrics, paired_cluster_bootstrap


def main(argv=None):
    parser = argparse.ArgumentParser(description="STEP-AF manuscript-derived reference implementation")
    sub = parser.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extract", help="Select the earliest QC-valid record each day")
    extract.add_argument("--manifest", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--audit", required=True)
    train = sub.add_parser("train", help="Nested patient-level cross-validation")
    train.add_argument("--features", required=True)
    train.add_argument("--labels", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--config")
    train.add_argument("--threshold-pp", type=float, default=10.)
    train.add_argument("--remove-features", type=int, nargs="*", default=[])
    train.add_argument("--permutation-seed", type=int)
    evaluate = sub.add_parser("evaluate", help="Summarize saved OOF probabilities")
    evaluate.add_argument("predictions")
    compare = sub.add_parser("compare", help="One prespecified paired AUROC comparison family")
    compare.add_argument("--reference", required=True)
    compare.add_argument("--alternatives", nargs="+", required=True)
    compare.add_argument("--samples", type=int, default=10000)
    infer = sub.add_parser("predict", help="Inference using a saved outer-fold checkpoint")
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--features", required=True, help="Private .npy array: patients x 96 x 10")
    infer.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "extract":
        result = extract_first_valid(args.manifest, args.output, args.audit)
    elif args.command == "train":
        config = json.loads(Path(args.config).read_text()) if args.config else {}
        result = run_nested_cv(load_cohort(args.features, args.labels), args.output,
                               ModelConfig(**config.get("model", {})),
                               TrainConfig(**config.get("training", {})),
                               config.get("learning_rates"), args.threshold_pp,
                               tuple(args.remove_features), args.permutation_seed)
    elif args.command == "evaluate":
        with np.load(args.predictions, allow_pickle=False) as p:
            result = repeated_metrics(p["labels"], p["probabilities"], p["predictions"])
    elif args.command == "compare":
        with np.load(args.reference, allow_pickle=False) as reference:
            y, p, ids = reference["labels"], reference["probabilities"], reference["patient_ids"]
            alternatives = []
            for filename in args.alternatives:
                with np.load(filename, allow_pickle=False) as other:
                    if not np.array_equal(other["patient_ids"], ids) or not np.array_equal(other["labels"], y):
                        raise ValueError("Paired comparisons require identical patient ordering and labels.")
                    if not np.array_equal(other["fold_ids"], reference["fold_ids"]):
                        raise ValueError("Paired runs must share their outer folds.")
                    alternatives.append(other["probabilities"])
            result = paired_cluster_bootstrap(y, p, np.asarray(alternatives), samples=args.samples)
    else:
        # Only load trusted, locally generated checkpoints. PyTorch weights_only avoids arbitrary pickle code.
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if saved.get("permutation_seed") is not None:
            raise ValueError("Permutation controls are not intended as clinical inference checkpoints.")
        model = STEP_AF(ModelConfig(**saved["model_config"]))
        model.load_state_dict(saved["state_dict"])
        x = np.load(args.features, allow_pickle=False)[:, :, saved["kept_features"]]
        prep = TrainingPreprocessor.from_state(saved["preprocessing"])
        model.eval()
        with torch.no_grad():
            probability = model(torch.as_tensor(prep.transform(x))).softmax(-1)[:, 1].numpy()
        np.savez_compressed(args.output, probability=probability,
                            prediction=probability >= saved["decision_threshold"])
        result = {"predicted_patients": len(probability), "decision_threshold": saved["decision_threshold"]}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
