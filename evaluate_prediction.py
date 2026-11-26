import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from collections import Counter, defaultdict

# --------------------------------------------------
# Configuration
# --------------------------------------------------

GT_PATH = "./Input/OncoClass_VICC_Input.json"
PRED_PATH = "./Output/OncoClass_Gemma_Finetune_V9.json"

PLOT_DIR = Path("./Output/OncoClass_Matrices")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

CRITERIA = [
    "OVS1", "OS1", "OS2", "OS3",
    "OM1", "OM2", "OM3", "OM4",
    "OP1", "OP2", "OP3", "OP4",
    "SBVS1", "SBS1", "SBS2",
    "SBP1", "SBP2"
]

FIVE_CLASSES = ["B", "LB", "VUS", "LO", "O"]
THREE_CLASSES = ["B", "VUS", "O"]

NUMERIC_TO_FULL = {
    1: "B",
    2: "LB",
    3: "VUS",
    4: "LO",
    5: "O",
}

NUMERIC_TO_SIMPLE = {
    1: "B",
    2: "B",
    3: "VUS",
    4: "O",
    5: "O",
}

FULL_TO_SIMPLE = {
    "B": "B",
    "LB": "B",
    "VUS": "VUS",
    "LO": "O",
    "O": "O",
}


# --------------------------------------------------
# Data normalization
# --------------------------------------------------

def parse_class(value):
    """Return an integer class in 1..5, or None if invalid."""
    if isinstance(value, bool):
        return None

    try:
        # Reject non-integer numeric values such as 3.5.
        number = int(value)
        if str(value).strip() not in {
            str(number), f"{number}.0"
        }:
            return None
        return number if number in NUMERIC_TO_FULL else None
    except (TypeError, ValueError, OverflowError):
        return None


def parse_llm_class(value):
    """Normalize an LLM class label to B, LB, VUS, LO, or O."""
    if not isinstance(value, str):
        return None

    value = value.strip().upper()
    return value if value in FIVE_CLASSES else None


def build_gt_lookup(gt):
    """Map record ID to its valid numerical ground-truth class."""
    lookup = {}

    for record in gt:
        record_id = record.get("ID")
        class_value = parse_class(record.get("Class"))

        if record_id is None or class_value is None:
            continue

        try:
            lookup[int(record_id)] = class_value
        except (TypeError, ValueError, OverflowError):
            continue

    return lookup


def get_record_id(record):
    """Return a normalized ID, or None if invalid."""
    try:
        value = record.get("ID")
        if value is None or isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


# --------------------------------------------------
# Shared classification evaluation
# --------------------------------------------------

def evaluate_classification(
    gt,
    predictions,
    prediction_getter,
    labels,
    normalize_prediction,
    normalize_gt,
    title,
    plot_filename
):
    """
    Evaluate predictions using a shared implementation.

    Accuracy denominator:
        Records with a valid ground-truth class and
        a valid prediction.

    Coverage denominator:
        All prediction records.
    """

    gt_lookup = build_gt_lookup(gt)

    confusion = Counter()
    failures = Counter()
    total = len(predictions)

    matched = 0
    correct = 0
    
    y_true = []
    y_pred = []

    for record in predictions:
        record_id = get_record_id(record)

        if record_id is None or record_id not in gt_lookup:
            failures["no_gt_match"] += 1
            continue

        gt_class = normalize_gt(gt_lookup[record_id])
        pred_class = normalize_prediction(
            prediction_getter(record)
        )

        if gt_class is None:
            failures["invalid_gt_class"] += 1
            continue

        if pred_class is None:
            failures["invalid_pred_class"] += 1
            continue

        matched += 1
        confusion[(gt_class, pred_class)] += 1

        if gt_class == pred_class:
            correct += 1
            
        y_true.append(gt_class)
        y_pred.append(pred_class)

    accuracy = 100 * correct / matched if matched else 0.0
    coverage = 100 * matched / total if total else 0.0
    
    f1_results = calculate_f1(y_true, y_pred, labels) if matched else {
        "macro_f1": 0.0,
        "weighted_f1": 0.0,
    }

    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)
    print(f"Total prediction records : {total}")
    print(f"Valid matched records    : {matched}")
    print(f"Coverage                 : {coverage:.2f}%")
    
    print(f"Correct predictions      : {correct}")
    print(f"Incorrect predictions    : {matched - correct}")
    print(f"Accuracy                 : {accuracy:.2f}%")
    print(f"Macro F1                 : {f1_results['macro_f1']:.4f}")
    print(f"Weighted F1              : {f1_results['weighted_f1']:.4f}")

    # Confusion matrix: rows = ground truth,
    # columns = predicted class.
    print("\nConfusion matrix (rows=GT, columns=prediction)")
    print(f"{'GT / Pred':<12}" + "".join(
        f"{label:>8}" for label in labels
    ))
    
    save_confusion_matrix(
        confusion,
        labels,
        title,
        plot_filename,
    )

    for gt_label in labels:
        row = f"{gt_label:<12}"
        for pred_label in labels:
            row += f"{confusion[(gt_label, pred_label)]:>8}"
        print(row)

    if failures:
        print("\nExcluded records")
        for reason, count in failures.items():
            pct = 100 * count / total if total else 0.0
            print(f"{reason:<22} {count} ({pct:.2f}%)")

    return {
        "total": total,
        "matched": matched,
        "correct": correct,
        "accuracy": accuracy,
        "coverage": coverage,
        "confusion": confusion,
        "failures": failures,
        "macro_f1": f1_results["macro_f1"],
        "weighted_f1": f1_results["weighted_f1"],
    }


# --------------------------------------------------
# 1. LLM-assigned class: five classes
# --------------------------------------------------

def llm_five_class_accuracy(gt, predictions):
    return evaluate_classification(
        gt=gt,
        predictions=predictions,
        prediction_getter=lambda r: (
            r.get("output", {}).get("Classification")
            if isinstance(r.get("output"), dict)
            else None
        ),
        labels=FIVE_CLASSES,
        normalize_prediction=parse_llm_class,
        normalize_gt=lambda c: NUMERIC_TO_FULL[c],
        title="LLM-assigned classification: 5 classes",
        plot_filename="llm_classification_5_classes.png",
    )


# --------------------------------------------------
# 2. LLM-assigned class: three classes
# --------------------------------------------------

def llm_three_class_accuracy(gt, predictions):
    return evaluate_classification(
        gt=gt,
        predictions=predictions,
        prediction_getter=lambda r: (
            r.get("output", {}).get("Classification")
            if isinstance(r.get("output"), dict)
            else None
        ),
        labels=THREE_CLASSES,
        normalize_prediction=lambda value: (
            FULL_TO_SIMPLE.get(parse_llm_class(value))
        ),
        normalize_gt=lambda c: NUMERIC_TO_SIMPLE[c],
        title="LLM-assigned classification: 3 classes",
        plot_filename="llm_classification_3_classes.png",
    )


# --------------------------------------------------
# 3. Calculated class: five classes
# --------------------------------------------------

def calculated_five_class_accuracy(gt, predictions):
    return evaluate_classification(
        gt=gt,
        predictions=predictions,
        prediction_getter=lambda r: r.get("VICC_Class"),
        labels=FIVE_CLASSES,
        normalize_prediction=lambda value: (
            NUMERIC_TO_FULL[c]
            if (c := parse_class(value)) is not None
            else None
        ),
        normalize_gt=lambda c: NUMERIC_TO_FULL[c],
        title="Calculated classification: 5 classes",
        plot_filename="calculated_classification_5_classes.png",
    )


# --------------------------------------------------
# 4. Calculated class: three classes
# --------------------------------------------------

def calculated_three_class_accuracy(gt, predictions):
    return evaluate_classification(
        gt=gt,
        predictions=predictions,
        prediction_getter=lambda r: r.get("VICC_Class"),
        labels=THREE_CLASSES,
        normalize_prediction=lambda value: (
            NUMERIC_TO_SIMPLE[c]
            if (c := parse_class(value)) is not None
            else None
        ),
        normalize_gt=lambda c: NUMERIC_TO_SIMPLE[c],
        title="Calculated classification: 3 classes",
        plot_filename="calculated_classification_3_classes.png",
    )


# --------------------------------------------------
# 5. Criteria analysis
# --------------------------------------------------

def factors(gt, predictions):
    gt_lookup = {
        int(record["ID"]): record
        for record in gt
        if record.get("ID") is not None
    }

    factor_counts = {
        factor: Counter()
        for factor in CRITERIA
    }
    
    factor_true = {factor: [] for factor in CRITERIA}
    factor_pred = {factor: [] for factor in CRITERIA}

    matched = 0
    total = len(predictions)

    for record in predictions:
        record_id = get_record_id(record)

        if record_id is None or record_id not in gt_lookup:
            continue

        gt_record = gt_lookup[record_id]
        pred = record.get("output", {})

        if not isinstance(pred, dict):
            pred = {}

        matched += 1

        for factor in CRITERIA:
            gt_value = gt_record.get(factor, "No")
            pred_value = pred.get(factor, "No")
            
            factor_true[factor].append(gt_value)
            factor_pred[factor].append(pred_value)

            if gt_value == pred_value:
                factor_counts[factor]["correct"] += 1
            else:
                factor_counts[factor]["incorrect"] += 1

    print(f"\n{'=' * 60}")
    print("Criteria analysis")
    print("=" * 60)
    print(f"Total prediction records : {total}")
    print(f"Matched records          : {matched}")
    print(f"\n{'Factor':<12}{'Correct':>10}{'Incorrect':>12}{'Accuracy':>14}{'Macro-F1':>14}{'Weighted-F1':>14}")

    for factor in CRITERIA:
        correct = factor_counts[factor]["correct"]
        incorrect = factor_counts[factor]["incorrect"]
        pct = 100 * correct / matched if matched else 0.0
        
        criterion_y_true = factor_true[factor]
        criterion_y_pred = factor_pred[factor]

        if criterion_y_true:
            scores = calculate_f1(
                criterion_y_true,
                criterion_y_pred,
                labels=["No", "Yes"],
            )
        else:
            scores = {
                "macro_f1": 0.0,
                "weighted_f1": 0.0,
            }

        print(
            f"{factor:<12}{correct:>10}{incorrect:>12}"
            f"{pct:>13.2f}%"
            f"{scores['macro_f1']:>12.4f}"
            f"{scores['weighted_f1']:>14.4f}"
        )

    return factor_counts

# --------------------------------------------------
# Metrics & Graphics
# --------------------------------------------------

def calculate_f1(y_true, y_pred, labels):
    return {
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="weighted",
            zero_division=0,
        ),
    }
    
    
def save_confusion_matrix(confusion, labels, title, filename):
    matrix = np.array([
        [confusion[(gt, pred)] for pred in labels]
        for gt in labels
    ])
    
    row_sums = matrix.sum(axis=1, keepdims=True)
    percentages = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums != 0,
    ) * 100

    fig, ax = plt.subplots(figsize=(8, 7))
    
    image = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    fig.colorbar(image, ax=ax, label="Number of variants")

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted class",
        ylabel="Expert-assigned class",
        title=title,
    )

    threshold = matrix.max() / 2 if matrix.max() else 0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            count = matrix[i, j]
            percentage = percentages[i, j]
            
            ax.text(
                j, i,
                f"{count}\n({percentage:.1f}%)",
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(PLOT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":
    with open(GT_PATH, encoding="utf-8") as f:
        gt_data = json.load(f)

    with open(PRED_PATH, encoding="utf-8") as f:
        cls_data = json.load(f)

    # Classification evaluations
    llm_5 = llm_five_class_accuracy(gt_data, cls_data)
    llm_3 = llm_three_class_accuracy(gt_data, cls_data)

    calculated_5 = calculated_five_class_accuracy(gt_data, cls_data)
    calculated_3 = calculated_three_class_accuracy(
        gt_data, cls_data
    )

    # Criteria evaluation
    factor_results = factors(gt_data, cls_data)

    # Compact final summary
    print(f"\n{'=' * 60}")
    print("FINAL ACCURACY SUMMARY")
    print("=" * 60)
    print(f"{'Evaluation':<42}{'Accuracy':>12}")

    for name, result in [
        ("LLM-assigned class (5 classes)", llm_5),
        ("LLM-assigned class (3 classes)", llm_3),
        ("Calculated class (5 classes)", calculated_5),
        ("Calculated class (3 classes)", calculated_3),
    ]:
        print(f"{name:<42}{result['accuracy']:>11.2f}%")
