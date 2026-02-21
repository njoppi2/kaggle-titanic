#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def survival_lookup(train_rows: list[dict[str, str]]) -> tuple[dict[tuple[str, str], float], dict[str, float], float]:
    by_sex_pclass: dict[tuple[str, str], list[int]] = {}
    by_sex: dict[str, list[int]] = {}
    overall: list[int] = []

    for row in train_rows:
        survived = int(row["Survived"])
        sex = row["Sex"].strip().lower()
        pclass = row["Pclass"].strip()

        by_sex_pclass.setdefault((sex, pclass), []).append(survived)
        by_sex.setdefault(sex, []).append(survived)
        overall.append(survived)

    sex_pclass_rate = {key: mean(values) for key, values in by_sex_pclass.items()}
    sex_rate = {key: mean(values) for key, values in by_sex.items()}
    overall_rate = mean(overall)

    return sex_pclass_rate, sex_rate, overall_rate


def predict_survival(
    row: dict[str, str],
    sex_pclass_rate: dict[tuple[str, str], float],
    sex_rate: dict[str, float],
    overall_rate: float,
) -> int:
    sex = row["Sex"].strip().lower()
    pclass = row["Pclass"].strip()

    probability = sex_pclass_rate.get((sex, pclass))
    if probability is None:
        probability = sex_rate.get(sex, overall_rate)

    return 1 if probability >= 0.5 else 0


def fold_for_row(row: dict[str, str], row_index: int, num_folds: int) -> int:
    passenger_id = int(row["PassengerId"])
    return (passenger_id + row_index) % num_folds


def cross_validate(train_rows: list[dict[str, str]], num_folds: int) -> list[dict[str, float | int]]:
    fold_reports: list[dict[str, float | int]] = []

    for fold in range(num_folds):
        fold_train = [
            row for index, row in enumerate(train_rows)
            if fold_for_row(row, index, num_folds) != fold
        ]
        fold_valid = [
            row for index, row in enumerate(train_rows)
            if fold_for_row(row, index, num_folds) == fold
        ]

        sex_pclass_rate, sex_rate, overall_rate = survival_lookup(fold_train)

        correct = 0
        for row in fold_valid:
            prediction = predict_survival(row, sex_pclass_rate, sex_rate, overall_rate)
            if prediction == int(row["Survived"]):
                correct += 1

        accuracy = correct / len(fold_valid) if fold_valid else 0.0
        fold_reports.append(
            {
                "fold": fold,
                "train_rows": len(fold_train),
                "validation_rows": len(fold_valid),
                "accuracy": round(accuracy, 6),
            }
        )

    return fold_reports


def write_submission(
    test_rows: list[dict[str, str]],
    sex_pclass_rate: dict[tuple[str, str], float],
    sex_rate: dict[str, float],
    overall_rate: float,
    submission_path: Path,
) -> None:
    submission_path.parent.mkdir(parents=True, exist_ok=True)

    with submission_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["PassengerId", "Survived"])
        writer.writeheader()

        for row in test_rows:
            writer.writerow(
                {
                    "PassengerId": row["PassengerId"],
                    "Survived": predict_survival(row, sex_pclass_rate, sex_rate, overall_rate),
                }
            )


def write_cv_report(report_path: Path, fold_reports: list[dict[str, float | int]], submission_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    mean_accuracy = mean([float(item["accuracy"]) for item in fold_reports]) if fold_reports else 0.0

    report = {
        "model": "sex_pclass_rate_baseline",
        "folds": len(fold_reports),
        "mean_accuracy": round(mean_accuracy, 6),
        "fold_metrics": fold_reports,
        "generated_files": {
            "submission": str(submission_path).replace("\\", "/")
        },
    }

    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate reproducible Titanic baseline artifacts.")
    parser.add_argument("--train", default="data/train.csv")
    parser.add_argument("--test", default="data/test.csv")
    parser.add_argument("--submission", default="solutions/cli_baseline_submission.csv")
    parser.add_argument("--cv-report", default="artifacts/cv_report.json")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    train_path = Path(args.train)
    test_path = Path(args.test)
    submission_path = Path(args.submission)
    cv_report_path = Path(args.cv_report)

    train_rows = read_rows(train_path)
    test_rows = read_rows(test_path)

    fold_reports = cross_validate(train_rows, args.folds)
    sex_pclass_rate, sex_rate, overall_rate = survival_lookup(train_rows)

    write_submission(test_rows, sex_pclass_rate, sex_rate, overall_rate, submission_path)
    write_cv_report(cv_report_path, fold_reports, submission_path)

    print(f"Wrote submission: {submission_path}")
    print(f"Wrote CV report: {cv_report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
