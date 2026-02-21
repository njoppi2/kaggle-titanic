import csv
import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
CV_REPORT_PATH = ROOT / "artifacts" / "cv_report.json"
CLI_SUBMISSION_PATH = ROOT / "solutions" / "cli_baseline_submission.csv"


class ReproducibleCliArtifactsTests(unittest.TestCase):
    def test_cv_report_exists_and_has_expected_fields(self) -> None:
        self.assertTrue(CV_REPORT_PATH.is_file(), msg="Missing artifacts/cv_report.json")

        with CV_REPORT_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        self.assertEqual(payload.get("model"), "sex_pclass_rate_baseline")
        self.assertGreaterEqual(payload.get("folds", 0), 2)
        self.assertIsInstance(payload.get("fold_metrics"), list)
        self.assertGreater(len(payload["fold_metrics"]), 0)
        self.assertIn("mean_accuracy", payload)

    def test_cli_submission_schema(self) -> None:
        self.assertTrue(CLI_SUBMISSION_PATH.is_file(), msg="Missing solutions/cli_baseline_submission.csv")

        with CLI_SUBMISSION_PATH.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(reader.fieldnames, ["PassengerId", "Survived"])

            row_count = 0
            for row in reader:
                row_count += 1
                self.assertTrue(row["PassengerId"].isdigit())
                self.assertIn(row["Survived"], {"0", "1"})

        self.assertGreater(row_count, 0)


if __name__ == "__main__":
    unittest.main()
