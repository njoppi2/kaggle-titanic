import csv
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOLUTIONS_DIR = ROOT / "solutions"


class SolutionFilesTests(unittest.TestCase):
    def test_expected_solution_files_exist(self) -> None:
        expected_files = {
            "gender_submission.csv",
            "kaggle_titanic.csv",
            "xgboost_kaggle_titanic.csv",
        }
        existing_files = {path.name for path in SOLUTIONS_DIR.glob("*.csv")}
        self.assertTrue(expected_files.issubset(existing_files))

    def test_solution_schema_and_values(self) -> None:
        for csv_path in SOLUTIONS_DIR.glob("*.csv"):
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(
                    reader.fieldnames,
                    ["PassengerId", "Survived"],
                    msg=f"Invalid header in {csv_path.name}",
                )
                row_count = 0
                for row in reader:
                    row_count += 1
                    self.assertTrue(row["PassengerId"].isdigit(), msg=f"Invalid PassengerId in {csv_path.name}")
                    self.assertIn(row["Survived"], {"0", "1"}, msg=f"Invalid Survived value in {csv_path.name}")
                self.assertGreater(row_count, 0, msg=f"No rows found in {csv_path.name}")


if __name__ == "__main__":
    unittest.main()
