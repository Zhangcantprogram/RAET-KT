import os
import tempfile
import unittest

import pandas as pd

from pykt.preprocess.assist2017_preprocess import (
    build_question_to_skills,
    read_data_from_csv,
)
from pykt.preprocess.split_datasets import (
    extend_multi_concepts,
    get_max_concepts,
    read_data,
)


class Assist2017PreprocessTest(unittest.TestCase):
    def test_question_skill_mapping_is_stable_and_escapes_delimiters(self):
        df = pd.DataFrame({
            "problemId": [10, 10, 10, 11, 12],
            "skill": [
                "skill_one",
                "skill-two",
                "noskill",
                "skill,three",
                "NoSkill",
            ],
        })

        mapping = build_question_to_skills(df)

        self.assertEqual(mapping[10], ["skill####one", "skill-two"])
        self.assertEqual(mapping[11], ["skill@@@@three"])
        self.assertNotIn(12, mapping)

    def test_reader_emits_multi_concept_protocol_for_shared_pipeline(self):
        raw = pd.DataFrame({
            "studentId": [1, 1, 1, 1, 2, 2, 2, 2],
            "problemId": [10, 11, 12, 99, 10, 13, 14, 99],
            "skill": [
                "kc-a", "kc-c", "kc-d", "noskill",
                "kc-b", "kc-e", "kc-f", "NoSkill",
            ],
            "correct": [1, 0, 1, 0, 0, 1, 0, 1],
            "timeTaken": [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0],
            "startTime": [100, 200, 300, 400, 100, 200, 300, 400],
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = os.path.join(tmpdir, "assist2017.csv")
            data_path = os.path.join(tmpdir, "data.txt")
            raw.to_csv(raw_path, index=False)

            read_data_from_csv(raw_path, data_path)
            sequences, effective_keys = read_data(data_path)

        self.assertEqual(get_max_concepts(sequences), 2)
        self.assertTrue(
            all("kc-a_kc-b" in concepts for concepts in sequences["concepts"])
        )
        self.assertTrue(all("noskill" not in value.lower() for value in sequences["concepts"]))

        expanded, effective_keys = extend_multi_concepts(
            sequences, effective_keys
        )
        self.assertIn("is_repeat", effective_keys)
        self.assertTrue(
            any("1" in repeats.split(",") for repeats in expanded["is_repeat"])
        )


if __name__ == "__main__":
    unittest.main()
