import unittest
import pandas as pd

from engine import SearchEngine


class TestSearchEngine(unittest.TestCase):

    def test_search_engine(self):
        df = pd.read_csv("../dataset/collection_with_abstracts.csv")

        engine = SearchEngine(vectorizer="hashing", n_features=2**7)

        df = engine.search(
            data=df,
            corpus="Abstract",
            context="Deep Learning",
            query=["computer vision", "text mining", "text mining computer vision"],
        )

        # Assert target columns exists
        self.assertIn("embeddings", df.columns)

        self.assertIn("score_computer_vision", df.columns)
        self.assertIn("class", df.columns)
        # Check if the class column contains the expected values
        self.assertTrue(
            df["class"].isin(["computer vision", "text mining", "both"]).any()
        )


if __name__ == "__main__":
    unittest.main()
