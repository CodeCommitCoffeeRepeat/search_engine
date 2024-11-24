import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityScore:

    def __init__(self, method: str = "cosine") -> None:

        if method != "cosine":
            return NotImplementedError
        self.method = method

    def compute(self, x: tuple = None, y: tuple = None):
        """
        Compute cosine similarity between the data embeddings 'x' and the query embeddings 'y'
        """
        cosine_similarities = cosine_similarity(y, x)
        return cosine_similarities

    def update_df(
        self,
        data: pd.DataFrame,
        id: str = None,
        scores: tuple = None,
        threshold: float = None,
    ) -> None:

        # Create a column with cosine similarity score
        data[id] = scores[0]

        if threshold == None:
            return data
        else:
            # Get indices of rows with similarity > threshold
            filtered_indices = data[data[id] > threshold].index

            # Create a new dataframe with the filtered rows
            data = data.loc[filtered_indices]

            # Argsort from highest similarity to the lowest
            data = data.sort_values(by=id, ascending=False)

            return data
