import pandas as pd

class Classifier:
    def __init__(self) -> None:
        pass

    def fit(self, data: pd.DataFrame) -> pd.DataFrame:

        # Extract score column ids
        score_ids = [col for col in data.columns if col.startswith("score_")]

        # Get column with max value for each row
        data["class"] = data[score_ids].idxmax(axis=1)

        # Replace 'score_' prefix and underscores
        data["class"] = data["class"].str.replace("score_", "").str.replace("_", " ")

        # Handle cases where max similarity is < 0
        data["class"] = data["class"].mask(data[score_ids].max(axis=1) < 0.0, "other")

        # Replace 'text mining' and 'computer vision' with 'both'
        data["class"] = data["class"].str.replace("text mining computer vision", "both")

        return data

