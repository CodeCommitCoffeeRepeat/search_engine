import pandas as pd
import re
import nltk
import logging
logging.getLogger("nltk").setLevel(logging.WARNING)

nltk.download("stopwords")
from nltk.corpus import stopwords


class Cleaner:
    def __init__(self):
        pass

    def remove_punctuation(self, x):

        if isinstance(x, str):
            x = re.sub(r"[^\w\s]", "", x)
            return x
        else:
            return x

    def remove_stopwords(self, x):

        stop_words = set(stopwords.words("english"))
        words = x.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return " ".join(filtered_words)

    def remove_whitespace(self, x):

        x = x.strip()
        x = re.sub(r"\s+", " ", x)
        return x

    def remove_NaN(self, x):
        x = x.dropna(subset=["Abstract"])
        return x

    def remove_numbers(self, x):
        x = re.sub(r"\d+", "", x)
        return x

    def data(self, x=None, id=None):
        assert isinstance(x, pd.DataFrame)
        assert isinstance(id, str)
        x.dropna(subset=[id], inplace=True)
        x[id] = (
            x[id]
            .apply(self.remove_punctuation)
            .apply(self.remove_stopwords)
            .apply(self.remove_whitespace)
            .apply(self.remove_numbers)
        )
        return x


class FeatureExtractor:
    def __init__(self):
        pass

    def generate_corpus(self, x):
        corpus = x["Abstract"].tolist()
        return corpus

    def ngrams(self, x, n):
        tokens = x.split()
        return list(nltk.ngrams(tokens, n))


# if __name__ == "__main__":

#     x = pd.read_csv("./dataset/collection_with_abstracts.csv")
#     cleaner = Cleaner()
#     x = cleaner.data(x=x, id="Abstract")
#     print(x)
#     feature_extractor = FeatureExtractor()
#     bigrams = x["Abstract"].apply(feature_extractor.ngrams, n=2)
#     print(bigrams)
