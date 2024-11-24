from sklearn.feature_extraction.text import HashingVectorizer
import pandas as pd
from typing import Union


class EmbeddingGenerator:

    def __init__(self, vectorizer: str = "hashing", n_features: int = 4) -> None:
        self.method = vectorizer
        self.n_features = n_features
        if self.method == "hashing":
            self.vectorizer = HashingVectorizer(n_features=self.n_features)
        else:
            raise NotImplementedError

    def generate_embeddings(self, corpus: list) -> tuple:

        assert all(isinstance(sentence, str) for sentence in corpus)

        embeddings = self.vectorizer.fit_transform(corpus)
        return embeddings

    def get_corpus(self, data: pd.DataFrame, corpus_col: str) -> tuple:

        corpus = data[corpus_col].values.tolist()

        return corpus

    def step(
        self,
        data: Union[str, tuple, pd.DataFrame],
        corpus: str = None,
        context: str = None,
        query: Union[str, list] = None,
    ) -> tuple:
        try:
            if context:
                embeddings = self.generate_embeddings(corpus=[context])

            elif query:
                embeddings = self.generate_embeddings(corpus=query)

            elif corpus:
                corpus = self.get_corpus(data, corpus)
                embeddings = self.generate_embeddings(corpus)
                data["embeddings"] = list(embeddings.toarray())
            else:
                raise ValueError("Must specify either 'corpus', 'context', or 'query'")
        except Exception as e:
            print(f"Error during Embedding generation: {e}")

        return data, embeddings
