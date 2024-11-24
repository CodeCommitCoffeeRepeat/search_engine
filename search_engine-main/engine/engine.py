from .data import Cleaner
from .embed import EmbeddingGenerator
from .classifier import Classifier
from .similarity import SimilarityScore
import pandas as pd
import numpy as np
from typing import Union
from pdb import set_trace as bp


class SearchEngine:
    def __init__(self, vectorizer="hashing", n_features: int = 4) -> None:

        self.method = vectorizer
        self.n_features = n_features

        self.cleaner = Cleaner()
        self.embedder = EmbeddingGenerator(
            vectorizer=self.method, n_features=n_features
        )
        self.similarity = SimilarityScore()
        self.classifier = Classifier()

    def _validate_input(func):
        """Decorator to validate the input data

        Args:
            func (function): The function to be decorated

        Returns:
            function: The decorated function"""

        def wrapper(
            self,
            data: Union[pd.DataFrame],
            corpus: str = None,
            context: str = None,
            query: Union[str, list] = None,
            *args,
            **kwargs,
        ):
            assert isinstance(
                data, (pd.DataFrame)
            ), "Input data must be a pandas DataFrame"

            if corpus:
                assert isinstance(corpus, str), "corpus must be a string"
            if context:
                assert isinstance(context, str), "context must be a string"
            if query:
                assert isinstance(
                    query, (str, list)
                ), "query must be a string or a list of strings"
                if isinstance(query, list):
                    for q in query:
                        assert isinstance(
                            q, str
                        ), "All elements in query list must be strings"

            return func(self, data, corpus, context, query, *args, **kwargs)

        return wrapper

    def filter(
        self, corpus: str, context: str, threshold: float = 0.01
    ) -> pd.DataFrame:
        """Filter the data by context

        Args:
            corpus (str): The column name containing the corpus
            context (str): The context to filter the data
            threshold (float): The similarity threshold to filter the data
        Returns:
            pd.DataFrame: Dataframe with the similarity scores for the context
        """

        # Clean the data
        data = self.cleaner.data(x=self.data, id=corpus)

        # Generate data embeddings
        data, embeddings = self.embedder.step(data=data, corpus=corpus)

        # Generate context embeddings
        data, context_embeddings = self.embedder.step(data=data, context=context)

        # Compute cosine similarity between the data embeddings and the context embeddings
        scores = self.similarity.compute(x=embeddings.toarray(), y=context_embeddings)

        # Update the dataframe with the similarity scores and filter rows with similarity > threshold
        self.data = self.similarity.update_df(
            data=data, id="context_score", scores=scores, threshold=threshold
        )

        return None

    def query(self, queries: Union[str, list]) -> pd.DataFrame:
        """Query the data within the context

        Args:
            queries (Union[str, list]): A string or a list of strings containing the queries
        Returns:
            pd.DataFrame: Dataframe with the similarity scores for the queries
        """

        # Compute embeddings for the queries
        _, query_embeddings = self.embedder.step(data=self.data, query=queries)

        # Compute cosine similarity between the data embeddings and the query embeddings
        query_scores = self.similarity.compute(
            x=self.data["embeddings"].tolist(), y=query_embeddings
        )

        # Update the dataframe with the similarity scores for the queries
        query_scores_df = pd.DataFrame(
            np.array(query_scores).T,
            columns=[f"score_{query.replace(' ', '_')}" for query in queries],
            index=self.data.index,
        )
        self.data = pd.concat([self.data, query_scores_df], axis=1)

        return self.data

    def classify(self) -> pd.DataFrame:
        """Embeddings that have a negative similarity score after being filtered by context will be classified as 'other'.

        Returns:
            pd.DataFrame: Dataframe with a 'class' column
        """
        self.data = self.classifier.fit(self.data)
        return self.data

    def get_max_score(self, abstract: list, queries: list) -> tuple:
        """Get the query with the maximum similarity score

        Args:
            abstract (list): A list containing the abstract
            queries (list): A list containing the queries

        Returns:
            tuple: A tuple containing the query with the maximum similarity score"""

        abstract_embedding = self.embedder.generate_embeddings(
            corpus=[abstract]
        )  # TODO: Duplicate, embeddings are already generated;
        query_embeddings = self.embedder.generate_embeddings(corpus=queries)
        scores = self.similarity.compute(abstract_embedding, query_embeddings)
        max_index = scores.argmax()

        if scores[max_index] < 0:
            return "other"
        else:
            return queries[max_index]

    def extract_methods(
        self, data: pd.DataFrame, corpus: str = "Abstract", class_queries: dict = None
    ) -> pd.DataFrame:
        """Extract the method from the abstract based on the class queries

        Args:
            class_queries (dict): A dictionary containing the class name as the
            key and the queries as the value

        Returns:
            pd.DataFrame: Dataframe with a 'method' column
        """
        data = self.cleaner.data(x=data, id=corpus)  # TODO: Duplicate step
        data["method"] = "other"  # Initialize 'method' column to "other"
        for class_name, queries in class_queries.items():
            class_indices = data[data["class"] == class_name].index
            for index in class_indices:
                data.loc[index, "method"] = self.get_max_score(
                    data.loc[index, corpus], queries
                )
        return data

    @_validate_input
    def search(
        self,
        data: Union[str, tuple, pd.DataFrame],
        corpus: str = "Abstract",
        context: str = "Deep Learning",
        query: Union[str, list] = None,
    ) -> pd.DataFrame:
        """Search the data based on the context and the queries

        Args:
            data (Union[str, tuple, pd.DataFrame]): The input data
            corpus (str): The column name containing the corpus
            context (str): The context to filter the data
            query (Union[str, list]): A string or a list of strings containing the queries

        Returns:
            pd.DataFrame: Dataframe with the similarity scores for the context and the queries
        """

        # Clean the data
        self.data = self.cleaner.data(x=data, id=corpus)

        # Filter the corpus by context
        self.filter(corpus, context)

        # Query the data within the context
        self.query(query)

        # Classify the data based on the context and the queries
        self.classify()

        return self.data


# if __name__ == "__main__":

#     df = pd.read_csv("./dataset/collection_with_abstracts.csv")

#     engine = SearchEngine(vectorizer="hashing", n_features=2**7)

#     df = engine.search(
#         data=df,
#         corpus="Abstract",
#         context="Deep Learning",
#         query=["computer vision", "text mining", "text mining computer vision"],
#     )
#     print(df.head())
#     print(df["class"].value_counts())
