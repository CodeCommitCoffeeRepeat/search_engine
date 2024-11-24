# Search Engine

``Search Engine`` is an online semantic search library designed for efficient filtering and classification of scientific articles. Leveraging natural language processing (NLP)
 techniques, it performs text cleaning, embedding generation, and similarity calculation in real-time to identify articles aligned with a specific context and user queries. 
 Instead of relying on complex LLM models, the package employs a heuristic approach based on 
 [Hash Vectorizer](https://scikit-learn.org/1.5/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html) for generating 
 embeddings and [Cosine Similarity](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html) for classifying the 
 data corpus according to the provided context and user queries.


# Installation

The project is in its **Alpha** stage, so installation is only possible directly from the source. 

**System Requirements**: This package has been tested on Ubuntu and is recommended to be installed in a virtual environment. In your terminal, 

1. Clone the repo:

`git clone https://github.com/sarannns/search_engine`

2. Navigate to the project directory

`cd search_engine`

3. Conda virtual environment

`conda create -n <env_name> python=3.12` 

Replace `<env_name>` with a name for your environment (e.g., `search_env`).

4. Activate the environment and Install the dependencies

``conda activate <env_name>``

``pip install -r requirements.txt``



# Usage 



````python

from engine import SearchEngine

# Data
data = pd.read_csv("datafile.csv")

# SearchEngine instance
engine = SearchEngine(vectorizer="hashing", n_features=128)

# Filter and classify the data using the context and query topics
data = engine.search(
    data=data,
    corpus="Abstract", # Corpus column name
    context="Deep Learning", 
    query=["computer vision", "text mining","computer vision text mining"],
)

data["class"].value_counts() 
````

````python 	
class 	            count
Computer Vision 	2640
Text Mining 	    2150
Both 	            657
other 	            575

dtype: int64

````
#### Extract Methods implemented in articles
````python
# Sample list of methods 
methods = {
    "computer vision" : ["multimodal model","multimodal neural network",
                        "vision transformer","diffusion model",....
                           ],

    "text mining": ["transformer models", "self-attention models" ,...],

    "computer vision text mining":["multimodal model",
                                "multimodal neural network","vision transformer",...]}

data = engine.extract_methods(data,corpus="Abstract", class_queries=methods)
````
````python
method
self-attention models               813
vision transformer                  773
multimodal model                    667
other                               626
diffusion model                     506
transformer models                  505
diffusion-based generative model    471
attention-based neural networks     435
multimodal neural network           396
sequence-to-sequence models         360
continuous diffusion model          253
generative diffusion model          217
Name: count, dtype: int64
````

# EDA

Visualize the class clusters by query similarity score

````python
from engine import plot_class_clusters

plot_class_clusters(
    data,
    feature_columns=[
        "score_computer_vision",
        "score_text_mining",
        "score_computer_vision_text_mining",
    ],
    class_column="class",
)

````

Due to private repository, the return figure is hosted here `./imgs/clusters.png`

<br/>

Other available statistical/visualization methods are,
    `plot_context_class_scores`,
    `plot_methods` &
    `plot_class_scores`. Please refer to the [Example Notebook](https://github.com/sarannns/search_engine/blob/main/example.ipynb) for more detailed usage. 

# Contribution Guidelines
To contribute, fork the repository, create a branch for your changes. Ensure your code follows `black` style formatting and includes **tests**. Commit your changes, push your branch, and create a pull request to the main branch. 

Thank you for your contributions!


