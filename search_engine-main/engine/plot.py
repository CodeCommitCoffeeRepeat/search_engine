import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px


def plot_class_scores(
    df: pd.DataFrame, feature_cols: list = None, save_fig: bool = False
):
    """Plot the distribution of similarity scores by class

    Args:

    df (pd.DataFrame): The dataframe containing the similarity scores
    feature_cols (list): The list of feature columns to plot
    save_fig (bool): Whether to save the plot as an image

    Returns:
    None

    """
    plt.figure(figsize=(10, 6))

    # Generate colors for each feature
    colors = sns.color_palette("husl", len(feature_cols))

    for i, feature_col in enumerate(feature_cols):
        # Extract class label from feature column name
        class_label = feature_col.replace("score_", "").replace("_", " ").title()

        sns.histplot(df[feature_col], kde=True, label=class_label, color=colors[i])

    plt.xlabel("Similarity Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Similarity Scores by Class")
    plt.legend()
    if save_fig:
        plt.savefig("class_scores.png")
    plt.show()


def plot_context_class_scores(data, save_fig: bool = False):
    """Plot the distribution of context scores by class

    Args:

    data (pd.DataFrame): The dataframe containing the context scores
    save_fig (bool): Whether to save the plot as an image

    Returns:
    None

    """

    for class_name in data["class"].unique():
        class_df = data[data["class"] == class_name]
        if not class_df.empty:
            sns.histplot(
                class_df["context_score"], kde=True, label=class_name, alpha=0.5
            )

    plt.xlabel("Similarity Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Classes by Context Score")
    plt.legend()
    if save_fig:
        plt.savefig("context_class_scores.png")
    plt.show()


def plot_class_clusters(df, feature_columns, class_column="class"):
    """Plot the clusters of similarity scores by class

    Args:

    df (pd.DataFrame): The dataframe containing the similarity scores
    feature_columns (list): The list of feature columns to plot
    class_column (str): The column containing the class labels

    Returns:
    None
    """

    fig = px.scatter_3d(
        df,
        x=feature_columns[0],
        y=feature_columns[1],
        z=feature_columns[2],
        color=class_column,
        color_discrete_sequence=px.colors.qualitative.Plotly,  # Use a discrete color sequence
        opacity=0.5,
        title="Clusters by Similarity Score",
        labels={
            feature_columns[0]: feature_columns[0]
            .replace("score_", "")
            .replace("_", " ")
            .title(),
            feature_columns[1]: feature_columns[1]
            .replace("score_", "")
            .replace("_", " ")
            .title(),
            feature_columns[2]: feature_columns[2]
            .replace("score_", "")
            .replace("_", " ")
            .title(),
        },
    )
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(
        width=1200,
        height=800,
        legend=dict(
            font=dict(size=16), itemsizing="constant", x=0.8, y=0.8, title_text=""
        ),
        autosize=True,
        # margin=dict(l=0, r=0, b=0, t=0)
    )
    # Figure title in center
    fig.update_layout(title_x=0.5, title_y=0.8)
    fig.show()


def plot_methods(data, save_fig: bool = False):
    """Plot the count of methods by class

    Args:

    data (pd.DataFrame): The dataframe containing the methods and classes
    save_fig (bool): Whether to save the plot as an image

    Returns:
    None
    """

    plt.figure(figsize=(10, 6))
    for class_name in data["class"].unique():
        class_df = data[data["class"] == class_name]
        if not class_df.empty:
            sns.histplot(class_df["method"], kde=False, label=class_name, alpha=0.5)

    plt.xlabel("Method")
    plt.ylabel("Frequency")
    plt.title("Methods count by class")
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    if save_fig:
        plt.savefig("methods.png")
    
    plt.show()
