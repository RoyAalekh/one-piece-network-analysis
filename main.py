import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import polars as pl

pio.renderers.default = "browser"


def read_data(file_path: str) -> pl.DataFrame:
    return pl.read_csv(file_path)


def coerce_types(df: pl.DataFrame):
    return df.with_columns(
            pl.col("chapter").cast(pl.Int64, strict=True),
            pl.col("volume").cast(pl.Int64, strict=True),
            pl.col("pages").cast(pl.Int64, strict=True),
    )


def average_number_of_pages_per_chapter(chapters_df: pl.DataFrame):
    avg_pages = chapters_df.group_by("chapter").agg(pl.col("pages").mean())
    chapters_df = chapters_df.join(avg_pages, on="chapter", how="left")
    return chapters_df


def average_number_of_pages_per_volume(chapters_df: pl.DataFrame):
    avg_pages = chapters_df.group_by("volume").agg(pl.col("pages").mean())
    chapters_df = chapters_df.join(avg_pages, on="volume", how="left")
    return chapters_df


def average_number_of_characters_per_chapter(characters_df: pl.DataFrame):
    avg_characters = characters_df.group_by("chapter").agg(pl.col("character").count())
    characters_df = characters_df.join(avg_characters, on="chapter", how="left")
    return characters_df


def prepare_data_for_time_series_of_chapters(chapters_df: pl.DataFrame):
    chapters_df = chapters_df.with_columns(
            pl.col("date").str.strptime(pl.Date, format="%B %d, %Y").alias("date")
    )
    return chapters_df


def plot_bubble_chart_for_time_series_of_chapters(chapters_df: pl.DataFrame):
    fig = go.Figure(
            data=go.Scatter(
                    x=chapters_df["date"],
                    y=chapters_df["chapter"],
                    mode="markers",
                    marker=dict(
                            size=chapters_df["pages"],
                            color=chapters_df["volume"],
                            colorscale="Viridis",
                            opacity=0.8,
                    ),
            )
    )
    fig.update_layout(
            title="One Piece Chapters Over Time",
            xaxis_title="Date",
            yaxis_title="Chapter",
    )
    fig.update_traces(marker=dict(line=dict(width=2, color="DarkSlateGrey")))
    # annotate the bubbles with the number of pages
    fig.update_traces(
            text=chapters_df["pages"],
            textposition="top center",
    )
    fig.show()


def plot_number_of_pages_per_chapter(chapters_df: pl.DataFrame):
    fig = go.Figure(
            data=go.Scatter(
                    x=chapters_df["chapter"],
                    y=chapters_df["pages"],
                    mode="markers",
                    marker=dict(
                            size=chapters_df["pages"],
                            color=chapters_df["volume"],
                            colorscale="Viridis",
                            opacity=0.8,
                    ),
            )
    )
    fig.update_layout(
            title="One Piece Chapters Over Time",
            xaxis_title="Chapter",
            yaxis_title="Pages",
    )
    fig.update_traces(marker=dict(line=dict(width=2, color="DarkSlateGrey")))
    # annotate the bubbles with the number of pages
    fig.update_traces(
            text=chapters_df["pages"],
            textposition="top center",
    )
    fig.show()


def calculate_frequency_of_pages(chapters_df: pl.DataFrame):
    return chapters_df.group_by("pages").agg(pl.col("pages").count().alias("pages_count"))


def plot_frequency_of_pages(frequency_df: pl.DataFrame):
    fig = go.Figure(
            data=go.Bar(
                    x=frequency_df["pages"],
                    y=frequency_df["pages_count"],
                    marker=dict(color='blue')
            )
    )
    fig.update_layout(
            title="Frequency Distribution of Pages",
            xaxis_title="Pages",
            yaxis_title="Frequency"
    )
    fig.show()


def plot_ecdf_of_pages(chapters_df: pl.DataFrame):
    fig = px.ecdf(chapters_df, x="pages", marginal="histogram")
    fig.show()


def create_data_for_character_appearance_network(characters_df: pl.DataFrame):
    """
    This function creates a dataframe for the character appearance network.
    """
    G = nx.Graph()
    # Nodes are characters, edges are chapters if they appear in the same chapter weighted by the number of times
    # they appear in the same chapter
    for chapter in characters_df["chapter"].unique():
        chapter_df = characters_df.filter(pl.col("chapter") == chapter)
        for character in chapter_df["character"].unique():
            G.add_node(character)
            for character2 in chapter_df["character"].unique():
                if character != character2:
                    G.add_edge(character, character2, weight=1)
                    G[character][character2]["weight"] += 1
                else:
                    G.add_edge(character, character2, weight=0)
                    G[character][character2]["weight"] += 0

    return G


def plot_character_appearance_network(G: nx.Graph):
    # use forced directed layout
    pos = nx.spring_layout(G, k=0.1, iterations=50)
    # plot using plotly
    fig = go.Figure()
    for node in G.nodes():
        x, y = pos[node]
        fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers', marker_size=10, marker_color='blue'))
        fig.add_trace(go.Scatter(x=[x], y=[y], mode='text', text=node, textposition='top center'))
        for neighbor in G.neighbors(node):
            x2, y2 = pos[neighbor]
            fig.add_trace(go.Scatter(x=[x, x2], y=[y, y2], mode='lines', line_color='black'))
            fig.add_trace(go.Scatter(x=[x, x2], y=[y, y2], mode='text', text=G[node][neighbor]["weight"],
                                     textposition='top center'))
    fig.show()


def plot_degree_distribution(G: nx.Graph):
    fig = go.Figure()
    degree_sequence = [d for n, d in G.degree()]
    degree_count = {x: degree_sequence.count(x) for x in set(degree_sequence)}
    x = list(degree_count.keys())
    y = list(degree_count.values())
    print(f"Degree distribution: {degree_count}")
    print(f"This means that the most common degree is {max(degree_count, key=degree_count.get)}")
    print(f"Degree sequence: {degree_sequence}")
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers'))
    # add title and axis labels
    fig.update_layout(title='Degree Distribution', xaxis_title='Degree', yaxis_title='Frequency')
    fig.update_traces(marker=dict(size=10, color='blue'))
    fig.update_traces(textposition='top center')
    # add curve fit to identify power law
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line_color='red'))
    # at one corner, show total number of nodes and edges, and average degree, and highest node degree with name
    fig.add_annotation(
            x=0.05, y=0.95, text=f"Total nodes: {len(G.nodes())}\nTotal edges: {len(G.edges())}\nAverage degree: "
                                  f"{sum(degree_sequence) / len(degree_sequence)}\nHighest degree: "
                                  f"{max(degree_sequence)}", showarrow=False, xref="paper", yref="paper")
    # name of the highest degree node
    fig.add_annotation(x=0.05, y=0.90, text=f"Highest degree node: {max(G.degree(), key=lambda x: x[1])[0]}", showarrow=False, xref="paper", yref="paper")
    # print the neighbors of the highest degree node
    fig.add_annotation(x=0.05, y=0.85, text=f"Neighbors of highest degree node: {list(G.neighbors(max(G.degree(), key=lambda x: x[1])[0]))}", showarrow=False, xref="paper", yref="paper")
    fig.show()


def main():
    characters_df = read_data("data/characters.csv")
    chapters_df = read_data("data/chapters.csv")

    # From "appearance" column, extract the chapter number and episode number. "Chapter 551 ; Episode 460",
    # in separate columns.
    characters_df = characters_df.with_columns(
            pl.col("appearance").str.extract(r"Chapter (\d+)", 1).cast(pl.Int64).alias("chapter"),
            pl.col("appearance").str.extract(r"Episode (\d+)", 1).cast(pl.Int64).alias("episode"),
    )
    # drop the appearance column
    characters_df = characters_df.drop("appearance")

    # merge the two dataframes on chapter column
    df = characters_df.join(chapters_df, on="chapter", how="left")
    df = df.drop_nulls("chapter")
    print(df)
    df = coerce_types(df)

    # network analysis
    G = create_data_for_character_appearance_network(df)
    plot_degree_distribution(G)


if __name__ == "__main__":
    main()
