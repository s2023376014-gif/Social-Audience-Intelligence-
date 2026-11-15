import pandas as pd
import plotly.express as px

PALETTE = px.colors.qualitative.Safe


def bar(df, x, y, title, color=None, horizontal=False, pct=False):
    fig = px.bar(df, x=y if horizontal else x, y=x if horizontal else y, color=color, title=title, orientation="h" if horizontal else "v", color_discrete_sequence=PALETTE)
    if pct:
        fig.update_layout(**({"xaxis_tickformat": ".1%"} if horizontal else {"yaxis_tickformat": ".1%"}))
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), legend_title_text="")
    return fig


def scatter_pca(df, color="segment"):
    fig = px.scatter(df, x="pc1", y="pc2", color=color, hover_data=["comment_text"], opacity=0.6, title="Comment clusters (PCA projection, sampled)", color_discrete_sequence=PALETTE)
    fig.update_traces(marker_size=5)
    return fig


def heatmap(df, index_col, title):
    d = df.set_index(index_col)
    return px.imshow(d, text_auto=".0%", aspect="auto", color_continuous_scale="Blues", title=title)


def box(posts, x, y, title):
    return px.box(posts, x=x, y=y, color=x, title=title, points=False, color_discrete_sequence=PALETTE)


def hist(df, x, title, nbins=40):
    return px.histogram(df, x=x, nbins=nbins, title=title, color_discrete_sequence=PALETTE)
