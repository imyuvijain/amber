"""Measuring an embedding space: can plain text find the right clothes?

`label_retrieval_score` asks the encoder for "a photo of a red dresses"-style queries, one
per (colour, article type) group of the catalog, and checks how many of the nearest items
really belong to that group. It is how Amber chooses between candidate encoders.
"""

import pandas as pd

from amber.space.encoder import Encoder
from amber.space.index import VectorIndex


def label_retrieval_score(
    index: VectorIndex,
    items: pd.DataFrame,
    encoder: Encoder,
    k: int = 20,
    min_group: int = 20,
) -> float:
    """Mean precision@`k` over the `(base_colour, article_type)` groups of `items` that have
    at least `min_group` items in `index`.

    The query for a group is `f"a photo of a {base_colour.lower()} {article_type.lower()}"`.
    Items are ranked by their raw (uncentered) vectors, the same way text search ranks them.
    Raises ValueError if no group is large enough.
    """
    indexed = items[items["id"].isin(set(index.ids))]
    sizes = indexed.groupby(["base_colour", "article_type"]).size()
    groups = sizes[sizes >= min_group].index.tolist()
    if not groups:
        raise ValueError(
            f"no (base_colour, article_type) group has min_group={min_group} items"
        )

    members = indexed.groupby(["base_colour", "article_type"])["id"].agg(set)
    queries = [
        f"a photo of a {colour.lower()} {kind.lower()}" for colour, kind in groups
    ]
    directions = encoder.embed_texts(queries)

    precisions = []
    for group, direction in zip(groups, directions):
        top = index.top_k(direction, k, use_centered=False)
        precisions.append(sum(item_id in members[group] for item_id, _ in top) / k)
    return float(sum(precisions) / len(precisions))
