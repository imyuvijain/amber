"""Turn the event log into topic documents: one cleaned text per event, with a weight.

Deliberate actions (searching, liking, saving, commenting) weigh more than passive
viewing. Rows that name a creator instead of describing content (follows, account
searches, a comment whose text is the username) are dropped: topics, never creators.
Ads and emoji-only comment likes are dropped too.
"""

import re

import pandas as pd

# (source, kind) -> (weight, deliberate). Anything missing is not a topic document.
KIND_WEIGHTS: dict[tuple[str, str], tuple[float, bool]] = {
    ("youtube", "watch"): (1.0, False),
    ("youtube", "activity"): (1.0, False),
    ("youtube", "like"): (3.0, True),
    ("youtube", "search"): (3.0, True),
    ("google", "search"): (3.0, True),
    ("google", "visit"): (1.0, False),
    ("google", "activity"): (0.5, False),
    ("google", "discover"): (0.5, False),
    ("instagram", "like"): (2.0, True),
    ("instagram", "save"): (4.0, True),
    ("instagram", "comment"): (3.0, True),
    ("instagram", "story_interaction"): (2.0, True),
    ("instagram", "view_post"): (1.0, False),
    ("instagram", "view_video"): (1.0, False),
    ("instagram", "view_story"): (1.0, False),
}
_VERB = re.compile(
    r"^(watched|searched for|visited|viewed|liked|saved|used|answered|commented on)\s+",
    re.IGNORECASE,
)
_URL = re.compile(r"^https?://\S+$")
_WORDISH = re.compile(r"\w", re.UNICODE)
DOC_COLUMNS = ["ts", "source", "kind", "text", "weight", "deliberate"]


def clean_text(s) -> str | None:
    """Strip a leading verb; None for empty, URL-only or emoji-only text."""
    if not isinstance(s, str):
        return None
    s = _VERB.sub("", " ".join(s.split()))
    if not s or _URL.match(s) or not _WORDISH.search(s):
        return None
    return s


def build_docs(events: pd.DataFrame) -> pd.DataFrame:
    """Topic documents from an event log; extra columns (life context) are kept."""
    ev = events[~events["from_ads"].astype(bool)]
    keys = list(zip(ev["source"], ev["kind"]))
    ev = ev[[k in KIND_WEIGHTS for k in keys]].copy()
    ev["text"] = ev["text"].map(clean_text)
    if "creator_id" in ev:
        ev = ev[ev["text"] != ev["creator_id"]]
    ev = ev[ev["text"].notna()]
    wd = [KIND_WEIGHTS[k] for k in zip(ev["source"], ev["kind"])]
    ev["weight"] = [w for w, _ in wd]
    ev["deliberate"] = [d for _, d in wd]
    extra = [c for c in ev.columns if c not in DOC_COLUMNS and c not in ("item_id", "creator_id", "url", "from_ads")]
    return ev[DOC_COLUMNS + extra].reset_index(drop=True)
