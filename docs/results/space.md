# Embedding space evaluation

## Result

| Model | Pretrained | Vector dim | Clothes embedded | Embedding time | P@20 |
| --- | --- | --- | --- | --- | --- |
| ViT-B-32 | laion2b_s34b_b79k | 512 | 41,802 | 366.4 s (Apple Silicon, MPS) | 0.4962 |

Art (2,000 artworks, same model) embedded in 22.5 s.

The model is `ViT-B-32` / `laion2b_s34b_b79k`, set in `src/amber/config.py`
(`MODEL_NAME`, `MODEL_PRETRAINED`).

**The SigLIP2 comparison was skipped by the user's choice to minimise downloads
(project storage budget).** The plan's second candidate, `ViT-B-16-SigLIP2`, was never
downloaded, so ViT-B-32 is the only model measured here rather than the winner of a
head-to-head. `open_clip.list_pretrained()` does list `("ViT-B-32", "laion2b_s34b_b79k")`;
the SigLIP2 listing was not checked.

## How P@20 is measured

`amber eval-space` builds one text query per `(base_colour, article_type)` group that has at
least 20 items, in the form `"a photo of a {colour} {article type}"` (lower-cased), and ranks
all 41,802 raw (uncentered) vectors against it, which is how text search will rank them.
P@20 is the share of the top 20 that really carry that colour and article type, averaged
over groups. Because a hit needs both the colour and the type to match, 0.50 is a demanding
score.

## Spot checks (contact sheets of the top 12)

* `amber search-text "red floral dress"` (clothes): every tile is red or red-and-white with
  a floral or printed pattern; the top hits are dresses and tunics, followed by kurtas and
  one floral skirt.
* `amber search-text "a stormy seascape" --domain art`: all twelve are sea pictures, mostly
  crashing surf, squalls or ships in rough water (Winslow Homer's Prout's Neck waves lead).

## Storage

Clothing thumbnails were shrunk in place from 288x384 (q85) to 216x288 (q80) with
`amber shrink-thumbs`, which took about 95 s and cut `data/clothes` from 588 MB to 373 MB.
The embeddings add 82 MB (clothes) and 4 MB (art); the whole `data/` directory is 512 MB.
