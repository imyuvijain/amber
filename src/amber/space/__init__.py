"""The shared vector space: encoders, the vector index, and the PCA basis.

Submodules are imported explicitly (`amber.space.index`, `amber.space.basis`,
`amber.space.encoder`) so that the numpy-only ones never pull in torch.
"""
