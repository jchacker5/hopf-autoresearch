# Hopf Latent Spaces: Nested Complex Hopf Fibrations as a Topological Inductive Bias

**Joseph Defendre**  
MSAI Candidate, Khoury College of Computer Sciences  
Northeastern University, Boston, MA 02115, USA  
j.defendre@northeastern.edu  
June 27, 2026

## Abstract

We introduce **Hopf Latent Spaces**, a novel geometric inductive bias for neural network latent representations inspired by the nested complex Hopf fibration \( S^1 \hookrightarrow S^{2n+1} \twoheadrightarrow \mathbb{CP}^n \). By projecting embeddings onto linked circle fibers over progressively higher-dimensional complex projective bases, the latent space naturally acquires hierarchical organization, intrinsic periodicity, and non-trivial topological linking. This structure is hypothesized to improve disentanglement, long-range dependency modeling, training stability, and performance on mathematical reasoning and structured data tasks. We provide the full mathematical formulation, a differentiable implementation sketch, and a reproducible experimental plan using small-scale models and automated research tooling. Preliminary design predicts measurable gains in generalization, sample efficiency, and exact-match accuracy on arithmetic and algorithmic tasks compared to Euclidean and hyperspherical baselines.

**Keywords:** geometric deep learning, manifold learning, Hopf fibration, topological inductive bias, hierarchical latent representations, latent space geometry

## 1. Introduction

Standard latent spaces in autoencoders, VAEs, and transformers are typically Euclidean vectors or simple spheres. While convenient, these spaces lack built-in mechanisms for hierarchy, periodicity, or topological relations.

Hopf Latent Spaces address this by encoding representations through the geometry of the Hopf fibration and its nested generalizations.

## 2. Mathematical formulation

### 2.1 Classical Hopf fibration

The Hopf map \( \pi: S^3 \to S^2 \) arises from viewing \( S^3 \subset \mathbb{C}^2 \) as unit pairs \( (z_1, z_2) \) with \( |z_1|^2 + |z_2|^2 = 1 \):

\[
\pi(z_1, z_2) = \left( |z_1|^2 - |z_2|^2,\; 2\,\mathrm{Re}(z_1 \bar{z}_2),\; 2\,\mathrm{Im}(z_1 \bar{z}_2) \right) \in S^2
\]

Each point on \( S^2 \) has a fiber \( \pi^{-1}(p) \cong S^1 \): a circle of phases. These fibers are **linked** — any two fibers link with linking number 1.

Quaternion form: \( q = w + xi + yj + zk \in S^3 \), same map:

\[
\pi(q) = \left( 2(wx + yz),\; 2(wy - zx),\; w^2 - x^2 - y^2 + z^2 \right)
\]

Fiber phase: \( \theta = \mathrm{atan2}(y, w) \) (one valid coordinate chart).

### 2.2 Nested complex Hopf fibrations

Generalize to \( S^{2n+1} \twoheadrightarrow \mathbb{CP}^n \). Nesting stacks fibrations:

- **Level 0:** \( S^3 \to S^2 \) — local hierarchy / binary branching
- **Level 1:** \( S^5 \to \mathbb{CP}^2 \) — higher relational structure
- **Level k:** fiber \( S^1 \), total space \( S^{2k+3} \), base \( \mathbb{CP}^{k+1} \)

Latent vector \( h \in \mathbb{R}^d \) is partitioned into \( L \) quaternion blocks \( q_\ell \in \mathbb{R}^4 \), each normalized to \( S^3 \), mapped through \( \pi_\ell \), producing:
- base coordinates \( b_\ell \in S^2 \subset \mathbb{R}^3 \)
- fiber phases \( \phi_\ell \in S^1 \subset \mathbb{R}^2 \) via \( (\cos\phi, \sin\phi) \)

Structured representation: \( \mathrm{concat}_\ell(b_\ell, \phi_\ell) \), then linear re-projection to model dimension.

### 2.3 Differentiable implementation (train.py)

```python
# Unit quaternion normalization → Hopf map → phase fiber → re-embed
q = normalize(x[..., 4])           # S^3
base = hopf_map(q)                 # S^2
fiber = (cos(atan2), sin(atan2))   # S^1
h' = out_proj(concat(base, fiber))
x = mix_alpha * h' + (1 - mix_alpha) * x   # residual blend
```

**Constraints for stable training:**
- Always `F.normalize(..., eps=1e-6)` before manifold maps
- Use residual blending (`HOPF_MIX_ALPHA`) — pure projection can collapse gradients early
- Keep parameter count comparable to baseline when comparing `val_bpb`

### 2.4 Baselines

| Baseline | Description |
|----------|-------------|
| Euclidean | Standard `nn.Embedding` + LayerNorm (default) |
| Hypersphere | `F.normalize(x, dim=-1)` on embeddings |
| Hopf (ours) | Nested Hopf fibration projector |

## 3. Experimental plan (autoresearch)

**Platform:** Karpathy autoresearch loop — 5 min fixed training, `val_bpb` metric.

**Hypotheses:**
1. Hopf input projection lowers `val_bpb` vs Euclidean baseline
2. `HOPF_NUM_LEVELS=2` beats `HOPF_NUM_LEVELS=1`
3. Hopf beats hypersphere-only normalization
4. Per-block Hopf helps deeper models (`DEPTH >= 4`)

**Future work (outside autoresearch scope):**
- Arithmetic / algorithmic eval harness (exact-match)
- VAE latent Hopf regularization
- Phase-consistency loss across sequence positions

## 4. References (selected)

- Hopf, H. (1931). Über die Abbildungen der dreidimensionalen Sphäre auf die Kugelfläche.
- Urbán, K. & Szirtes, G. (2017). Hopf fibrations and related topics.
- Nickel & Kiela (2017). Poincaré embeddings for learning hierarchical representations.
- Karpathy (2026). autoresearch — autonomous LLM pretraining experiments.