# Machine-Checked Exact-Rational Koashi–Imoto Decomposition, Qubit Dichotomy, and Pure-Atom Certificates for Holevo-Rate Visible Compression

**Author:** Lluis Eriksson (with transparent AI assistance via Google Antigravity)  
**License:** MIT (code, Lean 4 proofs, certificates) / CC-BY-4.0 (manuscript)

Zero-dependency Lean 4 formalization and pure-stdlib exact-$\mathbb{Q}$ certificate verifier (`KOASHI-IMOTO-CERT-1.0`) for the finite-dimensional algebraic structure of Polloreno (*A Koashi–Imoto Criterion for Holevo-Rate Visible Compression*, `arXiv:2611.33129`, Nov 24, 2026).

## Verification Summary
- **Lean 4 Kernel (`lean/KoashiImotoHolevo.lean`, Lean 4.35.0):** 18/18 theorems, 0 warnings, 0 `sorry`, 0 `Classical.choice` (all 18 depend solely on `[propext, Quot.sound]`, including full 16-entry $4\times 4$ Kronecker purity expansion and full biconditional sector-zero rigidity).
- **Exact-$\mathbb{Q}$ Certificate Atlas (`results/certificates/*.cert.json`):** 16 canonical certificates across 4 modes (qubit Bloch purity/commutator dichotomy; multi-sector block-tensor KI decomposition with exact rational $LDL^T$ PSD verification and Wedderburn/Schur trivial-commutant maximality check; pure-atom Carathéodory mixtures without square roots; Proposition 9–10 sector splitting and partial additivity), replayed with zero floats in `0.075`–`0.655 ms`.
- **Cross-Checks & Red-Team Suite (`tests/test_suite.py`):** Direct exact $\mathbb{Q}[i]$ $2\times 2$ Hermitian matrix purity & Frobenius commutator norm cross-checks, exact $LDL^T$ indefinite-matrix rejection, commutant-dimension irreducibility checks, 16/16 certificate replays, and **57/57** live adversarial attacks rejected.

## One-Command Reproducibility
```bash
python scripts/verify_all.py
```
To regenerate all certificates, experiment JSONs, and 300-DPI figures from scratch:
```bash
python scripts/run_all_experiments.py
python scripts/make_figures.py
python scripts/verify_all.py
```
