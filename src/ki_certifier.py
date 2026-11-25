"""Pure-stdlib zero-float verifier for KOASHI-IMOTO-CERT-1.0 certificates.
Supports four exact-Q verification modes aligned with Polloreno (arXiv:2611.33129, Nov 24, 2026)
"A Koashi-Imoto Criterion for Holevo-Rate Visible Compression":
  1. qubit_bloch_dichotomy_exact               (Abstract & Fig. 2: exact Bloch purity/commutator dichotomy)
  2. ki_block_tensor_criterion_exact           (Eq. 3 & Thm 1: exact block-tensor KI decomposition & purity criterion)
  3. pure_atom_caratheodory_decomp_exact       (Def 2, Lemma 3, Eq. 11: exact pure-atom convex decomposition <= |X|d^2+1)
  4. sector_splitting_and_partial_additivity_exact (Props 9-10, Eqs. 4, 5, 12, 18, 19: sector excess & additivity)

Author: Lluis Eriksson
"""
from __future__ import annotations

import re
from fractions import Fraction
from typing import Any, Dict, List, Tuple

from src.ki_exact import (
    MatQ,
    bloch_cross,
    bloch_norm_sq,
    commutant_dimension_exact,
    mat_add,
    mat_commutator,
    mat_direct_sum,
    mat_is_psd_exact,
    mat_is_zero,
    mat_kron,
    mat_mul,
    mat_scale,
    mat_trace,
    pure_projector_exact,
)

SCHEMA_VERSION = "KOASHI-IMOTO-CERT-1.0"
_CANON_INT_RE = re.compile(r"\A(?:0|-?[1-9][0-9]*)\Z")
_CANON_FRAC_RE = re.compile(r"\A-?[1-9][0-9]*/[1-9][0-9]*\Z")


class KoashiImotoCertificateError(ValueError):
    """Raised whenever any KOASHI-IMOTO-CERT-1.0 verification check fails."""


def _assert_strict_json_types(obj: Any, path: str = "root") -> None:
    t = type(obj)
    if t is int or t is str or obj is None:
        return
    if t is dict:
        for k, v in obj.items():
            if type(k) is not str:
                raise KoashiImotoCertificateError(f"Non-str dict key at {path}: {type(k).__name__}")
            _assert_strict_json_types(v, f"{path}.{k}")
        return
    if t is list:
        for i, v in enumerate(obj):
            _assert_strict_json_types(v, f"{path}[{i}]")
        return
    raise KoashiImotoCertificateError(f"Forbidden type {t.__name__} at {path}")


def _parse_int(val: Any, name: str) -> int:
    if type(val) is not int:
        raise KoashiImotoCertificateError(f"{name} must be strict int, got {type(val).__name__}")
    return val


def _parse_rat(val: Any, name: str) -> Fraction:
    if type(val) is int:
        return Fraction(val, 1)
    if type(val) is str:
        if _CANON_INT_RE.match(val):
            return Fraction(int(val), 1)
        if _CANON_FRAC_RE.match(val):
            num_s, den_s = val.split("/")
            num, den = int(num_s), int(den_s)
            if den < 2:
                raise KoashiImotoCertificateError(f"{name} denominator must be >= 2: {val!r}")
            q = Fraction(num, den)
            if q.numerator != num or q.denominator != den:
                raise KoashiImotoCertificateError(f"{name} unreduced fraction: {val!r}")
            return q
        raise KoashiImotoCertificateError(f"{name} non-canonical rational string: {val!r}")
    raise KoashiImotoCertificateError(f"{name} invalid type {type(val).__name__}")


def _parse_density_matrix(raw: Any, dim: int, name: str) -> MatQ:
    if type(raw) is not list or len(raw) != dim:
        raise KoashiImotoCertificateError(f"{name} must be {dim}x{dim} matrix")
    M: MatQ = []
    for i, row in enumerate(raw):
        if type(row) is not list or len(row) != dim:
            raise KoashiImotoCertificateError(f"{name}[{i}] must have length {dim}")
        M.append([_parse_rat(x, f"{name}[{i}][{j}]") for j, x in enumerate(row)])
    if not mat_is_psd_exact(M):
        raise KoashiImotoCertificateError(f"{name} is not symmetric PSD over Q")
    if mat_trace(M) != Fraction(1, 1):
        raise KoashiImotoCertificateError(f"{name} trace != 1 (got {mat_trace(M)})")
    return M


def _verify_mode1_qubit_dichotomy(cert: Dict[str, Any]) -> Dict[str, Any]:
    req = {
        "schema", "mode", "name", "priors", "bloch_vectors",
        "claimed_regime", "claimed_linear_mix_defect", "claimed_max_commutator_sq",
    }
    if set(cert.keys()) != req:
        raise KoashiImotoCertificateError(f"Mode 1 key mismatch: {sorted(set(cert.keys()) ^ req)}")
    p_raw = cert["priors"]
    b_raw = cert["bloch_vectors"]
    if type(p_raw) is not list or type(b_raw) is not list or len(p_raw) < 2 or len(p_raw) != len(b_raw) or len(p_raw) > 16:
        raise KoashiImotoCertificateError("priors and bloch_vectors must have equal length in [2, 16]")
    priors = [_parse_rat(x, f"priors[{i}]") for i, x in enumerate(p_raw)]
    if any(p <= 0 for p in priors) or sum(priors, Fraction(0, 1)) != Fraction(1, 1):
        raise KoashiImotoCertificateError("priors must be strictly positive and sum to 1")
    vecs: List[Tuple[Fraction, Fraction, Fraction]] = []
    for i, v in enumerate(b_raw):
        if type(v) is not list or len(v) != 3:
            raise KoashiImotoCertificateError(f"bloch_vectors[{i}] must have length 3")
        tv = (_parse_rat(v[0], f"b[{i}][0]"), _parse_rat(v[1], f"b[{i}][1]"), _parse_rat(v[2], f"b[{i}][2]"))
        if bloch_norm_sq(tv) > Fraction(1, 1):
            raise KoashiImotoCertificateError(f"bloch_vectors[{i}] outside unit Bloch ball")
        vecs.append(tv)

    # Exact purity Tr(rho^2) = (1 + |r|^2)/2 -> linear defect 1 - Tr(rho^2) = (1 - |r|^2)/2
    r_lin = Fraction(0, 1)
    all_pure = True
    for p, v in zip(priors, vecs):
        ns = bloch_norm_sq(v)
        if ns != Fraction(1, 1):
            all_pure = False
        r_lin += p * (Fraction(1, 1) - ns) / 2

    max_comm_sq = Fraction(0, 1)
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            cs = bloch_norm_sq(bloch_cross(vecs[i], vecs[j]))
            if cs > max_comm_sq:
                max_comm_sq = cs
    all_comm = (max_comm_sq == 0)

    if all_pure:
        expected = "holevo_achievable_pure"
    elif all_comm:
        expected = "holevo_achievable_commuting"
    else:
        expected = "strict_gap_mixed_noncommuting"

    if cert["claimed_regime"] != expected:
        raise KoashiImotoCertificateError(f"claimed_regime={cert['claimed_regime']!r} != expected {expected!r}")
    if _parse_rat(cert["claimed_linear_mix_defect"], "claimed_linear_mix_defect") != r_lin:
        raise KoashiImotoCertificateError(f"claimed_linear_mix_defect mismatch: exact={r_lin}")
    if _parse_rat(cert["claimed_max_commutator_sq"], "claimed_max_commutator_sq") != max_comm_sq:
        raise KoashiImotoCertificateError(f"claimed_max_commutator_sq mismatch: exact={max_comm_sq}")

    return {
        "verified": True,
        "mode": cert["mode"],
        "num_signals": len(vecs),
        "regime": expected,
        "r_lin": str(r_lin),
        "max_comm_sq": str(max_comm_sq),
    }


def _verify_mode2_ki_block_tensor(cert: Dict[str, Any]) -> Dict[str, Any]:
    req = {
        "schema", "mode", "name", "dim_W", "priors", "signal_states",
        "sector_core_dims", "sector_redundant_dims", "sector_probs",
        "core_states", "redundant_states", "claimed_criterion", "claimed_core_linear_mix",
    }
    if set(cert.keys()) != req:
        raise KoashiImotoCertificateError(f"Mode 2 key mismatch: {sorted(set(cert.keys()) ^ req)}")
    dW = _parse_int(cert["dim_W"], "dim_W")
    if not (2 <= dW <= 8):
        raise KoashiImotoCertificateError(f"dim_W must be in [2, 8], got {dW}")
    p_raw = cert["priors"]
    s_raw = cert["signal_states"]
    if type(p_raw) is not list or type(s_raw) is not list or len(p_raw) < 2 or len(p_raw) != len(s_raw) or len(p_raw) > 8:
        raise KoashiImotoCertificateError("priors and signal_states must have equal length in [2, 8]")
    X = len(p_raw)
    priors = [_parse_rat(x, f"priors[{i}]") for i, x in enumerate(p_raw)]
    if any(p <= 0 for p in priors) or sum(priors, Fraction(0, 1)) != Fraction(1, 1):
        raise KoashiImotoCertificateError("priors must be > 0 and sum to 1")
    signals = [_parse_density_matrix(m, dW, f"signal_states[{x}]") for x, m in enumerate(s_raw)]

    qdims_raw = cert["sector_core_dims"]
    ndims_raw = cert["sector_redundant_dims"]
    if type(qdims_raw) is not list or type(ndims_raw) is not list or not qdims_raw or len(qdims_raw) != len(ndims_raw):
        raise KoashiImotoCertificateError("sector_core_dims and sector_redundant_dims must be equal non-empty lists")
    J = len(qdims_raw)
    qdims = [_parse_int(v, f"sector_core_dims[{j}]") for j, v in enumerate(qdims_raw)]
    ndims = [_parse_int(v, f"sector_redundant_dims[{j}]") for j, v in enumerate(ndims_raw)]
    if any(v < 1 for v in qdims) or any(v < 1 for v in ndims):
        raise KoashiImotoCertificateError("All sector dimensions must be >= 1")
    if sum(q * n for q, n in zip(qdims, ndims)) != dW:
        raise KoashiImotoCertificateError("sum(d_Qj * d_Nj) != dim_W")

    red_raw = cert["redundant_states"]
    if type(red_raw) is not list or len(red_raw) != J:
        raise KoashiImotoCertificateError("redundant_states must have length J")
    xis = [_parse_density_matrix(red_raw[j], ndims[j], f"redundant_states[{j}]") for j in range(J)]

    sp_raw = cert["sector_probs"]
    cs_raw = cert["core_states"]
    if type(sp_raw) is not list or len(sp_raw) != X or type(cs_raw) is not list or len(cs_raw) != X:
        raise KoashiImotoCertificateError("sector_probs and core_states must have length |X|")

    r_lin = Fraction(0, 1)
    occupied_cores_by_sector: List[List[Tuple[int, MatQ, bool]]] = [[] for _ in range(J)]
    for x in range(X):
        if type(sp_raw[x]) is not list or len(sp_raw[x]) != J:
            raise KoashiImotoCertificateError(f"sector_probs[{x}] must have length J")
        if type(cs_raw[x]) is not list or len(cs_raw[x]) != J:
            raise KoashiImotoCertificateError(f"core_states[{x}] must have length J")
        pjx = [_parse_rat(v, f"sector_probs[{x}][{j}]") for j, v in enumerate(sp_raw[x])]
        if any(v < 0 for v in pjx) or sum(pjx, Fraction(0, 1)) != Fraction(1, 1):
            raise KoashiImotoCertificateError(f"sector_probs[{x}] must be >= 0 and sum to 1")
        blocks: List[MatQ] = []
        for j in range(J):
            rho_xj = _parse_density_matrix(cs_raw[x][j], qdims[j], f"core_states[{x}][{j}]")
            blocks.append(mat_scale(pjx[j], mat_kron(rho_xj, xis[j])))
            if pjx[j] > 0:
                rho2 = mat_mul(rho_xj, rho_xj)
                purity = mat_trace(rho2)
                is_pure = (rho2 == rho_xj) and (purity == Fraction(1, 1))
                r_lin += priors[x] * pjx[j] * (Fraction(1, 1) - purity)
                occupied_cores_by_sector[j].append((x, rho_xj, is_pure))
        reconstructed = mat_direct_sum(blocks)
        if reconstructed != signals[x]:
            raise KoashiImotoCertificateError(f"KI block-tensor reconstruction mismatch for signal {x}")

    # Determine Theorem 1 exact criterion with exact Wedderburn/Schur trivial-commutant maximality check
    all_pure = (r_lin == 0)
    if all_pure:
        expected_crit = "holevo_achievable_all_cores_pure"
    else:
        # EVERY mixed occupied sector must have trivial commutant (dim == 1), certifying irreducibility
        # and ruling out reducible direct sums, hidden redundant factors, and partially commuting sectors!
        for j in range(J):
            occ = occupied_cores_by_sector[j]
            if any(not ip for (_, _, ip) in occ):
                cdim = commutant_dimension_exact([rho for (_, rho, _) in occ])
                if cdim != 1:
                    raise KoashiImotoCertificateError(
                        f"Mixed core sector {j} is reducible / non-maximal (commutant dim={cdim} != 1)"
                    )
        expected_crit = "strict_gap_mixed_core"

    if cert["claimed_criterion"] != expected_crit:
        raise KoashiImotoCertificateError(f"claimed_criterion={cert['claimed_criterion']!r} != {expected_crit!r}")
    if _parse_rat(cert["claimed_core_linear_mix"], "claimed_core_linear_mix") != r_lin:
        raise KoashiImotoCertificateError(f"claimed_core_linear_mix mismatch: exact={r_lin}")

    return {
        "verified": True,
        "mode": cert["mode"],
        "dim_W": dW,
        "num_sectors": J,
        "criterion": expected_crit,
        "core_r_lin": str(r_lin),
    }


def _verify_mode3_pure_atom_decomp(cert: Dict[str, Any]) -> Dict[str, Any]:
    req = {
        "schema", "mode", "name", "dim_W", "priors", "signal_states",
        "num_flags_U", "flag_probs", "pure_atom_vectors", "claimed_caratheodory_bound",
    }
    if set(cert.keys()) != req:
        raise KoashiImotoCertificateError(f"Mode 3 key mismatch: {sorted(set(cert.keys()) ^ req)}")
    dW = _parse_int(cert["dim_W"], "dim_W")
    if not (2 <= dW <= 6):
        raise KoashiImotoCertificateError(f"dim_W must be in [2, 6], got {dW}")
    p_raw = cert["priors"]
    s_raw = cert["signal_states"]
    if type(p_raw) is not list or type(s_raw) is not list or len(p_raw) < 2 or len(p_raw) != len(s_raw) or len(p_raw) > 8:
        raise KoashiImotoCertificateError("priors and signal_states must have equal length in [2, 8]")
    X = len(p_raw)
    priors = [_parse_rat(x, f"priors[{i}]") for i, x in enumerate(p_raw)]
    if any(p <= 0 for p in priors) or sum(priors, Fraction(0, 1)) != Fraction(1, 1):
        raise KoashiImotoCertificateError("priors must be > 0 and sum to 1")
    signals = [_parse_density_matrix(m, dW, f"signal_states[{x}]") for x, m in enumerate(s_raw)]

    U = _parse_int(cert["num_flags_U"], "num_flags_U")
    carat = X * dW * dW + 1
    if _parse_int(cert["claimed_caratheodory_bound"], "claimed_caratheodory_bound") != carat:
        raise KoashiImotoCertificateError(f"claimed_caratheodory_bound != |X|d^2+1 = {carat}")
    if not (1 <= U <= carat):
        raise KoashiImotoCertificateError(f"num_flags_U={U} violates 1 <= |U| <= {carat} (Lemma 3)")

    fp_raw = cert["flag_probs"]
    pv_raw = cert["pure_atom_vectors"]
    if type(fp_raw) is not list or len(fp_raw) != X or type(pv_raw) is not list or len(pv_raw) != X:
        raise KoashiImotoCertificateError("flag_probs and pure_atom_vectors must have length |X|")

    for x in range(X):
        if type(fp_raw[x]) is not list or len(fp_raw[x]) != U or type(pv_raw[x]) is not list or len(pv_raw[x]) != U:
            raise KoashiImotoCertificateError(f"flag_probs[{x}] and pure_atom_vectors[{x}] must have length U")
        pux = [_parse_rat(v, f"flag_probs[{x}][{u}]") for u, v in enumerate(fp_raw[x])]
        if any(v < 0 for v in pux) or sum(pux, Fraction(0, 1)) != Fraction(1, 1):
            raise KoashiImotoCertificateError(f"flag_probs[{x}] must be >= 0 and sum to 1")
        acc = [[Fraction(0, 1)] * dW for _ in range(dW)]
        for u in range(U):
            v_raw = pv_raw[x][u]
            if type(v_raw) is not list or len(v_raw) != dW:
                raise KoashiImotoCertificateError(f"pure_atom_vectors[{x}][{u}] must have length {dW}")
            vec = [_parse_rat(c, f"vec[{x}][{u}][{k}]") for k, c in enumerate(v_raw)]
            if pux[u] > 0:
                if all(c == 0 for c in vec):
                    raise KoashiImotoCertificateError(f"Occupied pure atom ({x},{u}) has zero vector")
                proj = pure_projector_exact(vec)
                if mat_mul(proj, proj) != proj or mat_trace(proj) != Fraction(1, 1):
                    raise KoashiImotoCertificateError("Impossible: pure projector failed idempotence")
                acc = mat_add(acc, mat_scale(pux[u], proj))
        if acc != signals[x]:
            raise KoashiImotoCertificateError(f"Pure-atom convex mixture failed to reconstruct signal_states[{x}]")

    return {
        "verified": True,
        "mode": cert["mode"],
        "dim_W": dW,
        "num_signals": X,
        "num_flags_U": U,
        "caratheodory_bound": carat,
    }


def _verify_mode4_sector_splitting(cert: Dict[str, Any]) -> Dict[str, Any]:
    req = {
        "schema", "mode", "name",
        "sector_weights_q", "sector_F", "sector_chi", "I_X_J", "H_J_given_X", "r_mix",
        "claimed_total_F", "claimed_total_chi", "claimed_visible_excess", "claimed_blind_excess",
        "prod_chi1", "prod_F1", "prod_F2", "prod_F12",
    }
    if set(cert.keys()) != req:
        raise KoashiImotoCertificateError(f"Mode 4 key mismatch: {sorted(set(cert.keys()) ^ req)}")
    q_raw, f_raw, c_raw = cert["sector_weights_q"], cert["sector_F"], cert["sector_chi"]
    if type(q_raw) is not list or type(f_raw) is not list or type(c_raw) is not list or not q_raw:
        raise KoashiImotoCertificateError("sector lists must be non-empty")
    J = len(q_raw)
    if len(f_raw) != J or len(c_raw) != J:
        raise KoashiImotoCertificateError("sector_weights_q, sector_F, sector_chi length mismatch")
    qs = [_parse_rat(x, f"q[{j}]") for j, x in enumerate(q_raw)]
    fs = [_parse_rat(x, f"F[{j}]") for j, x in enumerate(f_raw)]
    cs = [_parse_rat(x, f"chi[{j}]") for j, x in enumerate(c_raw)]
    if any(q <= 0 for q in qs) or sum(qs, Fraction(0, 1)) != Fraction(1, 1):
        raise KoashiImotoCertificateError("sector_weights_q must be > 0 and sum to 1")
    for j in range(J):
        if fs[j] < cs[j] or cs[j] < 0:
            raise KoashiImotoCertificateError(f"Sector {j} violates 0 <= chi_j <= F_j")

    ixj = _parse_rat(cert["I_X_J"], "I_X_J")
    hjx = _parse_rat(cert["H_J_given_X"], "H_J_given_X")
    rmix = _parse_rat(cert["r_mix"], "r_mix")
    if ixj < 0 or hjx < 0 or rmix < 0:
        raise KoashiImotoCertificateError("I_X_J, H_J_given_X, r_mix must be >= 0")

    tot_F = ixj + sum((q * f for q, f in zip(qs, fs)), Fraction(0, 1))
    tot_chi = ixj + sum((q * c for q, c in zip(qs, cs)), Fraction(0, 1))
    vis_ex = sum((q * (f - c) for q, f, c in zip(qs, fs, cs)), Fraction(0, 1))
    if vis_ex != tot_F - tot_chi:
        raise KoashiImotoCertificateError("Impossible: Proposition 10 sector-splitting excess mismatch")
    if vis_ex > rmix:
        raise KoashiImotoCertificateError("Visible excess exceeds Theorem 1 upper bound r_mix")
    blind_ex = hjx + rmix

    if _parse_rat(cert["claimed_total_F"], "claimed_total_F") != tot_F:
        raise KoashiImotoCertificateError("claimed_total_F mismatch")
    if _parse_rat(cert["claimed_total_chi"], "claimed_total_chi") != tot_chi:
        raise KoashiImotoCertificateError("claimed_total_chi mismatch")
    if _parse_rat(cert["claimed_visible_excess"], "claimed_visible_excess") != vis_ex:
        raise KoashiImotoCertificateError("claimed_visible_excess mismatch")
    if _parse_rat(cert["claimed_blind_excess"], "claimed_blind_excess") != blind_ex:
        raise KoashiImotoCertificateError("claimed_blind_excess mismatch")

    # Proposition 9 partial additivity check
    c1 = _parse_rat(cert["prod_chi1"], "prod_chi1")
    f1 = _parse_rat(cert["prod_F1"], "prod_F1")
    f2 = _parse_rat(cert["prod_F2"], "prod_F2")
    f12 = _parse_rat(cert["prod_F12"], "prod_F12")
    if c1 < 0 or f1 < c1 or f2 < 0:
        raise KoashiImotoCertificateError("Invalid prod_chi1/F1/F2 bounds")
    if not (c1 + f2 <= f12 <= f1 + f2):
        raise KoashiImotoCertificateError("Proposition 9 partial-additivity sandwich violated")
    if f1 == c1 and f12 != f1 + f2:
        raise KoashiImotoCertificateError("Pure-core factor must force exact additivity F12 == F1 + F2")

    return {
        "verified": True,
        "mode": cert["mode"],
        "num_sectors": J,
        "total_F": str(tot_F),
        "visible_excess": str(vis_ex),
        "blind_excess": str(blind_ex),
        "exact_additive": (f1 == c1),
    }


def verify_koashi_imoto_certificate_exact(cert: Dict[str, Any]) -> Dict[str, Any]:
    if type(cert) is not dict:
        raise KoashiImotoCertificateError("Certificate root must be a dict")
    _assert_strict_json_types(cert)
    if cert.get("schema") != SCHEMA_VERSION:
        raise KoashiImotoCertificateError(f"Invalid schema: {cert.get('schema')!r}")
    name = cert.get("name")
    if type(name) is not str or not name.strip():
        raise KoashiImotoCertificateError(f"Certificate name must be non-empty str, got {name!r}")
    mode = cert.get("mode")
    if mode == "qubit_bloch_dichotomy_exact":
        return _verify_mode1_qubit_dichotomy(cert)
    if mode == "ki_block_tensor_criterion_exact":
        return _verify_mode2_ki_block_tensor(cert)
    if mode == "pure_atom_caratheodory_decomp_exact":
        return _verify_mode3_pure_atom_decomp(cert)
    if mode == "sector_splitting_and_partial_additivity_exact":
        return _verify_mode4_sector_splitting(cert)
    raise KoashiImotoCertificateError(f"Unknown mode: {mode!r}")
