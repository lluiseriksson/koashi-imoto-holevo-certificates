"""Generate all 16 canonical KOASHI-IMOTO-CERT-1.0 certificates and 4 experiment JSONs for Paper 24.
Author: Lluis Eriksson
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
import time
from fractions import Fraction
from typing import Dict, List, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ki_certifier import verify_koashi_imoto_certificate_exact
from src.ki_exact import (
    MatQ,
    bloch_cross,
    bloch_norm_sq,
    fmt_rat,
    mat_add,
    mat_direct_sum,
    mat_kron,
    mat_mul,
    mat_scale,
    mat_trace,
    pure_projector_exact,
)

RESULTS = ROOT / "results"
CERTS = RESULTS / "certificates"
RESULTS.mkdir(parents=True, exist_ok=True)
CERTS.mkdir(parents=True, exist_ok=True)


def _fmt_mat(M: MatQ) -> List[List[str]]:
    return [[fmt_rat(x) for x in row] for row in M]


def _bloch_to_real_dm(rx: Fraction, rz: Fraction) -> MatQ:
    """Real-symmetric qubit density matrix (I + rx*X + rz*Z)/2 over Q."""
    half = Fraction(1, 2)
    return [
        [half * (1 + rz), half * rx],
        [half * rx, half * (1 - rz)],
    ]


def _bin_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def _qubit_vn_entropy(rx: float, ry: float, rz: float) -> float:
    r = math.sqrt(rx * rx + ry * ry + rz * rz)
    return _bin_entropy(0.5 * (1.0 + min(1.0, r)))


def _emit_cert(filename: str, cert: Dict) -> Dict:
    t0 = time.perf_counter()
    summary = verify_koashi_imoto_certificate_exact(cert)
    ms = (time.perf_counter() - t0) * 1000.0
    (CERTS / filename).write_text(json.dumps(cert, indent=2) + "\n", encoding="utf-8")
    return {"file": filename, "mode": cert["mode"], "verify_ms": round(ms, 4), "summary": summary}


def main() -> None:
    atlas_rows = []

    # =========================================================================
    # MODE 1: 4 Qubit Bloch Dichotomy Certificates (Polloreno Fig. 2(a,b,c) + 3-signal)
    # =========================================================================
    # 1a: Mixed commuting (Fig. 2a): collinear along Z, r0 = (0,0,7/10), r1 = (0,0,-7/10)
    m1_cases = [
        (
            "qubit_fig2a_mixed_commuting.cert.json",
            "mode1_qubit_fig2a_mixed_commuting",
            [Fraction(1, 2), Fraction(1, 2)],
            [(Fraction(0), Fraction(0), Fraction(7, 10)), (Fraction(0), Fraction(0), Fraction(-7, 10))],
            "holevo_achievable_commuting",
        ),
        (
            "qubit_fig2b_pure_nonorthogonal.cert.json",
            "mode1_qubit_fig2b_pure_nonorthogonal",
            [Fraction(1, 2), Fraction(1, 2)],
            [(Fraction(3, 5), Fraction(0), Fraction(4, 5)), (Fraction(-3, 5), Fraction(0), Fraction(4, 5))],
            "holevo_achievable_pure",
        ),
        (
            "qubit_fig2c_mixed_noncommuting.cert.json",
            "mode1_qubit_fig2c_mixed_noncommuting",
            [Fraction(1, 2), Fraction(1, 2)],
            # Exact off-center chord: u=(3/5,0,4/5), v=(-3/5,0,4/5), t0=1/6 -> r0=(2/5,0,4/5), r1=(-2/5,0,4/5)
            [(Fraction(2, 5), Fraction(0), Fraction(4, 5)), (Fraction(-2, 5), Fraction(0), Fraction(4, 5))],
            "strict_gap_mixed_noncommuting",
        ),
        (
            "qubit_trine_mixed_3signal.cert.json",
            "mode1_qubit_trine_mixed_3signal",
            [Fraction(1, 3), Fraction(1, 3), Fraction(1, 3)],
            [
                (Fraction(0), Fraction(0), Fraction(4, 5)),
                (Fraction(3, 5), Fraction(0), Fraction(-2, 5)),
                (Fraction(-3, 5), Fraction(0), Fraction(-2, 5)),
            ],
            "strict_gap_mixed_noncommuting",
        ),
    ]
    for fname, cname, priors, vecs, regime in m1_cases:
        r_lin = sum((p * (1 - bloch_norm_sq(v)) / 2 for p, v in zip(priors, vecs)), Fraction(0, 1))
        mc = Fraction(0, 1)
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                cs = bloch_norm_sq(bloch_cross(vecs[i], vecs[j]))
                if cs > mc:
                    mc = cs
        cert = {
            "schema": "KOASHI-IMOTO-CERT-1.0",
            "mode": "qubit_bloch_dichotomy_exact",
            "name": cname,
            "priors": [fmt_rat(p) for p in priors],
            "bloch_vectors": [[fmt_rat(c) for c in v] for v in vecs],
            "claimed_regime": regime,
            "claimed_linear_mix_defect": fmt_rat(r_lin),
            "claimed_max_commutator_sq": fmt_rat(mc),
        }
        atlas_rows.append(_emit_cert(fname, cert))

    # =========================================================================
    # MODE 2: 4 Koashi-Imoto Block-Tensor Decomposition Certificates (Eq. 3 & Thm 1)
    # =========================================================================
    # 2a: Polloreno's exact Qutrit Erasure Source (p. 4):
    #     omega_x = 1/2 |e><e| (+) 1/2 |psi_x><psi_x| -> 2 sectors (dQ=[1,2], dN=[1,1]), pure cores => R_cr = chi < R_blind!
    psi0 = pure_projector_exact([Fraction(1), Fraction(0)])
    psi1 = pure_projector_exact([Fraction(3), Fraction(4)])  # |<psi0|psi1>|^2 = 9/25
    one1 = [[Fraction(1)]]
    erasure_sigs = [
        mat_direct_sum([mat_scale(Fraction(1, 2), one1), mat_scale(Fraction(1, 2), psi0)]),
        mat_direct_sum([mat_scale(Fraction(1, 2), one1), mat_scale(Fraction(1, 2), psi1)]),
    ]
    atlas_rows.append(_emit_cert("ki_qutrit_erasure_pure_cores.cert.json", {
        "schema": "KOASHI-IMOTO-CERT-1.0",
        "mode": "ki_block_tensor_criterion_exact",
        "name": "mode2_qutrit_erasure_pure_cores",
        "dim_W": 3,
        "priors": ["1/2", "1/2"],
        "signal_states": [_fmt_mat(s) for s in erasure_sigs],
        "sector_core_dims": [1, 2],
        "sector_redundant_dims": [1, 1],
        "sector_probs": [["1/2", "1/2"], ["1/2", "1/2"]],
        "core_states": [[_fmt_mat(one1), _fmt_mat(psi0)], [_fmt_mat(one1), _fmt_mat(psi1)]],
        "redundant_states": [_fmt_mat(one1), _fmt_mat(one1)],
        "claimed_criterion": "holevo_achievable_all_cores_pure",
        "claimed_core_linear_mix": "0",
    }))

    # 2b: 4D Single-Sector Tensor Product with Non-Trivial Mixed Redundant Factor xi (dQ=[2], dN=[2]):
    #     omega_x = |psi_x><psi_x| (x) xi_mixed -> global omega_x is MIXED, yet quantum core rho_x is PURE => R_cr = chi!
    xi_mix = [[Fraction(3, 4), Fraction(0)], [Fraction(0), Fraction(1, 4)]]
    s4_pure_core = [mat_kron(psi0, xi_mix), mat_kron(psi1, xi_mix)]
    atlas_rows.append(_emit_cert("ki_4d_pure_core_mixed_redundant.cert.json", {
        "schema": "KOASHI-IMOTO-CERT-1.0",
        "mode": "ki_block_tensor_criterion_exact",
        "name": "mode2_4d_pure_core_mixed_redundant",
        "dim_W": 4,
        "priors": ["1/2", "1/2"],
        "signal_states": [_fmt_mat(s) for s in s4_pure_core],
        "sector_core_dims": [2],
        "sector_redundant_dims": [2],
        "sector_probs": [["1"], ["1"]],
        "core_states": [[_fmt_mat(psi0)], [_fmt_mat(psi1)]],
        "redundant_states": [_fmt_mat(xi_mix)],
        "claimed_criterion": "holevo_achievable_all_cores_pure",
        "claimed_core_linear_mix": "0",
    }))

    # 2c: 3D Qutrit Erasure with MIXED Non-Commuting Qubit Core (strict gap R_cr > chi!):
    rho_m0 = _bloch_to_real_dm(Fraction(2, 5), Fraction(4, 5))
    rho_m1 = _bloch_to_real_dm(Fraction(-2, 5), Fraction(4, 5))
    erasure_mixed = [
        mat_direct_sum([mat_scale(Fraction(1, 3), one1), mat_scale(Fraction(2, 3), rho_m0)]),
        mat_direct_sum([mat_scale(Fraction(1, 3), one1), mat_scale(Fraction(2, 3), rho_m1)]),
    ]
    r_lin_2c = Fraction(1, 2) * Fraction(2, 3) * (1 - mat_trace(mat_mul(rho_m0, rho_m0))) + \
               Fraction(1, 2) * Fraction(2, 3) * (1 - mat_trace(mat_mul(rho_m1, rho_m1)))
    atlas_rows.append(_emit_cert("ki_3d_mixed_noncomm_core_gap.cert.json", {
        "schema": "KOASHI-IMOTO-CERT-1.0",
        "mode": "ki_block_tensor_criterion_exact",
        "name": "mode2_3d_mixed_noncomm_core_gap",
        "dim_W": 3,
        "priors": ["1/2", "1/2"],
        "signal_states": [_fmt_mat(s) for s in erasure_mixed],
        "sector_core_dims": [1, 2],
        "sector_redundant_dims": [1, 1],
        "sector_probs": [["1/3", "2/3"], ["1/3", "2/3"]],
        "core_states": [[_fmt_mat(one1), _fmt_mat(rho_m0)], [_fmt_mat(one1), _fmt_mat(rho_m1)]],
        "redundant_states": [_fmt_mat(one1), _fmt_mat(one1)],
        "claimed_criterion": "strict_gap_mixed_core",
        "claimed_core_linear_mix": fmt_rat(r_lin_2c),
    }))

    # 2d: 6D Two-Sector KI Decomposition (2D pure core (x) 1D) (+) (2D mixed non-comm core (x) 2D redundant xi):
    s6_sigs = [
        mat_direct_sum([mat_scale(Fraction(1, 4), psi0), mat_scale(Fraction(3, 4), mat_kron(rho_m0, xi_mix))]),
        mat_direct_sum([mat_scale(Fraction(1, 4), psi1), mat_scale(Fraction(3, 4), mat_kron(rho_m1, xi_mix))]),
    ]
    r_lin_2d = Fraction(3, 4) * (1 - mat_trace(mat_mul(rho_m0, rho_m0)))
    atlas_rows.append(_emit_cert("ki_6d_two_sector_mixed_core_gap.cert.json", {
        "schema": "KOASHI-IMOTO-CERT-1.0",
        "mode": "ki_block_tensor_criterion_exact",
        "name": "mode2_6d_two_sector_mixed_core_gap",
        "dim_W": 6,
        "priors": ["1/2", "1/2"],
        "signal_states": [_fmt_mat(s) for s in s6_sigs],
        "sector_core_dims": [2, 2],
        "sector_redundant_dims": [1, 2],
        "sector_probs": [["1/4", "3/4"], ["1/4", "3/4"]],
        "core_states": [[_fmt_mat(psi0), _fmt_mat(rho_m0)], [_fmt_mat(psi1), _fmt_mat(rho_m1)]],
        "redundant_states": [_fmt_mat(one1), _fmt_mat(xi_mix)],
        "claimed_criterion": "strict_gap_mixed_core",
        "claimed_core_linear_mix": fmt_rat(r_lin_2d),
    }))

    # =========================================================================
    # MODE 3: 4 Pure-Atom Caratheodory Decomposition Certificates (Def. 2, Lemma 3)
    # =========================================================================
    # 3a: Off-center chord endpoint decomposition (Fig. 2c, U=2 pure endpoints |u>, |v|)
    #     |u> = (3, 1)/sqrt(10) -> Bloch (3/5, 0, 4/5); |v> = (-3, 1)/sqrt(10)? Wait: check Bloch of [3, 1]:
    #     P_u = [[9/10, 3/10], [3/10, 1/10]] -> rz = 9/10 - 1/10 = 4/5, rx = 6/10 = 3/5! EXACT!!
    #     P_v = [[9/10, -3/10], [-3/10, 1/10]] -> rz = 4/5, rx = -3/5! EXACT!!
    #     5/6 P_u + 1/6 P_v -> rx = (5/6 - 1/6)*(3/5) = 2/5, rz = 4/5 == rho_m0! EXACT!!
    v_u = [Fraction(3), Fraction(1)]
    v_v = [Fraction(3), Fraction(-1)]
    assert mat_add(mat_scale(Fraction(5, 6), pure_projector_exact(v_u)),
                   mat_scale(Fraction(1, 6), pure_projector_exact(v_v))) == rho_m0
    assert mat_add(mat_scale(Fraction(1, 6), pure_projector_exact(v_u)),
                   mat_scale(Fraction(5, 6), pure_projector_exact(v_v))) == rho_m1
    atlas_rows.append(_emit_cert("pure_atom_chord_endpoints_U2.cert.json", {
        "schema": "KOASHI-IMOTO-CERT-1.0",
        "mode": "pure_atom_caratheodory_decomp_exact",
        "name": "mode3_chord_endpoints_U2",
        "dim_W": 2,
        "priors": ["1/2", "1/2"],
        "signal_states": [_fmt_mat(rho_m0), _fmt_mat(rho_m1)],
        "num_flags_U": 2,
        "flag_probs": [["5/6", "1/6"], ["1/6", "5/6"]],
        "pure_atom_vectors": [
            [[fmt_rat(c) for c in v_u], [fmt_rat(c) for c in v_v]],
            [[fmt_rat(c) for c in v_u], [fmt_rat(c) for c in v_v]],
        ],
        "claimed_caratheodory_bound": 2 * 4 + 1,
    }))

    # 3b: Qutrit erasure pure-atom decomposition (U=2 flags: erasure vs signal)
    atlas_rows.append(_emit_cert("pure_atom_qutrit_erasure_U2.cert.json", {
        "schema": "KOASHI-IMOTO-CERT-1.0",
        "mode": "pure_atom_caratheodory_decomp_exact",
        "name": "mode3_qutrit_erasure_U2",
        "dim_W": 3,
        "priors": ["1/2", "1/2"],
        "signal_states": [_fmt_mat(s) for s in erasure_sigs],
        "num_flags_U": 2,
        "flag_probs": [["1/2", "1/2"], ["1/2", "1/2"]],
        "pure_atom_vectors": [
            [["1", "0", "0"], ["0", "1", "0"]],
            [["1", "0", "0"], ["0", "3", "4"]],
        ],
        "claimed_caratheodory_bound": 2 * 9 + 1,
    }))

    # 3c: 4D pure-core (x) mixed-redundant pure-atom decomposition (U=2 flags selecting eigenstate of xi)
    atlas_rows.append(_emit_cert("pure_atom_4d_redundant_U2.cert.json", {
        "schema": "KOASHI-IMOTO-CERT-1.0",
        "mode": "pure_atom_caratheodory_decomp_exact",
        "name": "mode3_4d_redundant_U2",
        "dim_W": 4,
        "priors": ["1/2", "1/2"],
        "signal_states": [_fmt_mat(s) for s in s4_pure_core],
        "num_flags_U": 2,
        "flag_probs": [["3/4", "1/4"], ["3/4", "1/4"]],
        "pure_atom_vectors": [
            [["1", "0", "0", "0"], ["0", "1", "0", "0"]],
            [["3", "0", "4", "0"], ["0", "3", "0", "4"]],
        ],
        "claimed_caratheodory_bound": 2 * 16 + 1,
    }))

    # 3d: 3-signal mixed qubit ensemble decomposed into 3 pure flags (U=3 <= 3*4+1 = 13)
    p_z = pure_projector_exact([Fraction(1), Fraction(0)])
    p_x = pure_projector_exact([Fraction(1), Fraction(1)])
    p_xm = pure_projector_exact([Fraction(1), Fraction(-1)])
    s3_mix = [
        mat_add(mat_scale(Fraction(2, 3), p_z), mat_scale(Fraction(1, 3), p_x)),
        mat_add(mat_scale(Fraction(2, 3), p_x), mat_scale(Fraction(1, 3), p_xm)),
        mat_add(mat_scale(Fraction(2, 3), p_xm), mat_scale(Fraction(1, 3), p_z)),
    ]
    atlas_rows.append(_emit_cert("pure_atom_3signal_qubit_U3.cert.json", {
        "schema": "KOASHI-IMOTO-CERT-1.0",
        "mode": "pure_atom_caratheodory_decomp_exact",
        "name": "mode3_3signal_qubit_U3",
        "dim_W": 2,
        "priors": ["1/3", "1/3", "1/3"],
        "signal_states": [_fmt_mat(s) for s in s3_mix],
        "num_flags_U": 3,
        "flag_probs": [["2/3", "1/3", "0"], ["0", "2/3", "1/3"], ["1/3", "0", "2/3"]],
        "pure_atom_vectors": [
            [["1", "0"], ["1", "1"], ["0", "0"]],
            [["0", "0"], ["1", "1"], ["1", "-1"]],
            [["1", "0"], ["0", "0"], ["1", "-1"]],
        ],
        "claimed_caratheodory_bound": 3 * 4 + 1,
    }))

    # =========================================================================
    # MODE 4: 4 Sector-Splitting & Partial-Additivity Certificates (Props 9 & 10)
    # =========================================================================
    m4_cases = [
        (
            "sector_split_erasure_blind_sep.cert.json",
            "mode4_erasure_blind_separation",
            [Fraction(1, 2), Fraction(1, 2)],
            [Fraction(0), Fraction(3, 4)],
            [Fraction(0), Fraction(3, 4)],
            Fraction(0), Fraction(1), Fraction(0),  # H(J|X)=1, r_mix=0 -> visible excess 0, blind excess 1!
            Fraction(3, 8), Fraction(3, 8), Fraction(1, 2), Fraction(7, 8),  # exact additivity!
        ),
        (
            "sector_split_two_sector_mixed.cert.json",
            "mode4_two_sector_mixed_excess",
            [Fraction(1, 4), Fraction(3, 4)],
            [Fraction(3, 5), Fraction(2, 5)],
            [Fraction(3, 5), Fraction(3, 10)],
            Fraction(1, 5), Fraction(3, 10), Fraction(1, 5),
            Fraction(3, 5), Fraction(3, 5), Fraction(2, 5), Fraction(1),
        ),
        (
            "sector_split_three_sector_nonadd.cert.json",
            "mode4_three_sector_nonadditive_sandwich",
            [Fraction(1, 5), Fraction(2, 5), Fraction(2, 5)],
            [Fraction(1, 2), Fraction(3, 4), Fraction(4, 5)],
            [Fraction(1, 2), Fraction(1, 2), Fraction(3, 5)],
            Fraction(1, 10), Fraction(1, 4), Fraction(1, 4),
            Fraction(3, 10), Fraction(2, 5), Fraction(2, 5), Fraction(3, 4),  # strict subadditivity inside Prop 9!
        ),
        (
            "sector_split_four_sector_pure_cores.cert.json",
            "mode4_four_sector_pure_cores_exact",
            [Fraction(1, 4), Fraction(1, 4), Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 2), Fraction(2, 3), Fraction(3, 4), Fraction(1, 3)],
            [Fraction(1, 2), Fraction(2, 3), Fraction(3, 4), Fraction(1, 3)],
            Fraction(1, 2), Fraction(3, 2), Fraction(0),
            Fraction(1, 2), Fraction(1, 2), Fraction(7, 10), Fraction(6, 5),
        ),
    ]
    for fname, cname, qs, fs, cs, ixj, hjx, rmix, c1, f1, f2, f12 in m4_cases:
        tot_F = ixj + sum((q * f for q, f in zip(qs, fs)), Fraction(0, 1))
        tot_chi = ixj + sum((q * c for q, c in zip(qs, cs)), Fraction(0, 1))
        vis_ex = tot_F - tot_chi
        blind_ex = hjx + rmix
        atlas_rows.append(_emit_cert(fname, {
            "schema": "KOASHI-IMOTO-CERT-1.0",
            "mode": "sector_splitting_and_partial_additivity_exact",
            "name": cname,
            "sector_weights_q": [fmt_rat(x) for x in qs],
            "sector_F": [fmt_rat(x) for x in fs],
            "sector_chi": [fmt_rat(x) for x in cs],
            "I_X_J": fmt_rat(ixj),
            "H_J_given_X": fmt_rat(hjx),
            "r_mix": fmt_rat(rmix),
            "claimed_total_F": fmt_rat(tot_F),
            "claimed_total_chi": fmt_rat(tot_chi),
            "claimed_visible_excess": fmt_rat(vis_ex),
            "claimed_blind_excess": fmt_rat(blind_ex),
            "prod_chi1": fmt_rat(c1),
            "prod_F1": fmt_rat(f1),
            "prod_F2": fmt_rat(f2),
            "prod_F12": fmt_rat(f12),
        }))

    (RESULTS / "exp1_certified_atlas.json").write_text(
        json.dumps({"experiment": "exp1_certified_atlas", "count": len(atlas_rows), "rows": atlas_rows}, indent=2) + "\n",
        encoding="utf-8",
    )

    # =========================================================================
    # EXP 2: Qubit Off-Center Chord Sweep (Fig. 2c) — Exact Purity Defect,
    #        Holevo chi, Endpoint Simulation Cost C = I(X:U), and Schumacher S(bar_omega)
    # =========================================================================
    # Pure endpoints u = (3/5, 0, 4/5), v = (-3/5, 0, 4/5) (overlap |<u|v>| = 4/5 -> |<u|v>|^2 = 16/25?
    # Wait: (1 + u.v)/2 = (1 + (-9/25 + 16/25))/2 = (1 + 7/25)/2 = 16/25 -> |<u|v>| = 4/5!
    # For t0 in {0, 1/12, 1/6, 1/4, 1/3, 5/12, 1/2}: r0 = ((1-2*t0)*3/5, 0, 4/5), r1 = (-(1-2*t0)*3/5, 0, 4/5)
    exp2_rows = []
    for num, den in [(0, 1), (1, 12), (1, 6), (1, 4), (1, 3), (5, 12), (1, 2)]:
        t0 = Fraction(num, den)
        s = Fraction(1, 1) - 2 * t0
        rx0 = s * Fraction(3, 5)
        rz0 = Fraction(4, 5)
        r_lin = (1 - (rx0 * rx0 + rz0 * rz0)) / 2
        comm_sq = (2 * rx0 * rz0) ** 2
        # Floating-point reference entropies for Fig 1(a) plot
        s_avg = _qubit_vn_entropy(0.0, 0.0, float(rz0))
        s_sig = _qubit_vn_entropy(float(rx0), 0.0, float(rz0))
        chi = s_avg - s_sig
        c_end = 1.0 - _bin_entropy(float(t0))
        exp2_rows.append({
            "t0_exact": fmt_rat(t0),
            "t0": round(float(t0), 5),
            "exact_r_lin": fmt_rat(r_lin),
            "r_lin": round(float(r_lin), 6),
            "exact_comm_sq": fmt_rat(comm_sq),
            "holevo_chi": round(chi, 6),
            "endpoint_C": round(c_end, 6),
            "schumacher_S": round(s_avg, 6),
            "best_upper_gap": round(min(c_end, s_avg) - chi, 6),
        })
    (RESULTS / "exp2_chord_entropy_gap.json").write_text(
        json.dumps({"experiment": "exp2_chord_entropy_gap", "rows": exp2_rows}, indent=2) + "\n",
        encoding="utf-8",
    )

    # =========================================================================
    # EXP 3: Qutrit Erasure Channel Sweep (Erasure prob e in [0, 1]) —
    #        Blind Gap R_blind - chi = h2(e) vs Visible Gap R_cr - chi = 0!
    # =========================================================================
    exp3_rows = []
    # Surviving pure non-orthogonal states |psi0>, |psi1> with overlap 3/5 -> S(bar_rho) = h2((1+3/5)/2) = h2(4/5)
    s_surv = _bin_entropy(0.8)
    for k in range(0, 9):
        e = Fraction(k, 8)
        ef = float(e)
        chi_e = (1.0 - ef) * s_surv
        hjx = _bin_entropy(ef)
        r_blind = chi_e + hjx
        exp3_rows.append({
            "erasure_e_exact": fmt_rat(e),
            "erasure_e": round(ef, 4),
            "holevo_chi": round(chi_e, 6),
            "visible_Rcr": round(chi_e, 6),
            "blind_Rblind": round(r_blind, 6),
            "blind_gap": round(hjx, 6),
            "visible_gap": 0.0,
        })
    (RESULTS / "exp3_erasure_blind_vs_visible.json").write_text(
        json.dumps({"experiment": "exp3_erasure_blind_vs_visible", "rows": exp3_rows}, indent=2) + "\n",
        encoding="utf-8",
    )

    # =========================================================================
    # EXP 4: Mixed-Core Redundant-Factor Scaling (KI invariance under dim N_j)
    # =========================================================================
    exp4_rows = []
    for dN in (1, 2, 3, 4):
        dW = 2 * dN
        carat = 2 * dW * dW + 1
        exp4_rows.append({
            "dim_N": dN,
            "dim_W": dW,
            "caratheodory_bound": carat,
            "core_r_lin": "0",
        })
    (RESULTS / "exp4_caratheodory_scaling.json").write_text(
        json.dumps({"experiment": "exp4_caratheodory_scaling", "rows": exp4_rows}, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[PASS] Generated & verified {len(atlas_rows)}/16 exact-Q certificates and 4 experiment JSONs.")
    for r in atlas_rows:
        print(f"  {r['file']:44s} {r['verify_ms']:6.3f} ms  {r['summary']}")


if __name__ == "__main__":
    main()
