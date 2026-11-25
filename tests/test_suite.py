"""Gate-2 verification, direct Hermitian Bloch/PSD cross-checks, and 54-attack live red-team suite for Paper 24.
Author: Lluis Eriksson
"""
from __future__ import annotations

import copy
import json
import pathlib
import sys
from decimal import Decimal
from fractions import Fraction

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ki_certifier import KoashiImotoCertificateError, verify_koashi_imoto_certificate_exact
from src.ki_exact import (
    bloch_cross,
    bloch_norm_sq,
    commutant_dimension_exact,
    mat_commutator,
    mat_is_psd_exact,
    mat_is_zero,
    mat_mul,
    mat_trace,
    pure_projector_exact,
)

CERTS_DIR = ROOT / "results" / "certificates"


def _cmul_q(a, b):
    """Exact Gaussian-rational multiplication (re, im) in Q[i]."""
    return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


def _cadd_q(a, b):
    return (a[0] + b[0], a[1] + b[1])


def _csub_q(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _bloch_hermitian_2x2_exact(rx: Fraction, ry: Fraction, rz: Fraction):
    """Exact 2x2 Hermitian density matrix (I + rx*X + ry*Y + rz*Z)/2 over Q[i] (zero floats)."""
    h = Fraction(1, 2)
    z = Fraction(0, 1)
    return [
        [(h * (1 + rz), z), (h * rx, -h * ry)],
        [(h * rx, h * ry), (h * (1 - rz), z)],
    ]


def test_direct_hermitian_purity_commutator_and_psd_cross_checks() -> None:
    """Independently cross-check Bloch purity/commutator over exact Q[i], exact LDL^T PSD, and Wedderburn commutant dim."""
    r0 = (Fraction(2, 5), Fraction(0), Fraction(4, 5))
    r1 = (Fraction(-2, 5), Fraction(0), Fraction(4, 5))
    M0 = _bloch_hermitian_2x2_exact(*r0)
    M1 = _bloch_hermitian_2x2_exact(*r1)

    def matmul_qi(A, B):
        z = (Fraction(0, 1), Fraction(0, 1))
        out = [[z, z], [z, z]]
        for i in range(2):
            for j in range(2):
                acc = z
                for k in range(2):
                    acc = _cadd_q(acc, _cmul_q(A[i][k], B[k][j]))
                out[i][j] = acc
        return out

    # Exact Q[i] purity Tr(M0^2) == (1 + |r0|^2)/2 == 9/10
    M0sq = matmul_qi(M0, M0)
    tr_sq = _cadd_q(M0sq[0][0], M0sq[1][1])
    assert tr_sq == ((1 + bloch_norm_sq(r0)) / 2, Fraction(0, 1))

    # Exact Q[i] Frobenius norm squared of commutator [M0, M1]: Tr([M0,M1]^dag [M0,M1]) == 1/2 |r0 x r1|^2
    M01, M10 = matmul_qi(M0, M1), matmul_qi(M1, M0)
    C = [[_csub_q(M01[i][j], M10[i][j]) for j in range(2)] for i in range(2)]
    frob_sq = sum((C[i][j][0] ** 2 + C[i][j][1] ** 2 for i in range(2) for j in range(2)), Fraction(0, 1))
    assert frob_sq == bloch_norm_sq(bloch_cross(r0, r1)) / 2

    # Independent PSD and Wedderburn commutant dimension checks
    P = pure_projector_exact([Fraction(3), Fraction(4)])
    assert mat_is_psd_exact(P)
    assert mat_mul(P, P) == P
    assert not mat_is_psd_exact([[Fraction(0), Fraction(1)], [Fraction(1), Fraction(0)]])
    assert not mat_is_psd_exact([[Fraction(1), Fraction(2)], [Fraction(2), Fraction(1)]])
    assert mat_is_psd_exact([[Fraction(0), Fraction(0)], [Fraction(0), Fraction(1)]])

    # Reducible 3x3 direct sum has commutant_dim = 2 (!= 1), ruling out fake mixed core
    red0 = [[Fraction(1, 2), Fraction(0), Fraction(0)],
            [Fraction(0), Fraction(0), Fraction(0)],
            [Fraction(0), Fraction(0), Fraction(1, 2)]]
    red1 = [[Fraction(1, 4), Fraction(1, 4), Fraction(0)],
            [Fraction(1, 4), Fraction(1, 4), Fraction(0)],
            [Fraction(0), Fraction(0), Fraction(1, 2)]]
    assert commutant_dimension_exact([red0, red1]) == 2


def test_all_16_certificates_verify_cleanly() -> None:
    files = sorted(CERTS_DIR.glob("*.cert.json"))
    assert len(files) == 16
    for f in files:
        res = verify_koashi_imoto_certificate_exact(json.loads(f.read_text(encoding="utf-8")))
        assert res["verified"] is True


def test_54_live_adversarial_red_team_attacks_rejected() -> None:
    c1 = json.loads((CERTS_DIR / "qubit_fig2c_mixed_noncommuting.cert.json").read_text(encoding="utf-8"))
    c2 = json.loads((CERTS_DIR / "ki_6d_two_sector_mixed_core_gap.cert.json").read_text(encoding="utf-8"))
    c3 = json.loads((CERTS_DIR / "pure_atom_chord_endpoints_U2.cert.json").read_text(encoding="utf-8"))
    c4 = json.loads((CERTS_DIR / "sector_split_two_sector_mixed.cert.json").read_text(encoding="utf-8"))

    attacks = []

    def add(name, base, fn):
        cand = copy.deepcopy(base)
        fn(cand)
        attacks.append((name, cand))

    # Group A: Schema, Types & Canonical Rationals (16 attacks)
    attacks.append(("01_root_not_dict", ["bad"]))
    add("02_wrong_schema", c1, lambda x: x.__setitem__("schema", "KOASHI-IMOTO-CERT-9.9"))
    add("03_unknown_mode", c1, lambda x: x.__setitem__("mode", "bogus"))
    add("04_extra_key", c1, lambda x: x.__setitem__("evil", 1))
    add("05_missing_key", c1, lambda x: x.pop("priors"))
    add("06_float_injection", c1, lambda x: x["priors"].__setitem__(0, 0.5))
    add("07_bool_injection", c2, lambda x: x.__setitem__("dim_W", True))
    add("08_decimal_injection", c4, lambda x: x.__setitem__("I_X_J", Decimal("0.2")))
    add("09_complex_injection", c1, lambda x: x["priors"].__setitem__(0, 1 + 0j))
    add("10_tuple_injection", c1, lambda x: x.__setitem__("priors", tuple(x["priors"])))
    add("11_non_str_key", c1, lambda x: x.__setitem__(99, "bad"))
    add("12_unreduced_frac", c1, lambda x: x["priors"].__setitem__(0, "2/4"))
    add("13_denom_one_frac", c3, lambda x: x["pure_atom_vectors"][0][0].__setitem__(0, "3/1"))
    add("14_neg_zero", c1, lambda x: x["bloch_vectors"][0].__setitem__(1, "-0"))
    add("15_non_str_name", c1, lambda x: x.__setitem__("name", 123))
    add("16_empty_name", c1, lambda x: x.__setitem__("name", "   "))

    # Group B: Mode 1 Qubit Bloch Dichotomy Forgeries (10 attacks)
    add("17_priors_not_sum_1", c1, lambda x: x["priors"].__setitem__(0, "1/3"))
    add("18_zero_prior", c1, lambda x: x.__setitem__("priors", ["0", "1"]))
    add("19_bloch_outside_ball", c1, lambda x: x["bloch_vectors"].__setitem__(0, ["1", "1", "0"]))
    add("20_bloch_wrong_dim", c1, lambda x: x["bloch_vectors"].__setitem__(0, ["1", "0"]))
    add("21_wrong_regime_pure", c1, lambda x: x.__setitem__("claimed_regime", "holevo_achievable_pure"))
    add("22_wrong_regime_comm", c1, lambda x: x.__setitem__("claimed_regime", "holevo_achievable_commuting"))
    add("23_forged_r_lin", c1, lambda x: x.__setitem__("claimed_linear_mix_defect", "0"))
    add("24_forged_comm_sq", c1, lambda x: x.__setitem__("claimed_max_commutator_sq", "0"))
    add("25_single_signal_rejected", c1, lambda x: (x.__setitem__("priors", ["1"]), x.__setitem__("bloch_vectors", [["0", "0", "1"]])))
    add("26_mismatched_priors_vecs", c1, lambda x: x["bloch_vectors"].pop())

    # Group C: Mode 2 KI Block-Tensor Decomposition Forgeries (11 attacks)
    add("27_dim_W_below_2", c2, lambda x: x.__setitem__("dim_W", 1))
    add("28_sector_dim_sum_mismatch", c2, lambda x: x["sector_core_dims"].__setitem__(0, 1))
    add("29_non_psd_signal", c2, lambda x: x["signal_states"][0][0].__setitem__(0, "-1"))
    add("30_non_symmetric_signal", c2, lambda x: x["signal_states"][0][0].__setitem__(1, "1/10"))
    add("31_wrong_trace_redundant", c2, lambda x: x["redundant_states"][1][0].__setitem__(0, "1/2"))
    add("32_sector_probs_not_sum_1", c2, lambda x: x["sector_probs"][0].__setitem__(0, "1/2"))
    add("33_tampered_core_breaks_recon", c2, lambda x: x["core_states"][0][0][0].__setitem__(0, "1/2"))
    add("34_forged_criterion_pure", c2, lambda x: x.__setitem__("claimed_criterion", "holevo_achievable_all_cores_pure"))
    add("35_forged_core_r_lin", c2, lambda x: x.__setitem__("claimed_core_linear_mix", "0"))
    add("36_non_maximal_commuting_mixed_core", c2, lambda x: (
        x.__setitem__("dim_W", 2),
        x.__setitem__("signal_states", [[["7/10", "0"], ["0", "3/10"]], [["3/10", "0"], ["0", "7/10"]]]),
        x.__setitem__("sector_core_dims", [2]),
        x.__setitem__("sector_redundant_dims", [1]),
        x.__setitem__("sector_probs", [["1"], ["1"]]),
        x.__setitem__("core_states", [[[["7/10", "0"], ["0", "3/10"]]], [[["3/10", "0"], ["0", "7/10"]]]]),
        x.__setitem__("redundant_states", [[["1"]]]),
        x.__setitem__("claimed_criterion", "strict_gap_mixed_core"),
        x.__setitem__("claimed_core_linear_mix", "21/50"),
    ))
    add("37_dim_W_above_8", c2, lambda x: x.__setitem__("dim_W", 9))

    # Group D: Mode 3 Pure-Atom Caratheodory Decomposition Forgeries (8 attacks)
    add("38_wrong_caratheodory_bound", c3, lambda x: x.__setitem__("claimed_caratheodory_bound", 8))
    add("39_U_exceeds_caratheodory", c3, lambda x: x.__setitem__("num_flags_U", 10))
    add("40_U_zero", c3, lambda x: x.__setitem__("num_flags_U", 0))
    add("41_flag_probs_not_sum_1", c3, lambda x: x["flag_probs"][0].__setitem__(0, "1/2"))
    add("42_negative_flag_prob", c3, lambda x: x["flag_probs"].__setitem__(0, ["7/6", "-1/6"]))
    add("43_zero_occupied_atom_vec", c3, lambda x: x["pure_atom_vectors"][0].__setitem__(0, ["0", "0"]))
    add("44_wrong_atom_vec_breaks_recon", c3, lambda x: x["pure_atom_vectors"][0].__setitem__(0, ["1", "0"]))
    add("45_wrong_atom_vec_len", c3, lambda x: x["pure_atom_vectors"][0].__setitem__(0, ["1", "0", "0"]))

    # Group E: Mode 4 Sector-Splitting & Partial-Additivity Forgeries (9 attacks)
    add("46_sector_weights_not_sum_1", c4, lambda x: x["sector_weights_q"].__setitem__(0, "1/2"))
    add("47_sector_F_below_chi", c4, lambda x: x["sector_F"].__setitem__(1, "1/10"))
    add("48_vis_excess_exceeds_rmix", c4, lambda x: x.__setitem__("r_mix", "1/100"))
    add("49_forged_total_F", c4, lambda x: x.__setitem__("claimed_total_F", "1"))
    add("50_forged_visible_excess", c4, lambda x: x.__setitem__("claimed_visible_excess", "0"))
    add("51_forged_blind_excess", c4, lambda x: x.__setitem__("claimed_blind_excess", "0"))
    add("52_prop9_below_chi1_plus_F2", c4, lambda x: x.__setitem__("prod_F12", "1/2"))
    add("53_prop9_above_F1_plus_F2", c4, lambda x: x.__setitem__("prod_F12", "2"))
    add("54_pure_core_violates_exact_additivity", c4, lambda x: (
        x.__setitem__("prod_chi1", "1/2"),
        x.__setitem__("prod_F1", "1/2"),
        x.__setitem__("prod_F2", "1/2"),
        x.__setitem__("prod_F12", "9/10"),
    ))

    # Group F: Mode 2 Wedderburn/Schur Non-Maximal Irreducibility Exploits (3 attacks -> 57 total!)
    # 55: 3x3 reducible direct sum with non-commuting blocks (commutant dim = 2)
    red3_0 = [["1/2", "0", "0"], ["0", "0", "0"], ["0", "0", "1/2"]]
    red3_1 = [["1/4", "1/4", "0"], ["1/4", "1/4", "0"], ["0", "0", "1/2"]]
    add("55_reducible_3x3_direct_sum_fake_gap", c2, lambda x: (
        x.__setitem__("dim_W", 3),
        x.__setitem__("signal_states", [red3_0, red3_1]),
        x.__setitem__("sector_core_dims", [3]),
        x.__setitem__("sector_redundant_dims", [1]),
        x.__setitem__("sector_probs", [["1"], ["1"]]),
        x.__setitem__("core_states", [[red3_0], [red3_1]]),
        x.__setitem__("redundant_states", [[["1"]]]),
        x.__setitem__("claimed_criterion", "strict_gap_mixed_core"),
        x.__setitem__("claimed_core_linear_mix", "1/2"),
    ))
    # 56: 4x4 hidden redundant factor (|0><0| x xi, |+><+| x xi) (commutant dim = 4)
    hid4_0 = [["3/4", "0", "0", "0"], ["0", "1/4", "0", "0"], ["0", "0", "0", "0"], ["0", "0", "0", "0"]]
    hid4_1 = [["3/8", "0", "3/8", "0"], ["0", "1/8", "0", "1/8"], ["3/8", "0", "3/8", "0"], ["0", "1/8", "0", "1/8"]]
    add("56_hidden_4x4_redundant_factor_fake_gap", c2, lambda x: (
        x.__setitem__("dim_W", 4),
        x.__setitem__("signal_states", [hid4_0, hid4_1]),
        x.__setitem__("sector_core_dims", [4]),
        x.__setitem__("sector_redundant_dims", [1]),
        x.__setitem__("sector_probs", [["1"], ["1"]]),
        x.__setitem__("core_states", [[hid4_0], [hid4_1]]),
        x.__setitem__("redundant_states", [[["1"]]]),
        x.__setitem__("claimed_criterion", "strict_gap_mixed_core"),
        x.__setitem__("claimed_core_linear_mix", "3/8"),
    ))
    # 57: Multi-sector partially commuting mixed core (sector 0 commuting mixed, sector 1 non-commuting mixed)
    cm0 = [["7/10", "0"], ["0", "3/10"]]
    cm1 = [["3/10", "0"], ["0", "7/10"]]
    nc0 = [["9/10", "1/5"], ["1/5", "1/10"]]
    nc1 = [["9/10", "-1/5"], ["-1/5", "1/10"]]
    ms0 = [["7/20", "0", "0", "0"], ["0", "3/20", "0", "0"], ["0", "0", "9/20", "1/10"], ["0", "0", "1/10", "1/20"]]
    ms1 = [["3/20", "0", "0", "0"], ["0", "7/20", "0", "0"], ["0", "0", "9/20", "-1/10"], ["0", "0", "-1/10", "1/20"]]
    add("57_partial_commuting_multisector_fake_mix", c2, lambda x: (
        x.__setitem__("dim_W", 4),
        x.__setitem__("signal_states", [ms0, ms1]),
        x.__setitem__("sector_core_dims", [2, 2]),
        x.__setitem__("sector_redundant_dims", [1, 1]),
        x.__setitem__("sector_probs", [["1/2", "1/2"], ["1/2", "1/2"]]),
        x.__setitem__("core_states", [[cm0, nc0], [cm1, nc1]]),
        x.__setitem__("redundant_states", [[["1"]], [["1"]]]),
        x.__setitem__("claimed_criterion", "strict_gap_mixed_core"),
        x.__setitem__("claimed_core_linear_mix", "13/50"),
    ))

    assert len(attacks) == 57
    for name, cand in attacks:
        with pytest.raises(KoashiImotoCertificateError):
            verify_koashi_imoto_certificate_exact(cand)
