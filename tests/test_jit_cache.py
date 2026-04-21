# Tests for the C-source-hash JIT cache (issues #121, #123).
#
# The cache key is a SHA-256 hash of the canonicalised C source plus an ABI
# salt. This makes state equality *provable*:
#   same C source ⇒ same hash ⇒ same .so ⇒ same solver behaviour.
#
# Gate 1 — constant ↔ symbolic ↔ constant transition
#   Verifies that a Poisson solve with diff=constant, then diff=1+u, then
#   diff=constant returns the *exact* state-1 result on state 3. With the
#   old sympy-structural cache, state 3 either cache-missed (recompile) or
#   returned a contaminated state-2 result.
#
# Gate 2 — value cycle
#   Toggling a constant's value between two fixed numbers should produce
#   exactly one cache entry (the C source is value-independent).

import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.utilities._jitextension import _ext_dict

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


@pytest.fixture(autouse=True)
def reset_model_state():
    uw.reset_default_model()
    uw.use_strict_units(False)
    uw.use_nondimensional_scaling(False)
    yield
    uw.reset_default_model()
    uw.use_strict_units(False)
    uw.use_nondimensional_scaling(False)


def _t_max(u_field, sample_points):
    u = uw.function.evaluate(u_field.sym[0], sample_points, rbf=False).squeeze()
    return float(np.max(np.abs(u)))


def test_c_s_c_transition_returns_state_1_result_on_state_3():
    """diff=0.5 → 1+u → 0.5. The third solve must match the first exactly.

    This is the bug that motivated the C-source-hash architecture: with
    the old sympy-structural cache, state 3 returned a contaminated
    state-2 numerical result even though it was structurally identical
    to state 1.
    """

    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.15)

    u = uw.discretisation.MeshVariable("u_csc", mesh, 1, degree=2)

    K = uw.expression("K_csc", 0.5)

    poisson = uw.systems.Poisson(mesh, u_Field=u)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = K
    poisson.f = 1.0

    poisson.add_dirichlet_bc(0.0, "Top")
    poisson.add_dirichlet_bc(0.0, "Bottom")

    sample_y = np.linspace(0.05, 0.95, 15)
    sample_x = np.full_like(sample_y, 0.5)
    sample_points = np.column_stack([sample_x, sample_y])

    # State 1: K = 0.5 (constant)
    poisson.solve()
    assert poisson.snes.getConvergedReason() > 0
    t_max_1 = _t_max(u, sample_points)

    # State 2: K = 1 + u (symbolic, nonlinear)
    poisson.constitutive_model.Parameters.diffusivity = 1.0 + u.sym[0]
    poisson.solve()
    assert poisson.snes.getConvergedReason() > 0
    t_max_2 = _t_max(u, sample_points)

    # State 3: K = 0.5 again (constant, same as state 1)
    poisson.constitutive_model.Parameters.diffusivity = K
    poisson.solve()
    assert poisson.snes.getConvergedReason() > 0
    t_max_3 = _t_max(u, sample_points)

    assert t_max_2 != pytest.approx(t_max_1, abs=1e-3), (
        f"State 2 (K=1+u) result {t_max_2:.6f} is suspiciously close to state 1 "
        f"(K=0.5) result {t_max_1:.6f}; the symbolic branch is not being taken."
    )

    assert t_max_3 == pytest.approx(t_max_1, abs=1e-6), (
        f"State 3 must equal state 1 (same K=0.5), got "
        f"T_max_1={t_max_1:.8f}, T_max_3={t_max_3:.8f}. "
        f"C-source hash keying should guarantee identical behaviour."
    )

    del poisson


def test_value_cycle_produces_single_cache_entry():
    """Toggle a constant between two values 20×; C source is value-independent."""

    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.2)

    u = uw.discretisation.MeshVariable("u_cycle", mesh, 1, degree=2)

    K = uw.expression("K_cycle", 1.0)

    poisson = uw.systems.Poisson(mesh, u_Field=u)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = K
    poisson.f = -2.0

    poisson.add_dirichlet_bc(0.0, "Bottom")
    poisson.add_dirichlet_bc(1.0, "Top")

    # Warm the cache with the first solve
    poisson.solve()
    n_entries_after_first = len(_ext_dict)

    # Toggle between two values 20 times; each solve must cache-hit
    values = [0.5, 1.0]
    for i in range(20):
        K.sym = values[i % 2]
        poisson.solve()
        assert poisson.snes.getConvergedReason() > 0

    n_entries_after_cycle = len(_ext_dict)

    assert n_entries_after_cycle == n_entries_after_first, (
        f"Toggling a constant's value caused {n_entries_after_cycle - n_entries_after_first} "
        f"new JIT compile(s). With C-source-hash keying, the source is value-independent."
    )

    del poisson


def test_cache_key_is_deterministic_hex_string():
    """The cache_key on _GextResult is a 16-char hex string (source hash)."""

    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.25)

    u = uw.discretisation.MeshVariable("u_k", mesh, 1, degree=2)

    poisson = uw.systems.Poisson(mesh, u_Field=u)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = 0.0
    poisson.add_dirichlet_bc(1.0, "Bottom")
    poisson.add_dirichlet_bc(0.0, "Top")
    poisson.solve()

    key = poisson._current_jit_cache_key
    assert isinstance(key, str), f"cache_key should be str, got {type(key).__name__}"
    assert len(key) == 16, f"cache_key should be 16 hex chars, got {len(key)}: {key!r}"
    assert all(c in "0123456789abcdef" for c in key), (
        f"cache_key should be lowercase hex, got {key!r}"
    )

    del poisson
