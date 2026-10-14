"""Public responses against the preserved independent radial reference oracle."""
from pathlib import Path
import runpy
import numpy as np
import pytest
from underworld3.postprocessing import geoid

# This preserved oracle covers ell>=1. Degree-zero kernels retain their
# separate continuity/jump/continuation checks in test_1075.
REFERENCE=runpy.run_path(str(Path(__file__).parent/'oracles/spherical_gravity.py'))
pytestmark=[pytest.mark.level_1,pytest.mark.tier_a]

@pytest.mark.parametrize('ell',[1,2,4,8])
@pytest.mark.parametrize('load',[0.,-.37,1.])
def test_normalized_responses_match_independent_oracle(ell,load):
    args=dict(radius_inner=.55,radius_outer=1.,harmonic_degree=ell,
        surface_topography_coefficient=.4192,cmb_topography_coefficient=.7706,
        internal_load_radius=.775,internal_load_coefficient=load)
    expected=REFERENCE['spherical_shell_geoid_response'](**args)
    actual=geoid.spherical_shell_geoid_response(**args)
    np.testing.assert_allclose([actual.surface_geoid,actual.cmb_geoid],
                               [expected.surface_geoid,expected.cmb_geoid],rtol=2e-14)
    args.update(surface_density_contrast=3300.,cmb_density_contrast=5400.,
                planet_radius=6370000.,gravity=9.8,gravitational_constant=6.67e-11)
    expected=REFERENCE['spherical_shell_self_gravity_response'](**args)
    actual=geoid.spherical_shell_self_gravity_response(**args)
    for field in ('surface_topography','cmb_topography','surface_geoid','cmb_geoid','q_surface','q_cmb'):
        np.testing.assert_allclose(getattr(actual,field),getattr(expected,field),rtol=3e-14,atol=1e-15)
