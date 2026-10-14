"""Lazy SNES setup, independent radial-force moments and owned RHS lifetime."""
import os
from contextlib import ExitStack
import numpy as np
import sympy as sp
from scipy.integrate import dblquad
from petsc4py import PETSc
import underworld3 as uw
from underworld3.postprocessing.spherical_interface_load import install_linear_spherical_stokes_sheet_load


def test_sheet_adapter_sets_up_lazy_snes_and_releases_rhs():
    root_pid=uw.mpi.comm.bcast(os.getpid() if uw.mpi.rank==0 else None,root=0)
    mesh=uw.meshing.UnstructuredSimplexBox(minCoords=(1.,1.,1.),maxCoords=(2.,2.,2.),
        cellSize=.5,regular=True,qdegree=4,filename=f'/tmp/uw-sheet-setup-{root_pid}.msh')
    u=uw.discretisation.MeshVariable('SheetSetupU',mesh,3,degree=2,vtype=uw.VarType.VECTOR)
    p=uw.discretisation.MeshVariable('SheetSetupP',mesh,1,degree=1)
    stokes=uw.systems.Stokes(mesh,velocityField=u,pressureField=p)
    stokes.constitutive_model=uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0=1.
    stokes.bodyforce=sp.zeros(1,3)
    stokes.petsc_use_nullspace=False
    stokes.petsc_use_pressure_nullspace=False
    # Top is z=2 for this box. No solve/nullspace setup can hide lazy matrices.
    arguments=dict(stokes=stokes,interface='Top',harmonic_degree=1,density_coefficient=1.)
    rhs=install_linear_spherical_stokes_sheet_load(**arguments)
    assert bool(stokes.snes.getJacobian()[0]) and rhs.norm()>0.
    iset,subdm=stokes._subdict['velocity']
    sub=rhs.getSubVector(iset)
    try:
        with ExitStack() as cleanup:
            local=u.vec.duplicate();cleanup.callback(local.destroy)
            global_=subdm.createGlobalVec();cleanup.callback(global_.destroy)
            for axis in range(3):
                local.array[:]=0.;local.array.reshape(-1,3)[:,axis]=1.
                global_.set(0.)
                subdm.localToGlobal(local,global_,addv=PETSc.InsertMode.INSERT_VALUES)
                actual=float(sub.dot(global_))
                # Independent P_1(cos(theta))=z/r times the radial unit vector.
                expected=dblquad(lambda y,x: [x,y,2.][axis]*2./(x*x+y*y+4),
                    1.,2.,lambda x:1.,lambda x:2.,epsabs=1e-12,epsrel=1e-12)[0]
                np.testing.assert_allclose(actual,expected,rtol=1e-11,atol=1e-13)
    finally:rhs.restoreSubVector(iset,sub)
    second=install_linear_spherical_stokes_sheet_load(**(arguments|{'density_coefficient':-.7}))
    assert not bool(rhs) and second.norm()>0.
    # A full DM rebuild must also release the callback's old layout/vector.
    stokes.is_setup=False
    stokes._needs_dm_rebuild=True
    stokes._build(False,False,None)
    assert not bool(second) and stokes._linear_spherical_sheet_rhs is None
    third=install_linear_spherical_stokes_sheet_load(**arguments)
    stokes._reset()
    assert not bool(third)
