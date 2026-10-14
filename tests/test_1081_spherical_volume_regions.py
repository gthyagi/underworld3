"""Labelled DG0 source support: exact cell moments, overlap and empty ranks."""
from contextlib import ExitStack
import os
import numpy as np
import pytest
import sympy as sp
from petsc4py import PETSc
import underworld3 as uw
from underworld3.postprocessing.spherical_potential import _volume_region_indicator

pytestmark=[pytest.mark.level_1,pytest.mark.tier_a]


def test_region_source_exact_constant_and_linear_moments_with_empty_ranks():
    root_pid=uw.mpi.comm.bcast(os.getpid() if uw.mpi.rank==0 else None,root=0)
    mesh=uw.meshing.UnstructuredSimplexBox(minCoords=(0.,0.,0.),maxCoords=(1.,1.,1.),
        cellSize=.5,regular=True,qdegree=4,filename=f'/tmp/uw-region-test-{root_pid}.msh')
    comm=mesh.dm.getComm().tompi4py()
    start,end=mesh.dm.getHeightStratum(0)
    vstart,vend=mesh.dm.getDepthStratum(0)
    coordinate_section=mesh.dm.getCoordinateSection()
    coordinates=np.asarray(mesh.dm.getCoordinatesLocal().array).reshape(-1,3)
    ilocal=None
    if comm.size>1:_,ilocal,_=mesh.dm.getPointSF().getGraph()
    ghosts=set() if ilocal is None else set(map(int,ilocal))
    cells={}
    for point in range(start,end):
        vertices=[coordinates[coordinate_section.getOffset(int(p))//3]
                  for p in mesh.dm.getTransitiveClosure(point)[0] if vstart<=int(p)<vend]
        assert len(vertices)==4
        cells[point]=tuple(sorted(tuple(row) for row in vertices))
    owned=[key for point,key in cells.items() if point not in ghosts]
    candidates=comm.allgather(min(owned) if owned else None)
    chosen=min(key for key in candidates if key is not None)
    # The two labels select the same cell: their union must not double density.
    for name,value in [('VolumeA',101),('VolumeB',102)]:
        mesh.dm.createLabel(name)
        label=mesh.dm.getLabel(name)
        for point,key in cells.items():
            if key==chosen:label.setValue(point,value)
    indicator,report=_volume_region_indicator(mesh,'SingleCellSupport',{'VolumeA':101,'VolumeB':102})
    assert report['global_owned_cells']==1
    assert report['cells_by_region']=={'VolumeA':1,'VolumeB':1}
    if comm.size>1:assert report['owned_cells_per_rank'].count(0)==comm.size-1
    field=uw.discretisation.MeshVariable('SourceMomentTest',mesh,1,degree=2)
    poisson=uw.systems.Poisson(mesh,u_Field=field)
    poisson.constitutive_model=uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity=1.
    poisson.f=indicator.sym[0]
    poisson._build(False,False,None)
    mesh.update_lvec();poisson.dm.setAuxiliaryVec(mesh.lvec,None)
    poisson._update_constants(record=True);poisson.snes.setUp()
    vertices=np.asarray(chosen)
    volume=abs(np.linalg.det((vertices[1:]-vertices[0]).T))/6
    centroid=vertices.mean(axis=0)
    with ExitStack() as cleanup:
        zero=poisson.dm.createGlobalVec();zero.set(0.);cleanup.callback(zero.destroy)
        residual=zero.duplicate();cleanup.callback(residual.destroy)
        poisson.snes.computeFunction(zero,residual)
        for coefficients in [(1.,0.,0.,0.),(0.,1.,0.,0.),(0.,0.,1.,0.),(0.,0.,0.,1.)]:
            a,b,c,d=coefficients
            with mesh.access(field):
                x,y,z=field.coords.T
                field.data[:,0]=a+b*x+c*y+d*z
            actual=-float(residual.dot(field._gvec))
            expected=volume*(a+np.dot([b,c,d],centroid))
            np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-14)
    with pytest.raises(ValueError,match='no labelled cells globally'):
        _volume_region_indicator(mesh,'BadRegion',{'Missing':999})
    with pytest.raises(ValueError,match='no labelled cells globally'):
        _volume_region_indicator(mesh,'BadStratum',{'VolumeA':999})
    with pytest.raises(ValueError,match='no labelled cells globally'):
        _volume_region_indicator(mesh,'BoundaryRegion',{'Top':mesh.boundaries.Top.value})


@pytest.mark.parametrize('regions,error',[(None,ValueError),({},ValueError),([],ValueError),
    ({'':1},TypeError),({'Mantle':True},TypeError),({'Mantle':1.5},TypeError)])
def test_invalid_region_mapping_is_rejected_before_mesh_access(regions,error):
    with pytest.raises(error):_volume_region_indicator(None,'Unused',regions)
