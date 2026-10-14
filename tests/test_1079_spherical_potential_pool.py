"""A DM work-vector alias must not overwrite the base Poisson solution."""
from types import SimpleNamespace
import numpy as np
import sympy as sp
from underworld3.postprocessing import spherical_potential as module


def test_sheet_correction_preserves_volume_solution_when_dm_reuses_snes_storage(monkeypatch):
    class Vec:
        def __init__(self,value=0.):self.array=np.array([value],dtype=float)
        def set(self,value):self.array[:]=value
        def copy(self):return Vec(self.array[0])
        def axpy(self,scale,other):self.array[:]+=scale*other.array
        def destroy(self):pass
    class DM:
        def __init__(self):self.base=Vec(5.);self.requests=0
        def getComm(self):return SimpleNamespace(tompi4py=lambda:SimpleNamespace(allreduce=lambda value,op:value))
        def createSubDM(self,field):return SimpleNamespace(destroy=lambda:None),SimpleNamespace(
            destroy=lambda:None,globalToLocal=lambda g,l,addv:l.array.__setitem__(slice(None),g.array))
        def getLocalVec(self):return Vec()
        def getGlobalVec(self):
            self.requests+=1
            # Native SNES kept a reference to a returned pool vector: the second
            # borrow hands out that exact storage as the correction workspace.
            return Vec() if self.requests==1 else self.base
        def restoreLocalVec(self,vec):pass
        def restoreGlobalVec(self,vec):pass
        def localToGlobal(self,local,global_,addv):global_.array[:]=local.array
    from underworld3.postprocessing import _spherical_solver_contract
    monkeypatch.setattr(_spherical_solver_contract, "require_native_solver_mesh", lambda mesh: None)
    dm=DM()
    variables=[]
    def variable(*args,**kwargs):
        local=Vec()
        value=SimpleNamespace(sym=[sp.Symbol('phi')],field_id=0,_gvec=Vec(),vec=local,_lvec=local,value=0.,_sync_lvec_to_gvec=lambda:None)
        variables.append(value);return value
    class Poisson:
        def __init__(self,mesh,u_Field,**kwargs):
            self.dm=dm;self.field=u_Field;self.petsc_options={}
            self.snes=SimpleNamespace(getSolution=lambda:dm.base,getJacobian=lambda:(object(),))
            self.solve_report=SimpleNamespace(converged=True,reason=1)
        def add_natural_bc(self,*args):pass
        def solve(self):pass
        def _scatter_global_to_fields(self,solution):self.field.value=float(solution.array[0])
    class KSP:
        def create(self,comm):return self
        def setOperators(self,operator):pass
        def setType(self,value):pass
        def getPC(self):return SimpleNamespace(setType=lambda value:None)
        def setTolerances(self,**kwargs):pass
        def solve(self,rhs,solution):solution.array[:]=rhs.array # unit operator
        def getConvergedReason(self):return 1
        def destroy(self):pass
    monkeypatch.setattr(module.uw.discretisation,'MeshVariable',variable)
    monkeypatch.setattr(module.uw.systems,'Poisson',Poisson)
    monkeypatch.setattr(module.uw.constitutive_models,'DiffusionModel',SimpleNamespace(Parameters=SimpleNamespace(diffusivity=1.)))
    monkeypatch.setattr(module,'PETSc',SimpleNamespace(KSP=KSP,InsertMode=SimpleNamespace(INSERT_VALUES=1)))
    monkeypatch.setattr(module,'assemble_spherical_harmonic_sheet_load',lambda **kwargs:(Vec(3.),{}))
    monkeypatch.setattr(module,'project_spherical_harmonic_trace_coefficient',lambda **kwargs:kwargs['variable'].value)
    xyz=sp.symbols('x y z',real=True)
    mesh=SimpleNamespace(dim=3,X=sp.Matrix([xyz]),dm=dm,_coords=np.array([[0.,0.,0.]]))
    result=module.solve_spherical_potential(mesh=mesh,field_name='Phi',harmonic_degree=2,
        radius_inner=.55,radius_outer=1.,radius_exterior=1.5,gravitational_constant=.01,
        sheet_densities={'Internal':1.},volume_density=1.)
    # The independent base (5) plus the sheet correction (3) is 8. Using the
    # borrowed SNES vector after the KSP overwrites it would instead give 6.
    assert result['outer_potential']==8.
    assert result['inner_potential']==8.
