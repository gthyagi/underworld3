#!/usr/bin/env python3
"""Independent spherical evaluator accuracy and seven-repeat setup/batch timings."""
import argparse
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
from time import perf_counter
import tracemalloc
from types import SimpleNamespace
import assess
import numpy as np
import sympy as sp
import underworld3 as uw
from mpi4py import MPI
from benchmark_kramer_evaluators import measure, scaled_error
from underworld3.maths.spherical_harmonics import real_harmonic_values, solid_harmonic_sympy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-json',type=Path,required=True)
    args=parser.parse_args()
    if MPI.COMM_WORLD.size!=1:raise ValueError('Timing requires one process.')
    xyz=sp.symbols('x y z',real=True)
    mesh=SimpleNamespace(dim=3,X=sp.Matrix([xyz]))
    records=[]
    for density,boundary in [('delta','free'),('delta','zero'),('smooth','free'),('smooth','zero')]:
        family=getattr(assess,f'SphericalStokesSolution{density.capitalize()}{"FreeSlip" if boundary=="free" else "ZeroSlip"}')
        for count in (100,10000,100000):
            sol=uw.analytic.SphericalStokes(mesh,normalization='real_orthonormal',density=density,boundary=boundary)
            points=sol.sample_points(count)
            radii=np.linalg.norm(points,axis=1)
            for field,method in [('velocity','velocity_cartesian'),('pressure','pressure_cartesian'),('radial_normal_stress','radial_stress_cartesian')]:
                start=perf_counter()
                common=dict(Rm=sol.r_inner,Rp=sol.r_outer,g=sol.g,nu=sol.viscosity)
                branches=[family(2,0,sign,rp=sol.r_int,**common) for sign in (-1,1)] if density=='delta' else [family(2,0,2,**common)]*2
                setup=perf_counter()-start
                def oracle():
                    return np.asarray([getattr(branches[int(r>=sol.r_int)],method)(p) for r,p in zip(radii,points)])
                start=perf_counter();expected=oracle();first=perf_counter()-start
                methods=[('assess_point_loop',oracle,setup,0.,first)]
                for cse in (False,True):
                    start=perf_counter()
                    native=uw.analytic.SphericalStokes(mesh,normalization='real_orthonormal',density=density,boundary=boundary)
                    construction=perf_counter()-start
                    native.evaluation_cse=cse
                    start=perf_counter();native.compile_evaluators(field);compilation=perf_counter()-start
                    fn=lambda native=native:native.evaluate(field,points)
                    start=perf_counter();actual=fn();first=perf_counter()-start
                    scaled_error(actual,expected)
                    methods.append((f'numpy_cse_{cse}',fn,construction,compilation,first))
                if count==100:
                    def substitute():
                        rows=[]
                        for p,r in zip(points,radii):
                            expression=sol._cartesian_fields[int(r>=sol.r_int) if density=='delta' else 0][field]
                            terms=list(expression) if isinstance(expression,sp.MatrixBase) else [expression]
                            subs=dict(zip(sol._xyz,p))|{sol._r:r}
                            rows.append([float(term.evalf(subs=subs)) for term in terms])
                        return np.asarray(rows).reshape(expected.shape)
                    methods.append(('sympy_substitution_small_batch',substitute,None,None,None))
                for label,fn,setup,compilation,first in methods:
                    errors=scaled_error(fn(),expected)
                    fn() # warm up before seven measured repeats
                    timing=measure(fn,7)
                    tracemalloc.start();fn();_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
                    records.append(dict(geometry='spherical',case=f'{density}-{boundary}',field=field,points=count,method=label,
                                        construction_seconds=setup,evaluator_generation_seconds=compilation,
                                        first_evaluation_seconds=first,python_heap_peak_bytes=peak,**errors,**timing))
                    print(f'{density}-{boundary} {field} {count} {label} {timing["median_seconds"]:.6g}',flush=True)
    harmonic_records=[]
    points=sol.sample_points(100000)
    points[:2]=[[0,0,1],[0,0,-1]]
    for ell,m,component in [(2,0,'cos'),(3,1,'cos'),(3,2,'sin'),(4,4,'cos')]:
        kw=dict(degree=ell,order=m,component=component,normalization='real_orthonormal')
        H=solid_harmonic_sympy(xyz,**kw)
        start=perf_counter();symbolic=sp.lambdify(xyz,H/(sum(x*x for x in xyz))**sp.Rational(ell,2),'numpy',cse=True);setup=perf_counter()-start
        expected=real_harmonic_values(points,**kw)
        errors=scaled_error(symbolic(*points.T),expected)
        for label,fn in [('scipy_vectorized',lambda:real_harmonic_values(points,**kw)),('cartesian_symbolic',lambda:symbolic(*points.T))]:
            fn()
            harmonic_records.append(dict(degree=ell,order=m,component=component,points=len(points),method=label,
                                         evaluator_generation_seconds=setup if label=='cartesian_symbolic' else 0.,
                                         **errors,**measure(fn,7)))
    root=Path(uw.__file__).resolve().parents[2]
    payload=dict(schema_version=1,records=records,harmonics=harmonic_records,repetitions=7,
                 uw_source=str(uw.__file__),uw_revision=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
                 uw_dirty=bool(subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True).strip()),
                 python=sys.version,numpy=np.__version__,sympy=sp.__version__,platform=platform.platform(),
                 whole_run_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 note='Analytic evaluator timing, no FE solve. Heap measurements are not per-method RSS. Pole points included in harmonic checks.')
    args.output_json.parent.mkdir(parents=True,exist_ok=True)
    args.output_json.write_text(json.dumps(payload,indent=2)+'\n')

if __name__=='__main__':main()
