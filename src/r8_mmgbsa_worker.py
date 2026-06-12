#!/usr/bin/env python3
"""MM-GBSA ΔG_bind for ONE complex (fresh process -> no OpenMM state corruption)."""
import sys, warnings; warnings.filterwarnings("ignore")
from pdbfixer import PDBFixer
from openmm import app, unit, LangevinIntegrator, Platform
ff=app.ForceField("amber14-all.xml","implicit/gbn2.xml"); plat=Platform.getPlatformByName("CPU"); KCAL=unit.kilocalorie_per_mole
def E(top,pos,mini=False):
    s=ff.createSystem(top,nonbondedMethod=app.NoCutoff)
    sim=app.Simulation(top,s,LangevinIntegrator(300*unit.kelvin,1/unit.picosecond,0.002*unit.picoseconds),plat)
    sim.context.setPositions(pos)
    if mini: sim.minimizeEnergy(maxIterations=300)
    st=sim.context.getState(getEnergy=True,getPositions=True); return st.getPotentialEnergy().value_in_unit(KCAL), st.getPositions()
fx=PDBFixer(sys.argv[1]); fx.findMissingResidues(); fx.findMissingAtoms(); fx.addMissingAtoms(); fx.addMissingHydrogens(7.0)
Ec,minpos=E(fx.topology,fx.positions,mini=True)
def chainE(keep):
    m=app.Modeller(fx.topology,minpos); m.delete([c for c in list(m.topology.chains()) if c.id!=keep]); return E(m.topology,m.positions)[0]
print(f"DG {Ec-chainE('A')-chainE('B'):.2f}")
