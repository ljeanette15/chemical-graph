import tequila as tq
import numpy as np

# Define the geometry
g = "h 0.0 0.0 0.0\nh 0.0 0.0 1.5\nh 0.0 0.0 3.0\nh 0.0 0.0 4.5"

# Get two molecules (one with the JW transformation, one with the new transformation)
mol = tq.Molecule(backend="pyscf", geometry=g, basis_set="sto-3g", frozen_core=True)
mol2 = tq.Molecule(backend="pyscf", geometry=g, transformation='TaperedBinary', basis_set="sto-3g", frozen_core=True)

H = mol.make_hamiltonian()
H2 = mol2.make_hamiltonian()

U = mol.make_ansatz("spa", edges=[(0,1), (2,3)])
U2 = mol2.make_ansatz("spa", edges=[(0,1), (2,3)])

# Define a variable - what is this actually doing?
var = (tq.Variable("R0") + 0.5) * np.pi

UR0 = mol.UR(0,1,var) + mol.UR(2,3, var)
U = U + UR0

UR0 = mol2.UR(0,1,var) + mol2.UR(2,3, var)
U2 = U2 + UR0

res = tq.minimize(tq.ExpectationValue(U=U, H=H), silent=True)
res2 = tq.minimize(tq.ExpectationValue(U=tq.compile_circuit(U2), H=H2), silent=True)

print("difference JW and parity", abs(res.energy - res2.energy)*1000, "meh")
tq.draw(U2, backend="cirq") # U need to do pip install cirq or qpic as backends to do tq.draw