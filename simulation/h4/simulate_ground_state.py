
import tequila as tq
import numpy as np
import sys
import copy
from qiskit import QuantumCircuit

import matplotlib.pyplot as plt

sys.path.append('../../../../Gates_Lab_Suite')
sys.path.append('../../utils')

from hamiltonian_reduction import (
    reduce_measurements_naive
)

from simulation_tools import (
    simulate_data_qiskit_ionq,
    convert_tq_H_to_dict_H,
    simulate_data_qiskit,
    convert_population_data_to_expectation_values,
    convert_reduced_population_data_to_expectation_values,
    get_energy_from_expectation_values,
    build_ionq_noise_model
    )




def define_molecule(molecule="h2", spacing=1.5, transformation="JordanWigner"):

    if molecule == "h2":
        # Define the geometry
        g = f"h 0.0 0.0 0.0\nh 0.0 0.0 {spacing}"
    elif molecule == "h4":
        # Define the geometry
        g = f"h 0.0 0.0 0.0\nh 0.0 0.0 {spacing}\nh 0.0 0.0 {2*spacing}\nh 0.0 0.0 {3*spacing}"
    elif molecule == "h6":
        # Define the geometry
        g = f"h 0.0 0.0 0.0\nh 0.0 0.0 {spacing}\nh 0.0 0.0 {2*spacing}\nh 0.0 0.0 {3*spacing}\nh 0.0 0.0 {4*spacing}\nh 0.0 0.0 {5*spacing}"
    else:
        return -1
    
    # Get molecule
    mol = tq.Molecule(
            backend="pyscf", 
            geometry=g, 
            basis_set="sto-3g", 
            transformation=transformation
        ).use_native_orbitals()
    
    hf = mol.compute_energy('HF')
    fci = mol.compute_energy('FCI')

    return mol, hf, fci




def define_G1_circuit(mol):
    """
    Gets the G1 circuit for a given molecule. G1 is the simplest approximation in 
        this formulation. For example, H4 is just broken down into two H2 pairs.

    Inputs:
        mol: a tequila molecule object.

    Outputs: 
        qiskit: A qiskit circuit to prepare the state using the unitary
        cirq: A cirq circuit to prepare the state using the unitary
        energy: The calculated energy (using the unitary and Hamiltonian)
        H: The Hamiltonian which is used to measure the energy
    
    """

    edges = []
    for i in range(int(mol.n_orbitals/2)):
        edges.append((2*i, 2*i+1))

    U_SPA = mol.make_ansatz("SPA", edges=edges)
    
    H = mol.make_hamiltonian()

    # Unitary for two orbital rotations (parameterized by 'a' and 'b')
    U0 = mol.UR(0,1,'0')
    for i in range(1, int(mol.n_orbitals/2)):
        U0 += mol.UR(2*i, 2*i+1,f'{i}')

    U = U0 + U_SPA + U0.dagger()

    res = tq.minimize(tq.ExpectationValue(H=H, U=U), silent=True)

    U_mapped = U.map_variables(variables=res.variables)

    for gate in U_mapped.gates:
        try:
            if not isinstance(gate.parameter, tq.objective.objective.Variable):
                gate.parameter = gate.parameter.transformation(gate.parameter.args[0])
        except:
            x = 1

    cirq = tq.compile(U_mapped, backend="cirq").circuit
    qiskit = tq.compile(U_mapped, backend="qiskit").circuit

    energy = res.energy

    return qiskit, cirq, energy, H


def define_G2_circuit(mol):
    """ 
    Gets the G2 circuit for a given molecule. G2 expands on G1 by adding correlation between more
        orbital pairs. Here we just add a correlator between the middle two orbitals.

    Only valid for H4 at the moment. This method is meaningless for H2 but could work for H6
    
    Inputs:
        mol: a tequila molecule object.

    Outputs: 
        qiskit: A qiskit circuit to prepare the state using the unitary
        cirq: A cirq circuit to prepare the state using the unitary
        energy: The calculated energy (using the unitary and Hamiltonian)
        H: The Hamiltonian which is used to measure the energy
    """

    edges = []
    for i in range(int(mol.n_orbitals/2)):
        edges.append((2*i, 2*i+1))

    U_SPA = mol.make_ansatz("SPA", edges=edges)
    
    H = mol.make_hamiltonian()

    # Orbital rotations (same as in G1) - for the outer two pairs
    U0 = mol.UR(0,1,'0')
    for i in range(1, int(mol.n_orbitals/2)):
        U0 += mol.UR(2*i, 2*i+1,f'{i}')

    # Orbital Correlator for more graphs eg. (1,2):
    U1 = mol.UR(1, 2,'c')
    UC = mol.UC(1, 2,'d')
    U = U_SPA + U0 + U1 + UC + U0.dagger() + U1.dagger()
    res = tq.minimize(tq.ExpectationValue(H=H,U=U), silent=True)

    U_mapped = U.map_variables(variables=res.variables)

    for gate in U_mapped.gates:
        try:
            if not isinstance(gate.parameter, tq.objective.objective.Variable):
                gate.parameter = gate.parameter.transformation(gate.parameter.args[0])
        except:
            x = 1

    cirq = tq.compile(U_mapped, backend="cirq").circuit
    qiskit = tq.compile(U_mapped, backend="qiskit").circuit

    energy = res.energy

    return qiskit, cirq, energy, H


def plot_probabilities(data, labels, basis=None):
    """
    Plots the state populations for each set of populations in data. 

    inputs:
        data: a list of state populations
        labels: a list of labels for each population set in data
        basis: the measurement basis (just for the title)
    """
    n_datasets = len(data)
    n_states = len(data[0])
    
    # Width of each bar
    bar_width = 0.8 / n_datasets
    
    # X positions for the states
    x = np.arange(n_states)
    
    # Plot each dataset with offset
    for i, probs in enumerate(data):
        offset = (i - n_datasets/2 + 0.5) * bar_width
        plt.bar(x + offset, probs, width=bar_width, label=labels[i])
    
    plt.title(f"State Populations in {basis} basis")
    plt.xlabel("State")
    plt.ylabel("Population")
    plt.legend()
    plt.xticks(x)  # Center the x-tick labels on the groups
    plt.show()


if __name__ == '__main__':

    # Get molecule as a tequila object and calculate energies classically
    mol, hf, fci = define_molecule(molecule="h4", spacing=1.5, transformation="JordanWigner")

    # Need two qubits per orbital for Jordan Wigner encoding
    nqubits = mol.n_orbitals * 2

    print("-----------------------------------------------------------------")
    print('Hartree Fock Energy (Approximate) = ', hf)
    print('Full Configuration Interaction (FCI) Energy = ', fci)
    print("-----------------------------------------------------------------")

    # Get circuits for calculating energy
    qiskit_circuit, cirq_circuit, energy, H = define_G1_circuit(mol)
    print(f"difference from fci (G1): {abs(energy-fci)*1000} mH")

    qiskit_circuit_G2, cirq_cicuit_G2, energy_G2, H = define_G2_circuit(mol)
    print(f"difference from fci (G2): {abs(energy_G2-fci)*1000} mH")
    
    print("-----------------------------------------------------------------")

    print("Reducing measurements...")
    print(f"Original number of measurements: {len(H.keys())}")
    reduced_measurements = reduce_measurements_naive(H, nqubits)
    print(f"Reduced number of measurements: {len(reduced_measurements)}")

    print("-----------------------------------------------------------------")

    # Get Hamiltonian as a dictionary 
    H_dict = convert_tq_H_to_dict_H(H, nqubits)

    # Simulate and get G1 energy using qiskit (no shot noise and reduced H)
    qiskit_res_dict = simulate_data_qiskit(reduced_measurements, qiskit_circuit, nqubits)
    qiskit_evs = convert_reduced_population_data_to_expectation_values(qiskit_res_dict, reduced_measurements) 
    qiskit_energy = get_energy_from_expectation_values(H_dict, qiskit_evs)

    print(f"difference from fci (using qiskit (G1, reduced H)): {abs(qiskit_energy-fci)*1000} mH")

    # Simulate and get G2 energy using qiskit (no shot noise)
    qiskit_res_dict_G2 = simulate_data_qiskit(reduced_measurements, qiskit_circuit_G2, nqubits)
    qiskit_evs_G2 = convert_reduced_population_data_to_expectation_values(qiskit_res_dict_G2, reduced_measurements)
    qiskit_energy_G2 = get_energy_from_expectation_values(H_dict, qiskit_evs_G2)

    print(f"difference from fci (using qiskit) (G2): {abs(qiskit_energy_G2-fci)*1000} mH")

    print("-----------------------------------------------------------------")

    # ####################### Simulate using IonQ noise models using my own definition #########################

    aria1_noise_model = build_ionq_noise_model(0.0005, 0.0133)
    aria2_noise_model = build_ionq_noise_model(0.0006573, 0.01856)
    forte1_noise_model = build_ionq_noise_model(0.0002666, 0.0049493)

    nshots = 10000

    # # Simulate and get G1 energy using qiskit (no shot noise)
    # qiskit_sn_res_dict = simulate_data_qiskit(H_dict, qiskit_circuit, nqubits, nshots=nshots)
    # qiskit_sn_evs = convert_population_data_to_expectation_values(qiskit_sn_res_dict)
    # qiskit_sn_energy = get_energy_from_expectation_values(H_dict, qiskit_sn_evs)

    # print(f"difference from fci (using qiskit & shot noise (G1)): {abs(qiskit_sn_energy-fci)*1000} mH")

    # # Simulate and get G2 energy using qiskit (no shot noise)
    # qiskit_sn_res_dict_G2 = simulate_data_qiskit(H_dict, qiskit_circuit_G2, nqubits, nshots=nshots)
    # qiskit_sn_evs_G2 = convert_population_data_to_expectation_values(qiskit_sn_res_dict_G2)
    # qiskit_sn_energy_G2 = get_energy_from_expectation_values(H_dict, qiskit_sn_evs_G2)

    # print(f"difference from fci (using qiskit & shot noise (G2)): {abs(qiskit_sn_energy_G2-fci)*1000} mH")


    # Aria 1 System (G1)
    aria1_res_dict = simulate_data_qiskit(reduced_measurements, qiskit_circuit, nqubits, noise_model=aria1_noise_model, nshots=nshots)
    aria1_evs = convert_reduced_population_data_to_expectation_values(aria1_res_dict, reduced_measurements)
    aria1_energy = get_energy_from_expectation_values(H_dict, aria1_evs)
    
    print(f"difference from fci (local qiskit & aria 1 noise (G1)): {abs(aria1_energy-fci)*1000} mH")

    # Aria 1 System (G2)
    aria1_res_dict_G2 = simulate_data_qiskit(reduced_measurements, qiskit_circuit_G2, nqubits, noise_model=aria1_noise_model, nshots=nshots)
    aria1_evs_G2 = convert_reduced_population_data_to_expectation_values(aria1_res_dict_G2, reduced_measurements)
    aria1_energy_G2 = get_energy_from_expectation_values(H_dict, aria1_evs_G2)
    
    print(f"difference from fci (local qiskit & aria 1 noise (G2)): {abs(aria1_energy_G2-fci)*1000} mH")

    print("-----------------------------------------------------------------")

    # # # Aria 2 System
    # # aria2_res_dict = simulate_data_qiskit(H_dict, qiskit_circuit, nqubits, noise_model=aria2_noise_model, nshots=nshots)
    # # aria2_evs = convert_population_data_to_expectation_values(aria2_res_dict)
    # # aria2_energy = get_energy_from_expectation_values(H_dict, aria2_evs)

    # # print(f"difference from fci (using qiskit & aria 2 noise): {abs(aria2_energy-fci)*1000} mH")

    # # # Forte 1 System
    # # forte1_res_dict = simulate_data_qiskit(H_dict, qiskit_circuit, nqubits, noise_model=forte1_noise_model, nshots=nshots)
    # # forte1_evs = convert_population_data_to_expectation_values(forte1_res_dict)
    # # forte1_energy = get_energy_from_expectation_values(H_dict, forte1_evs)

    # # print(f"difference from fci (using qiskit & forte noise): {abs(forte1_energy-fci)*1000} mH")


    # ####################### Simulate using IonQ's noisy simulator API  #########################


    # Ideal (G1)
    ionq_ideal_res_dict = simulate_data_qiskit_ionq(reduced_measurements, qiskit_circuit, nqubits, nshots, noise_model="ideal", num_threads=20)
    ionq_ideal_evs = convert_reduced_population_data_to_expectation_values(ionq_ideal_res_dict, reduced_measurements)
    ionq_ideal_energy = get_energy_from_expectation_values(H_dict, ionq_ideal_evs)

    print(f"difference from fci (ionq ideal API (G1)): {abs(ionq_ideal_energy-fci)*1000} mH")

    # Ideal (G2)
    ionq_ideal_res_dict_G2 = simulate_data_qiskit_ionq(reduced_measurements, qiskit_circuit_G2, nqubits, nshots, noise_model="ideal", num_threads=20)
    ionq_ideal_evs_G2 = convert_reduced_population_data_to_expectation_values(ionq_ideal_res_dict_G2, reduced_measurements)
    ionq_ideal_energy_G2 = get_energy_from_expectation_values(H_dict, ionq_ideal_evs_G2)

    print(f"difference from fci (ionq ideal API (G2)): {abs(ionq_ideal_energy_G2-fci)*1000} mH")

    # Aria 1 system
    ionq_aria1_res_dict = simulate_data_qiskit_ionq(reduced_measurements, qiskit_circuit, nqubits, nshots, noise_model="aria-1", num_threads=20)
    ionq_aria1_evs = convert_reduced_population_data_to_expectation_values(ionq_aria1_res_dict, reduced_measurements)
    ionq_aria1_energy = get_energy_from_expectation_values(H_dict, ionq_aria1_evs)

    print(f"difference from fci (ionq  aria 1 API (G1)): {abs(ionq_aria1_energy-fci)*1000} mH")

    # Aria 1 system
    ionq_aria1_res_dict_G2 = simulate_data_qiskit_ionq(reduced_measurements, qiskit_circuit_G2, nqubits, nshots, noise_model="aria-1", num_threads=20)
    ionq_aria1_evs_G2 = convert_reduced_population_data_to_expectation_values(ionq_aria1_res_dict_G2, reduced_measurements)
    ionq_aria1_energy_G2 = get_energy_from_expectation_values(H_dict, ionq_aria1_evs_G2)

    print(f"difference from fci (ionq  aria 1 API (G2)): {abs(ionq_aria1_energy_G2-fci)*1000} mH")
