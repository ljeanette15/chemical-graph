# Tools for reducing the number of Pauli terms in a Hamiltonian
#   This is done by collecting co-measurable Pauli terms based
#   on where the identity character is in each string (since we
#   can measure in any basis then trace it out to get Id)

# Liam Jeanette
# 7-15-2025


import cirq
import tequila as tq
import numpy as np
# import qibo
import copy

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Kraus, SuperOp
from qiskit.visualization import plot_histogram
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
 
# Import from Qiskit Aer noise module
from qiskit_aer.noise import (
    NoiseModel,
    QuantumError,
    ReadoutError,
    depolarizing_error,
    pauli_error,
    thermal_relaxation_error,
    kraus_error
)

from qiskit_aer import Aer
from qiskit.quantum_info import SparsePauliOp, Statevector

import qiskit

import sys
sys.path.append("../Gates_Lab_Suite")

from Auto_Algorithm import *
from Core_Definition import *
from Visualization import *


##################################################################################################
###########                              General                                     #############
##################################################################################################

def build_ionq_noise_model(r1q, r2q):

    I = np.eye(2)
    X = np.array([[0, 1],[1, 0]], dtype=complex)
    Y = np.array([[0, -1j],[1j, 0]], dtype=complex)
    Z = np.array([[1, 0],[0, -1]], dtype=complex)
    paulis = [X, Y, Z, I]

    K0_1q = np.sqrt(1 - (3 * r1q / 4)) * I
    K1_1q = [np.sqrt(r1q/4) * P for P in paulis if not np.allclose(P, I)]
    kraus_1q = [K0_1q] + K1_1q

    pauli_tensors = [np.kron(Pa, Pb) for Pa in paulis for Pb in paulis]
    K00 = np.sqrt(1 - (15 * r2q / 16)) * np.kron(I, I)

    # Non-identity terms (15 of them)
    Kij = [np.sqrt(r2q/16) * PT for PT in pauli_tensors if not np.allclose(PT, np.kron(I,I))]

    kraus_2q = [K00] + Kij

    # Create QuantumError via Kraus
    error_1q = kraus_error(kraus_1q)
    error_2q = kraus_error(kraus_2q)

    # Build the NoiseModel
    noise_model = NoiseModel()

    one_qubit_gates = ["u1", "u2", "u3", "x", "h", "s", "sdg", "ry", "rz"]
    two_qubit_gates = ["cx", "cz", "swap"]

    noise_model.add_all_qubit_quantum_error(error_1q, one_qubit_gates)
    noise_model.add_all_qubit_quantum_error(error_2q, two_qubit_gates)

    return noise_model


def get_measurement_dict_from_H(H):

    measurements = {}

    for key in H.keys():
        pstring = "I" * 8
        for pchar in key:
            pstring = pstring[:pchar[0]] + pchar[1] + pstring[pchar[0] + 1:]
        measurements[pstring] = [pstring]

    return measurements


def convert_tq_H_to_dict_H(H, nqubits):

    H_dict = {}

    for key in H.keys():
        pstring = "I" * nqubits
        for pchar in key:
            pstring = pstring[:pchar[0]] + pchar[1] + pstring[pchar[0] + 1:]
        H_dict[pstring] = H[key]

    return H_dict


def zero_state(nqubits):

    state = np.zeros((2 ** nqubits, 2 ** nqubits))
    state[0][0] = 1

    return state


def get_results_from_frequencies(frequencies, nshots, nqubits):

    result = np.zeros((2 ** nqubits))

    for i in range(2 ** nqubits):
    
        result[i] = frequencies[format(i, f'0{nqubits}b')] / nshots

    return result


def expectation_value_measurements(pauli_char):

    if pauli_char == "I":
        return np.array([[1, 0], [0, 1]])

    return np.array([[1, 0], [0, -1]])


def expectation_value_measurements(pauli_char):

    if pauli_char == "I":
        return np.array([[1, 0], [0, 1]])

    return np.array([[1, 0], [0, -1]])


def normal_pauli_to_tequila_pauli(normal_pauli):

    pauli_list = []

    for i in range(len(normal_pauli)):
        if normal_pauli[i] != 'I':
            inner_tuple = (i, normal_pauli[i])
            pauli_list.append(inner_tuple)

    pauli_tuple = tuple(pauli_list)

    return pauli_tuple


def get_energy_from_data(H, data):

    """
    Calculates energy from a given Hamiltonian and state probability measurements. 

    Input:
        H: a dictionary with keys as Pauli strings and values as Hamiltonian weight
        data: a dictionary with keys as Pauli strings and values as results

    Returns an energy
    """


    energy = 0

    for pauli_string in data.keys():

        # Go through and construct the kronecker product of the Paulis as matrices
        measurement = expectation_value_measurements(pauli_string[0])
        for i in range(1, len(pauli_string)):
            measurement = np.kron(measurement, expectation_value_measurements(pauli_string[i]))

        # The expectation value of this pauli is just the inner product of the diagonals and the state probabilities  
        expectation_value = measurement.diagonal() @ data[pauli_string]

        # Then the energy is just the expectation value times the weight from the hamiltonian
        energy_contribution = expectation_value * H[normal_pauli_to_tequila_pauli(pauli_string)]

        energy += energy_contribution

    return energy


def get_energy_from_expectation_values(H, data):
    """
    Calculates energy from a given Hamiltonian and expectation value measurements. 

    Input:
        H: a dictionary with keys as Pauli strings and values as Hamiltonian weight
        data: a dictionary with keys as Pauli strings and values as expectation value measurement of that Pauli

    Returns an energy
    """
    energy = 0

    for pauli_string in data.keys():

        energy_contribution = data[pauli_string] * H[pauli_string]

        energy += energy_contribution

    return energy


def convert_population_data_to_expectation_values(res_dict):

    ev_dict = {}

    for pauli_string in res_dict.keys():

        # Go through and construct the kronecker product of the Paulis as matrices
        measurement = expectation_value_measurements(pauli_string[0])
        for i in range(1, len(pauli_string)):
            measurement = np.kron(measurement, expectation_value_measurements(pauli_string[i]))

        # The expectation value of this pauli is just the inner product of the diagonals and the state probabilities  
        expectation_value = measurement.diagonal() @ res_dict[pauli_string]

        ev_dict[pauli_string] = expectation_value

    return ev_dict


##################################################################################################
###########                       Gates lab suite                                    #############
##################################################################################################
def measurment_gate_gates(pauli_character, qubit):
    if pauli_character == "Y":
        return Quantum_Gate("SKAX", qubit, angle=-0.5)
    elif pauli_character == "X":
        return Quantum_Gate("SKAY", qubit, angle=0.5)
    else:
        return None
    

##################################################################################################
###########                                Qibo                                      #############
##################################################################################################
# def measurment_gate_qibo(pauli_character, qubit):
#     if pauli_character == "Y":
#         return qibo.gates.RX(qubit,0.5 * np.pi)
#     elif pauli_character == "X":
#         return qibo.gates.RY(qubit,-0.5 * np.pi)
    
#     return None


# def simulate_data_qibo(measurements, qibo_circuit, nqubits):

#     initial_rho = zero_state(nqubits)
#     nshots = 5000

#     res = []
#     res_sn = []

#     data_dict = {}
#     data_dict_sn = {}


#     # Go through each Pauli measurement basis
#     for i in range(len(measurements)):

#         print(i, end="\r")

#         # Need to make a new copy each time as to not overwrite the original
#         qibo_copy = copy.deepcopy(qibo_circuit)

#         # Go through each character (j) in the pauli string (i) and add the appropriate gate for that basis
#         for j in range(nqubits):
#             pauli_char = list(measurements.keys())[i][j]
#             if pauli_char not in ["I", "Z"]:
#                 qibo_copy.add(measurment_gate_qibo(pauli_char, j))

#         qibo_copy.add(qibo.gates.M(*range(nqubits)))

#         # Get results with and without shot noise
#         result_sn = qibo_copy(nshots=nshots, initial_state=initial_rho)
#         result = qibo_copy(nshots=1, initial_state=initial_rho)

#         res.append(result.probabilities())
#         data_dict[list(measurements.keys())[i]] = result.probabilities()

#         result_shotnoise = get_results_from_frequencies(result_sn.frequencies(), nshots, nqubits)
#         res_sn.append(result_shotnoise)
        
#         data_dict_sn[list(measurements.keys())[i]] = result_shotnoise

#     return data_dict, data_dict_sn

# #change

##################################################################################################
###########                               Cirq                                       #############
##################################################################################################
def measurment_gate_cirq(pauli_character, qubit):
    if pauli_character == "Y":
        return cirq.rx(-0.5 * np.pi)(qubit)
    elif pauli_character == "X":
        return cirq.ry(0.5 * np.pi)(qubit)


def measurment_gate_qiskit(pauli_character, qubit):
    if pauli_character == "Y":
        return qiskit.rx(-0.5 * np.pi)(qubit)
    elif pauli_character == "X":
        return qiskit.ry(0.5 * np.pi)(qubit)
    

def convert_tq_H_to_cirq_H(H, qubits):

    pauli_strings = []

    for key in H.keys():
        
        pauli_list = []
        for pauli in key:
            pauli_list.append(tq_to_cirq_pauli(pauli, qubits))

        pauli_strings.append(cirq.PauliString(H[key], *pauli_list))

    pauli_sum = cirq.PauliSum.from_pauli_strings(pauli_strings)

    return pauli_sum, pauli_strings


def get_qiskit_pauli_list_from_tq_H(H):

    pauli_op_list = []

    for key, value in H.items():

        pauli = "IIIIIIII"

        for i in range(len(key)):

            pauli = pauli[:key[i][0]] + key[i][1] + pauli[key[i][0] + 1:]
            

        pauli_tuple = (pauli, value)

        pauli_op_list.append(pauli_tuple)

    return pauli_op_list


def tq_to_cirq_pauli(tq_pauli, qubits):
    if tq_pauli[1] == 'Z':
        return cirq.Z(qubits[tq_pauli[0]])
    if tq_pauli[1] == 'X':
        return cirq.X(qubits[tq_pauli[0]])
    if tq_pauli[1] == 'Y':
        return cirq.Y(qubits[tq_pauli[0]])
    if tq_pauli[1] == 'I':
        return cirq.I(qubits[tq_pauli[0]])


def simulate_data_cirq(H_dict, cirq_circuit, nqubits):

    res = []

    data_dict = {}

    cirq_simulator = cirq.Simulator()
    qubits = cirq.LineQubit.range(nqubits)
    
    # Go through each Pauli measurement basis
    for i in range(len(H_dict.keys())):

        print(i, end="\r")

        # Need to make a new copy each time as to not overwrite the original
        cirq_copy = copy.deepcopy(cirq_circuit)

        # Go through each character (j) in the pauli string (i) and add the appropriate gate for that basis
        for j in range(nqubits):
            pauli_char = list(H_dict.keys())[i][j]
            if pauli_char not in ["I", "Z"]:
                cirq_copy.append(measurment_gate_cirq(pauli_char, qubits[j]))

        # Get results without shot noise
        result = cirq_simulator.simulate(cirq_copy)

        res.append(abs(result.final_state_vector ** 2))

        data_dict[list(H_dict.keys())[i]] = abs(result.final_state_vector ** 2)

    return data_dict


def simulate_data_qiskit(H_dict, qiskit_circuit, nqubits):

    res = []

    data_dict = {}

    qiskit_simulator = Aer.get_backend("statevector_simulator")
    
    # Go through each Pauli measurement basis
    for i in range(len(H_dict.keys())):

        print(i, end="\r")

        # Need to make a new copy each time as to not overwrite the original
        qiskit_copy = copy.deepcopy(qiskit_circuit)

        # Go through each character (j) in the pauli string (i) and add the appropriate gate for that basis
        for j in range(nqubits):
            pauli_char = list(H_dict.keys())[i][j]
            if pauli_char not in ["I", "Z"]:
                qiskit_copy.append(measurment_gate_qiskit(pauli_char, qubits[j]))

        # Get results without shot noise
        meas = SparsePauliOp.from_list([("Z" * nqubits, 1.0)])
        state = qiskit_simulator.run(qiskit_copy).result().get_statevector()
        result = Statevector(state).expectation_value(meas)

        res.append(abs(result ** 2))

        data_dict[list(H_dict.keys())[i]] = abs(result.final_state_vector ** 2)

    return data_dict

