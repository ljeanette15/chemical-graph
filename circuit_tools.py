import cirq
import tequila as tq
import numpy as np
from itertools import product
import qibo
import numbers

import sys
sys.path.append("../Gates_Lab_Suite")

from Auto_Algorithm import *
from Core_Definition import *
from Visualization import *


############ Functions for converting between Gates, Cirq, and Qibo #####################

def convert_gate_cirq_to_gates(op, **kwargs):
    """
    Takes a cirq gate and converts it to a Gates Lab suite gate (or series of gates)

    Inputs:
        op: a cirq operation

    Returns a Quantum_Gate object of the appropriate gate
    """

    # TODO: handle global phases

    if isinstance(op.gate, cirq.XPowGate):
        if isinstance(op.gate.exponent, numbers.Number):
            return Quantum_Gate("RX", op.qubits[0].x, angle=op.gate.exponent)
        else:
            return Quantum_Gate("RX", op.qubits[0].x, angle=0.5)
    
    elif isinstance(op.gate, cirq.CNotPowGate):
        return Quantum_Gate("CNOT", op.qubits[0].x, op.qubits[1].x)
    
    elif isinstance(op.gate, cirq.HPowGate):
        return Quantum_Gate("HAD", op.qubits[0].x)
    
    elif isinstance(op.gate, cirq.CZPowGate):
        return Quantum_Gate("CNOT", op.qubits[0].x, op.qubits[1].x)
    
    elif isinstance(op.gate, cirq.YPowGate):
        if isinstance(op.gate.exponent, numbers.Number):
            return Quantum_Gate("RY", op.qubits[0].x, angle=op.gate.exponent)
        else:
            return Quantum_Gate("RY", op.qubits[0].x, angle=0.5)
    
    # TODO: make this actually detect CCCRy gates (and implement a CCCY gate)
    elif isinstance(op.gate, cirq.ControlledGate):
        if isinstance(op.gate.exponent, numbers.Number):
            return Quantum_Gate("RY", op.qubits[0].x, angle=op.gate.exponent)
        else:
            return Quantum_Gate("RY", op.qubits[0].x, angle=0.5)
    
    elif isinstance(op.gate, cirq.ZPowGate):
        if isinstance(op.gate.exponent, numbers.Number):
            return Quantum_Gate("RZ", op.qubits[0].x, angle=op.gate.exponent)
        else:
            return Quantum_Gate("RZ", op.qubits[0].x, angle=0.5)
    

def convert_gate_cirq_to_qibo(op, **kwargs):
    """
    Takes a cirq gate and converts it to a Qibo gate (or series of gates)

    Inputs:
        op: a cirq operation

    Returns a qibo.gates object of the appropriate gate
    """

    # TODO: take into account global phase

    if isinstance(op.gate, cirq.XPowGate):
        if isinstance(op.gate.exponent, numbers.Number):
            return qibo.gates.RX(int(op.qubits[0].x), op.gate.exponent)
        else:
            return qibo.gates.RX(int(op.qubits[0].x), 0.5)
    
    elif isinstance(op.gate, cirq.CNotPowGate):
        return qibo.gates.CNOT(int(op.qubits[0].x), int(op.qubits[1].x))
    
    elif isinstance(op.gate, cirq.HPowGate):
        return qibo.gates.H(int(op.qubits[0].x))
    
    elif isinstance(op.gate, cirq.CZPowGate):
        return qibo.gates.CZ(int(op.qubits[0].x), int(op.qubits[1].x))
    
    elif isinstance(op.gate, cirq.YPowGate):
        if isinstance(op.gate.exponent, numbers.Number):
            return qibo.gates.RY(int(op.qubits[0].x), op.gate.exponent)
        else:
            return qibo.gates.RY(int(op.qubits[0].x), 0.5)
    
    elif isinstance(op.gate, cirq.ControlledGate):
        # TODO: make this actually do a CCCRy gate
        if isinstance(op.gate.exponent, numbers.Number):
            return qibo.gates.RY(int(op.qubits[0].x), op.gate.exponent)
        else:
            return qibo.gates.RY(int(op.qubits[0].x), 0.5)

    elif isinstance(op.gate, cirq.ZPowGate):
        if isinstance(op.gate.exponent, numbers.Number):
            return qibo.gates.RZ(int(op.qubits[0].x), op.gate.exponent)
        else:
            return qibo.gates.RZ(int(op.qubits[0].x), 0.5)


def convert_gate_gates_to_qibo(gate, **kwargs):
    """
    Takes a Gates Lab suite gate and converts it to a Qibo gate (or series of gates)

    Inputs:
        gate: a Gates Lab suite gate

    Returns a qibo.gates object of the appropriate gate
    """

    # TODO: implement this...
    pass


def cirq_to_gates(circuit, nqubits, name=None):

    gates_circuit = Quantum_Circuit(nqubits, name)

    for moment in circuit:
        for op in moment:

            gates_gate = convert_gate_cirq_to_gates(op)

            gates_circuit.Add_Gate(gates_gate)

    return gates_circuit


def cirq_to_qibo(circuit, nqubits, name=None):

    qibo_circuit = qibo.Circuit(nqubits, density_matrix=True)

    for moment in circuit:
        for op in moment:

            qibo_gate = convert_gate_cirq_to_qibo(op)

            qibo_circuit.add(qibo_gate)

    return qibo_circuit


def gates_to_qibo(circuit, nqubits, name=None):

    # TODO: Implement this...

    qibo_circuit = qibo.Circuit(nqubits, density_matrix=True)

    pass



############ Functions for reducing the number of measurements in a Hamiltonian #####################

def get_pauli_stack(hamiltonian, nqubits):
    
    pauli_stack = []

    # Add strings to stack in order of least to most identity measurements
    for i in range(nqubits):

        # Each key is a pauli measurement
        for key in hamiltonian.keys():

            pauli_string = "I" * nqubits

            # Tequila hamiltonian encodes the pauli strings using tuples. The first element in the tuple is
            # the location of the character, and the second element is the character itself

            for pauli_tuple in key:
                pauli_string = pauli_string[:pauli_tuple[0]] + pauli_tuple[1] + pauli_string[pauli_tuple[0]+1:]

            if pauli_string.count('I') == i:
                pauli_stack.append(pauli_string)

    return pauli_stack


def get_all_pauli_strings_no_identity(nqubits):

    all_pauli_strings = []
    for p in product(['X', 'Y', 'Z'], repeat=nqubits):
        pstring = "".join(p)
        all_pauli_strings.append(pstring)

    return all_pauli_strings


def get_I_indeces(pauli_str):

    indeces = []
    for i in range(len(pauli_str)):
        if pauli_str[i] == 'I':
            indeces.append(i)
    
    return indeces


def remove_at_indeces(pauli_str, indeces):

    # if there were any identities, remove them
    if len(indeces) > 0:
        shortened_pauli = pauli_str[:indeces[0]]
        for j in range(len(indeces) - 1):
            shortened_pauli += pauli_str[indeces[j]+1:indeces[j+1]]
        shortened_pauli += pauli_str[indeces[-1]+1:]

    else:
        shortened_pauli = pauli_str

    return shortened_pauli


def check_equality(pauli_str1, pauli_str2):

    pauli1_indeces = get_I_indeces(pauli_str1)

    shortened_pauli1 = remove_at_indeces(pauli_str1, pauli1_indeces)
    shortened_pauli2 = remove_at_indeces(pauli_str2, pauli1_indeces)

    shortened_pauli2_indeces = get_I_indeces(shortened_pauli2)

    shorter_pauli1 = remove_at_indeces(shortened_pauli1, shortened_pauli2_indeces)
    shorter_pauli2 = remove_at_indeces(shortened_pauli2, shortened_pauli2_indeces)

    if shorter_pauli1 == shorter_pauli2:
        return True
    
    return False


def reduce_measurements_naive(hamiltonian, nqubits):

    all_pauli_strings = get_all_pauli_strings_no_identity(nqubits)
    pauli_stack = get_pauli_stack(hamiltonian, nqubits)

    all_pauli_strings_dict = {}
    for pauli_string in all_pauli_strings:
        all_pauli_strings_dict[pauli_string] = []

    for pauli_string in all_pauli_strings:
        for pauli_basis in pauli_stack:
            if check_equality(pauli_basis, pauli_string):
                all_pauli_strings_dict[pauli_string].append(pauli_basis)

    basis_measurements = {}

    max_list_size = -1
    max_list = []
    max_string = ""

    while max_list_size != 0:

        next_max_list_size = 0
        next_max_list = []
        next_max_string = ""

        for pauli_string in all_pauli_strings:

            new_list = []
            for pauli in all_pauli_strings_dict[pauli_string]:
                    
                if pauli not in max_list:
                    new_list.append(pauli)

            all_pauli_strings_dict[pauli_string] = new_list

            if len(new_list) > next_max_list_size:
                next_max_list_size = len(new_list)
                next_max_list = all_pauli_strings_dict[pauli_string]
                next_max_string = pauli_string

        if next_max_list_size == 0:
            break

        basis_measurements[next_max_string] = next_max_list

        del all_pauli_strings_dict[next_max_string]
        all_pauli_strings.remove(next_max_string)

        max_list_size = next_max_list_size
        max_list = next_max_list
        max_string = next_max_string

    return basis_measurements


def reduce_measurements_wrong(hamiltonian, nqubits):

    """
    Takes a tequila hamiltonian object and reduces the number of measurements necessary to take experimentally.
    When an identity measurement is required, the qubit can be measured experimentally in any basis, then traced over,
    so the identity is a sort of wildcard. Therefore we can check if any measurements are "double counted" by the
    identity and remove those measurements, calculating them post-data taking.

    Inputs: 
        hamiltonian: a tequila hamiltonian object
        nqubits: the number of qubits used in the circuit

    Output: a minimal subset of measurements (pauli strings)

    """

    pauli_stack = get_pauli_stack(hamiltonian, nqubits)


    basis_measurements = {}


    for pauli in pauli_stack:

        # Get the indeces of any identity character in the string
        pauli_indeces = get_I_indeces(pauli)
        
        shortened_pauli = remove_at_indeces(pauli, pauli_indeces)

        # Go through each basis already included in the minimal set
        matched = False
        for basis in basis_measurements:

            # Remove the characters from the location of the identities in the pauli string
            shortened_basis = remove_at_indeces(basis, pauli_indeces)

            # Now find the indeces of the identity characters in the basis string
            basis_indeces = get_I_indeces(shortened_basis)

            shorter_basis = remove_at_indeces(shortened_basis, basis_indeces)
            shorter_pauli = remove_at_indeces(shortened_pauli, basis_indeces)

            if shorter_pauli == shorter_basis:
                matched = True
                basis_measurements[basis].append(pauli)
                break

        if matched == False:
            basis_measurements[pauli] = [pauli]

    return basis_measurements


def measurment_gate_gates(pauli_character, qubit):
    if pauli_character == "Y":
        return Quantum_Gate("SKAX", qubit, angle=-0.5)
    elif pauli_character == "X":
        return Quantum_Gate("SKAY", qubit, angle=0.5)
    else:
        return None
    

def measurment_gate_qibo(pauli_character, qubit):
    if pauli_character == "Y":
        return qibo.gates.RX(qubit,-0.5)
    elif pauli_character == "X":
        return qibo.gates.RY(qubit, 0.5)
    
    return None



############ Functions for calculating energy from data and Hamiltonian #####################

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


def get_energy_from_data(hamiltonian, data_dictionary, reduced_measurements):

    energy = 0

    for measured_string in data_dictionary.keys():

        for calculated_string in reduced_measurements[measured_string]:

            # Go through and construct the kronecker product of the Paulis as matrices
            measurement = expectation_value_measurements(calculated_string[0])
            for i in range(1, len(calculated_string)):
                measurement = np.kron(measurement, expectation_value_measurements(calculated_string[i]))

            # The expectation value of this pauli is just the inner product of the diagonals and the state probabilities  
            expectation_value = measurement.diagonal() @ data_dictionary[measured_string]

            # Then the energy is just the expectation value times the weight from the hamiltonian
            energy_contribution = expectation_value * hamiltonian[normal_pauli_to_tequila_pauli(calculated_string)]

            energy += energy_contribution

    return energy

