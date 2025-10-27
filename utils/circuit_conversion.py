import cirq
from cirq_ionq import GPI2Gate, GPIGate, MSGate
import tequila as tq
import numpy as np
from itertools import product
# import qibo
import numbers
import copy

import sys
sys.path.append("../Gates_Lab_Suite")

from Auto_Algorithm import *
from Core_Definition import *
from Visualization import *

####################################################################################################
####################### Functions for converting between Gates, Cirq #####################
####################################################################################################

def convert_gate_cirq_to_gates(op, **kwargs):
    """
    Takes a cirq gate and converts it to a Gates Lab suite gate (or series of gates)

    Inputs:
        op: a cirq operation

    Returns a Quantum_Gate object of the appropriate gate
    """

    # TODO: handle global phases

    if isinstance(op.gate, cirq.XPowGate):
        return Quantum_Gate("RX", op.qubits[0].x, angle=op.gate.exponent)
    
    elif isinstance(op.gate, cirq.CNotPowGate):
        return Quantum_Gate("CNOT", op.qubits[0].x, op.qubits[1].x)
    
    elif isinstance(op.gate, cirq.HPowGate):
        return Quantum_Gate("HAD", op.qubits[0].x)
    
    elif isinstance(op.gate, cirq.CZPowGate):
        return Quantum_Gate("CNOT", op.qubits[0].x, op.qubits[1].x)
    
    elif isinstance(op.gate, cirq.YPowGate):
        return Quantum_Gate("RY", op.qubits[0].x, angle=op.gate.exponent)
    
    elif isinstance(op.gate, cirq.ControlledGate):
        return Quantum_Gate("CCCRY", op.qubits[0].x, angle=op.gate.sub_gate.exponent)
    
    elif isinstance(op.gate, cirq.ZPowGate):
        return Quantum_Gate("RZ", op.qubits[0].x, angle=op.gate.exponent)


def cz_to_cnot(op):
    if isinstance(op.gate, cirq.CZPowGate) and op.gate.exponent == 1:
        c, t = op.qubits
        # Choose target as the *second* qubit in the CZ
        return [
            cirq.H(t),
            cirq.CNOT(c, t),
            cirq.H(t)
        ]
    return op



def cirq_to_gates(circuit, nqubits, name=None):

    gates_circuit = Quantum_Circuit(nqubits, name)

    for moment in circuit:
        for op in moment:

            gates_gate = convert_gate_cirq_to_gates(op)

            gates_circuit.Add_Gate(gates_gate)

    return gates_circuit


def cirq_to_ionq_cirq(circuit, nqubits, name=None):

    new_circuit = circuit.map_operations(cz_to_cnot)

    qubits = cirq.LineQubit.range(nqubits)
    meas = cirq.measure(qubits, key='output')
    new_circuit.append(meas)

    return new_circuit