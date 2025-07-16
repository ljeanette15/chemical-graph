import cirq
import tequila as tq
import numpy as np
from itertools import product
import qibo
import numbers
import copy

import sys
sys.path.append("../Gates_Lab_Suite")

from Auto_Algorithm import *
from Core_Definition import *
from Visualization import *

####################################################################################################
####################### Functions for converting between Gates, Cirq, and Qibo #####################
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


def convert_gate_cirq_to_qibo(op, **kwargs):
    """
    Takes a cirq gate and converts it to a Qibo gate (or series of gates)

    Inputs:
        op: a cirq operation

    Returns a qibo.gates object of the appropriate gate
    """

    if isinstance(op.gate, cirq.XPowGate):
        return qibo.gates.RX(int(op.qubits[0].x), op.gate.exponent)
    
    elif isinstance(op.gate, cirq.CNotPowGate):
        return qibo.gates.CNOT(int(op.qubits[0].x), int(op.qubits[1].x))
    
    elif isinstance(op.gate, cirq.HPowGate):
        return qibo.gates.H(int(op.qubits[0].x))
    
    elif isinstance(op.gate, cirq.CZPowGate):
        return qibo.gates.CZ(int(op.qubits[0].x), int(op.qubits[1].x))
    
    elif isinstance(op.gate, cirq.YPowGate):
        return qibo.gates.RY(int(op.qubits[0].x), op.gate.exponent)
    
    elif isinstance(op.gate, cirq.ControlledGate):
        return qibo.gates.RY(int(op.qubits[-1].x), op.gate.sub_gate.exponent).controlled_by(int(op.qubits[0].x), int(op.qubits[1].x), int(op.qubits[2].x))

    elif isinstance(op.gate, cirq.ZPowGate):
        return qibo.gates.RZ(int(op.qubits[0].x), op.gate.exponent)


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
