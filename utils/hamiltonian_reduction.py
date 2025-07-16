# Tools for reducing the number of Pauli terms in a Hamiltonian
#   This is done by collecting co-measurable Pauli terms based
#   on where the identity character is in each string (since we
#   can measure in any basis then trace it out to get Id)

# Liam Jeanette
# 7-15-2025


from itertools import product
import numpy as np

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
