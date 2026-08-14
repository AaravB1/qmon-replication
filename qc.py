import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace, concurrence

def fig1():
    qc = QuantumCircuit(3,3)
    qc.h(0)
    qc.cx(0,1)
    qc.h(2)
    qc.swap(1,2)
    qc.h(0)
    qc.h(1)
    qc.h(2)
    return qc

def schmidt_coeffs(sv , qubit , n_qubits):
    data = np.asarray(sv.data).reshape([2] * n_qubits)
    axis = n_qubits - 1 - qubit
    mat = np.moveaxis(data, axis, 0).reshape(2, -1)
    return np.linalg.svd(mat, compute_uv=False)

def is_separable(sv, qubit, n_qubits, tol):
    coeffs = schmidt_coeffs(sv, qubit, n_qubits)
    return len(coeffs) < 2 or coeffs[1] < tol


#implementation of algorithm for monitorable node selection 
def algo1(qc , tol=1e-9):

    #step 1
    n = qc.num_qubits
    locked = {q: False for q in range(n)}
    m_lists = {q: [] for q in range(n)}

    #step 2
    for i, instruction in enumerate(qc.data):
        name = instruction.operation.name
        if name in ("measure" , "barrier", "reset"):
            continue
        qubits = [qc.find_bit(qb).index for qb in instruction.qubits]

        prefix = QuantumCircuit(n)
        for op in qc.data[: i+1]:
            if op.operation.name not in ("measure", "barrier", "reset"):
                prefix.append(op.operation , [qc.find_bit(qb).index for qb in op.qubits])
        sv = Statevector(prefix)
        if len(qubits) == 1:
            q = qubits[0]
            if not locked[q]:
                m_lists[q].append(i)
        elif len(qubits) == 2:
            a, b = qubits
            rho_ab = partial_trace(sv, [x for x in range(n) if x not in (a,b)])
            c_ab = concurrence(rho_ab)

            if name == "swap":
                for q in (a,b):
                    locked[q] = not is_separable(sv , q, n, tol)
            else:
                for q in (a,b):
                    if c_ab > tol or not is_separable(sv, q, n, tol):
                        locked[q] = True

            for q in (a,b):
                if not locked[q]:
                    m_lists[q].append(i)
    return m_lists



if __name__ == "__main__":
    result = algo1(fig1())
    expected = {0: [0], 1: [3, 5], 2: [2]}
    print("got:     ", result)
    print("expected:", expected)