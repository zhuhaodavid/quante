# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-16 22:24:32
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-16 22:24:38

from qiskit.quantum_info import random_clifford

cliff = random_clifford(2)  # Two-qubit random Clifford
print(cliff.to_circuit())   # View as a quantum circuio


