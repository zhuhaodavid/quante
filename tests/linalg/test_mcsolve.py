# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-16 15:10:39
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-16 18:22:56

# # import numpy as np

# # times = np.linspace(0.0, 10.0, 200)
# # psi0 = tensor(fock(2, 0), fock(10, 5))
# # a  = tensor(qeye(2), destroy(10))
# # sm = tensor(destroy(2), qeye(10))
# # H = 2*np.pi*a.dag()*a + 2*np.pi*sm.dag()*sm + 2*np.pi*0.25*(sm*a.dag() + sm.dag()*a)
# # data = mcsolve(H, psi0, times, [np.sqrt(0.1) * a], [a.dag() * a, sm.dag() * sm])
