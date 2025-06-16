# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-16 18:45:10
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-16 18:47:13

# reference https://qutip.org/docs/4.7/guide/dynamics/dynamics-monte.html

# it should look like,

# code at qutip/solver/mcsolver.py - 294
# code at qutip/solver/integrator/scipy_integrator.py - 294
# def integrate(self, t, copy=False):
#     t_old, y_old = self._integrator.get_state(copy=False)
#     norm_old = self._prob_func(y_old)
#     while t_old < t:
#         t_step, state = self._integrator.mcstep(t, copy=False)
#         norm = self._prob_func(state)
#         if norm <= self.target_norm:
#             t_col, state = self._find_collapse_time(norm_old, norm,
#                                                     t_old, t_step)
#             self._do_collapse(t_col, state)
#             t_old, y_old = self._integrator.get_state(copy=False)
#             norm_old = 1.
#         else:
#             t_old, y_old = t_step, state
#             norm_old = norm

#     return t_old, _data.mul(y_old, 1 / self._norm_func(y_old))

# def run(self, tlist):
#     for t in tlist[1:]:
#         yield self.integrate(t, False)



# Example use solve_ivp

# import numpy as np
# from scipy.integrate import solve_ivp

# def fun(t, y):
#     # 你的微分方程
#     return -y

# thresh = 0.1

# def event_mod_below_thresh(t, y):
#     return np.linalg.norm(y) - thresh
# event_mod_below_thresh.terminal = True  # 触发后终止积分
# event_mod_below_thresh.direction = -1   # 只在从大到小穿过阈值时触发

# y0 = [1.0]
# t_span = (0, 10)

# sol = solve_ivp(
#     fun, t_span, y0,
#     events=event_mod_below_thresh,
#     max_step=1e-3  # 可选，限制最大步长，获得更细的时间分辨率
# )

# print("积分终止时间：", sol.t_events[0])
# print("终止时解：", sol.y_events[0])
