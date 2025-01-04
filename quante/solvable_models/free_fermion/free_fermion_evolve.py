# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-08-12 23:00:14
# @Last Modified by:   hzhu
# @Last Modified time: 2024-08-12 23:07:40

import numpy as _np
import scipy as _scipy
import scipy.linalg as _sla


#######################################
# 时间演化 # TODO 效率较低，待优化
######################################


class FreeFermion:
    """自由费米子模型，只有在 obc 的时候，它与 obc 的 XY 模型是完全相同的（默认使用的是 自旋算符 而不是 Pauli 算符）

    只有最近邻，均匀相互作用时，与自旋参数的转换关系为：
        λ = (Jx + Jy) / 4 + 1j/4 * ( Jyx - Jxy )
        γ = (Jx - Jy) / 4 + 1j/4 * ( Jyx + Jxy )
        h = hz / 2

    核心步骤：self.get_core_eig()
    时间演化：self.vacuum_evol()

    example:
        ff = FreeFermion(site_num, λ, γ, h,"obc")
        ff.get_core_eig()
        t_list = _np.linspace(0,10,100)
        res = ff.vacuum_evol(t_list,oper="Sz_i",par=0)
        plt.plot(t_list, res)
    """

    def __init__(self, site_num, λ, γ, h, BC="obc"):
        # 22 位置 上三角
        H22 = _np.zeros((site_num, site_num), dtype=complex)
        if type(λ) in [float, int, complex]:
            H22 += _np.diag([λ / 2] * (site_num - 1), 1)
            if BC == "pbc":
                H22[0, -1] += _np.conj(λ) / 2
        elif type(λ) in [list, _np.ndarray]:
            temp = []
            for λi in λ:
                length = len(λi)
                if length in temp:
                    raise Exception("duplicated interaction")
                temp.append(length)
                H22 += _np.diag([i / 2 for i in λi], site_num - length)
        else:
            raise Exception("Wrong lambda type")
        # 22 位置 下三角
        H22 += H22.transpose().conj()
        # 22 位置 对角元
        if type(h) in [float, int, complex]:
            H22 += +_np.diag([-h] * site_num)
        elif type(h) in [list]:
            H22 += +_np.diag([-i for i in h])
        else:
            raise Exception("Wrong h type")
        # 21 位置 上三角
        H21 = _np.zeros((site_num, site_num), dtype=complex)
        if type(γ) in [float, int, complex]:
            H21 = _np.diag([γ / 2] * (site_num - 1), 1)
            if BC == "pbc":
                H21[0, -1] += -γ / 2
        elif type(γ) in [list, _np.ndarray]:
            temp = []
            for γi in γ:
                length = len(γi)
                if length in temp:
                    raise Exception("duplicated interaction")
                temp.append(length)
                H21 += +_np.diag([i / 2 for i in γi], site_num - length)
        # 21 位置 下三角
        H21 = H21 - H21.transpose()
        # 完整的矩阵
        self.H = _np.vstack(
            [
                _np.hstack([-H22.transpose(), H21.transpose().conj()]),
                _np.hstack([H21, H22]),
            ]
        )
        self.site_num = site_num
        self.core_eig = None
        self.core_vec = None
        self.eigenval = None
        self.groundeng = None
        self.firstexcitationeng = None

    def get_firstexcitationeng(self):
        if self.core_eig is None:
            self.get_core_eig()
        self.firstexcitationeng = 2 * min(self.core_eig[self.site_num :])

    def get_core_eig(self):
        self.core_eig, self.core_vec = _sla.eigh(self.H)
        self.groundeng = sum(self.core_eig[: self.site_num])

    def get_particular_eig(self, index):
        if self.core_eig is None:
            self.get_core_eig()
        core_eig = self.core_eig[: self.site_num]
        eigengIndex = bin(index)[2:]  # 转成二进制
        indexlength = len(eigengIndex)
        eigengIndex = (
            "0" * (self.site_num - indexlength) + eigengIndex
            if indexlength < self.site_num
            else eigengIndex
        )
        return sum((-1) ** int(i) * core_eig[ind] for ind, i in enumerate(eigengIndex))

    def get_eig(self):
        self.eigenval = []
        for i in range(2**self.site_num):
            self.eigenval.append(self.get_particular_eig(i))
        self.eigenval.sort()

    def get_AtBt(self, t, opt="all"):
        u = self.core_vec[: self.site_num, : self.site_num]
        v = self.core_vec[self.site_num :, : self.site_num].conj()
        Lambda = _np.hstack(
            [self.core_eig[: self.site_num], -self.core_eig[: self.site_num]]
        )
        if opt == "At":
            return (
                _np.hstack([u, v])
                @ _np.diag(_np.exp(-2j * Lambda * t))
                @ _np.vstack([u.conj().transpose(), v.conj().transpose()])
            )
        elif opt == "Bt":
            return (
                _np.hstack([u, v])
                @ _np.diag(_np.exp(-2j * Lambda * t))
                @ _np.vstack([v.transpose(), u.transpose()])
            )
        else:
            return (
                _np.hstack([u, v])
                @ _np.diag(_np.exp(-2j * Lambda * t))
                @ _np.vstack([u.conj().transpose(), v.conj().transpose()])
            ), (
                _np.hstack([u, v])
                @ _np.diag(_np.exp(-2j * Lambda * t))
                @ _np.vstack([v.transpose(), u.transpose()])
            )

    def vacuum_evol(self, t_list, oper: str, par: int = 0):
        """计算真空态某个算符随时间的演化

        Args:
            t_list (list): _description_
            oper (string): 支持的算符包括：['sz_i','sz_tot','fdagf_ij','fdagf_i_tot']
            par (int, optional): 格点的位置.Defaults to 0.

        Returns:
            list: 演化
        """
        if self.core_eig is None:
            self.get_core_eig()
        res = []
        if oper == "Sz_i":  # 计算每个格点 sz 的演化
            for t in t_list:
                Bt = self.get_AtBt(t, "Bt")
                A = Bt.conj() @ Bt.transpose()
                res.append(_np.diag(A).real - 1 / 2)  # 这里返回的是每个格点的演化
            return _np.array(res)
        elif oper == "fdag_f_i":  # 计算激发数 fdag_i f_i 的演化
            for t in t_list:
                Bt = self.get_AtBt(t, "Bt")
                A = Bt.conj() @ Bt.transpose()
                res.append(_np.diag(A).real)  # 这里返回的是每个格点激发数的演化
            return _np.array(res)
        elif oper == "Sz_tot":  # 计算总磁矩 sz_tot 的演化
            for t in t_list:
                Bt = self.get_AtBt(t, "Bt")
                res.append(_sla.norm(Bt, ord="fro") ** 2 - self.site_num / 2)
            return _np.array(res)
        elif oper == "fdagf_ij":  # 计算跳跃 fdag_i f_j 的演化
            for t in t_list:
                Bt = self.get_AtBt(t, "Bt")
                res.append((Bt[par[0], :].conj() @ Bt[par[1], :]))
            return _np.array(res)
        elif oper == "fdagf_ij_matrix":  # 计算空间关联矩阵的演化
            for t in t_list:
                Bt = self.get_AtBt(t, "Bt")
                res.append(Bt.conj() @ Bt)
            return _np.array(res)
        elif oper == "ff_ij":  # 计算跳跃 f_i f_j 的演化
            for t in t_list:
                At, Bt = self.get_AtBt(t)
                res.append((At[par[0], :] @ Bt[par[1], :]))
            return _np.array(res)
        elif oper == "ff_ij_matrix":  # 计算跳跃 f_i f_j 的演化
            for t in t_list:
                At, Bt = self.get_AtBt(t)
                res.append(At @ Bt.transpose())
            return _np.array(res)
        elif oper == "Sx_i_Sx_ip1":
            # 计算自旋 Sp_i (Sz_i+1 Sz_i+2 ... Sz_j-1) Sm_j 的演化
            for t in t_list:
                At, Bt = self.get_AtBt(t)
                res.append(
                    (
                        (At[par, :].transpose() - Bt[par, :].conj().transpose())
                        @ Bt[par + 1, :]
                    ).real
                    / 2
                )
            return _np.array(res)
        elif oper == "Sx_i_Sx_ip1_matrix":
            # 计算自旋 Sp_i (Sz_i+1 Sz_i+2 ... Sz_j-1) Sm_j 的演化
            for t in t_list:
                At, Bt = self.get_AtBt(t)
                res.append(
                    _np.diag(
                        (At[:-1, :].transpose() - Bt[:-1, :].conj().transpose())
                        @ Bt[1:, :]
                    ).real
                    / 2
                )
            return _np.array(res)
        elif oper == "Sy_i_Sy_ip1":
            # 计算自旋 Sp_i (Sz_i+1 Sz_i+2 ... Sz_j-1) Sm_j 的演化
            for t in t_list:
                At, Bt = self.get_AtBt(t)
                res.append(
                    -(
                        (At[par, :].transpose() + Bt[par, :].conj().transpose())
                        @ Bt[par + 1, :]
                    ).real
                    / 2
                )
            return _np.array(res)
        elif oper == "Sy_i_Sy_ip1_matrix":
            # 计算自旋 Sp_i (Sz_i+1 Sz_i+2 ... Sz_j-1) Sm_j 的演化
            for t in t_list:
                At, Bt = self.get_AtBt(t)
                res.append(
                    -_np.diag(
                        (At[:-1, :].transpose() + Bt[:-1, :].conj().transpose())
                        @ Bt[1:, :]
                    ).real
                    / 2
                )
            return _np.array(res)
        elif oper == "fdagf_tot":  # 计算总激发数 fdagf_tot 的演化
            for t in t_list:
                Bt = self.get_AtBt(t, "Bt")
                res.append(_sla.norm(Bt, ord="fro") ** 2)
            return _np.array(res)
        else:
            raise Exception("Not supported")


class OBCFermionExactSol:
    """只有形如：

        H = sum_{i=1}^{N-1}
            λ * (c_{i} c†_{i+1} - c†_{i} c_{i+1})
            +
            h sum_{i=1}^{N} ( c†_{i} c_{i} - c_{i} c†_{i} )

    也即自旋 XY 模型：

        H = sum_{i=1}^{N-1}
            J (s^x_i s^x_{i+1}
            +
            s^y_i s^y_{i+1})
            +
            hz
            sum_{i=1}^{N}
            s^z_i

        λ = J / 2;   h = hz / 2

    可以用 DST 解。因为真空态演化这里就没给计算
    """

    def __init__(self, site_num, λ, h):
        if type(λ) not in [float, complex, int]:
            raise Exception("Not supported")
        if type(λ) == complex:
            if λ.imag != 0:
                raise Exception("Not supported")
            λ = λ.real
        self.lbd = λ
        self.h = h
        self.groundeng = -sum(
            abs(λ * _np.cos(_np.pi / (site_num + 1) * (k + 1)) - h)
            for k in range(site_num)
        )
        self.core_eig = None
        self.firstexcitationeng = None
        self.site_num = site_num
        self.At = None

    def get_firstexcitationeng(self):
        k = 0
        self.firstexcitationeng = 2 * abs(
            self.lbd * _np.cos(_np.pi / (self.site_num + 1) * (k + 1)) - self.h
        )
        for k in range(1, self.site_num):
            temp = 2 * abs(
                self.lbd * _np.cos(_np.pi / (self.site_num + 1) * (k + 1)) - self.h
            )
            if temp < self.firstexcitationeng:
                self.firstexcitationeng = temp

    def get_core_eig(self):
        self.core_eig = [
            self.lbd * _np.cos(_np.pi / (self.site_num + 1) * (k + 1)) - self.h
            for k in range(self.site_num)
        ]

    def get_At(self, t):
        """计算 B(t)

        Args:
            t (float): 时间

        Note:
            A[k-j] =
            At(t) [site_num + 1 - abs(site_num + 1 - 2*((k-j)%(site_num + 1)))]
            c_j 算符的演化为：
            c_j(t) =
            1/(2(site_num + 1)) * sum_{k=1} ^N
                (A[k-j] - A[k+j]) * c_k

        Returns:
            _np.ndarray: A_i(t)
        """
        if self.core_eig is None:
            self.get_core_eig()
        data = _np.exp(2j * _np.array(self.core_eig) * t)
        self.At = _scipy.fft.dct([0] + list(data) + [0], type=1)

    def A(self, i):
        return self.At[
            abs(i)
            if abs(i) < self.site_num + 2
            else self.site_num + 2 - abs(i) % (self.site_num + 2) - 2
        ]


class PBCFermionExactSol:
    """只有 PBC 下形如：

        H = sum_{i=1}^{N}
            λ * (c_{i} c†_{i+1} - c†_{i} c_{i+1})
            +
            γ * (c_{i} c_{i+1} - c†_{i} c†_{i+1})
            +
            h sum_{i=1}^{N} ( c†_{i} c_{i} - c_{i} c†_{i} )

    其对应自旋系统是 J-W 边界条件，在有限尺寸下不能模拟自旋 PBC,

        H = sum_{i=1}^{N}
            J_x * s_j^x s_{j+1}^x
            +
            J_y * s_j^x s_{j+1}^x
            +
            J_yx * s_j^y s_{j+1}^x
            +
            J_xy * s_j^x s_{j+1}^x
            +
            h_z sum_{j=1}^N s_j^z

    两个模型的对应关系为：

        λ = (J_x + J_y)/4 - i(J_{yx} + J_{xy})/4
        γ = (J_x - J_y)/4 + i(J_{yx} - J_{xy})/4
        h = h_z / 2
    """

    def __init__(self, site_num, λ, γ, h):
        self.lbd = λ
        self.gam = γ
        self.h = h
        self.chi = _np.angle(λ)
        self.kappa = κ = _np.angle(γ)
        self.site_num = site_num
        self.groundeng = -sum(
            _np.sqrt(
                (h - λ.real * _np.cos(2 * _np.pi / site_num * (k))) ** 2
                + (abs(γ) * _np.sin(2 * _np.pi / site_num * (k))) ** 2
            )
            for k in range(1, site_num + 1)
        )
        self.firstexcitationeng = None
        self.eigenval = None
        self.omegalist = None
        self.thetalist = None

    def get_firstexcitationeng(self):
        self.firstexcitationeng = 2 * self.epsilon(0)
        for k in range(1, self.site_num):
            temp = 2 * self.epsilon(k)
            if temp < self.firstexcitationeng:
                self.firstexcitationeng = temp

    def omega(self, k):
        dn = self.h - abs(self.lbd) * _np.cos(self.chi) * _np.cos(
            2 * _np.pi / self.site_num * k
        )
        return _np.sqrt(
            dn**2 + (abs(self.gam) * _np.sin(2 * _np.pi / self.site_num * k)) ** 2
        )

    def theta(self, k):
        temp = _np.arccos(
            (
                self.h
                - abs(self.lbd)
                * _np.cos(self.chi)
                * _np.cos(2 * _np.pi / self.site_num * k)
            )
            / self.omega(k)
        )
        if _np.sin(2 * _np.pi / self.site_num * k) < 0:
            temp = 2 * _np.pi - temp
        return temp / 2

    def epsilon(self, k):
        return abs(self.lbd) * _np.sin(self.chi) * _np.sin(
            2 * _np.pi / self.site_num * k
        ) + self.omega(k)

    def get_core_eig(self):
        self.thetalist = [self.theta(k) for k in range(1, self.site_num + 1)]
        self.omegalist = [self.omega(k) for k in range(1, self.site_num + 1)]

    def get_particular_eig(self, index):
        core_eig = [self.epsilon(k) for k in range(1, self.site_num + 1)]
        eigengIndex = bin(index)[2:]  # 转成二进制
        indexlength = len(eigengIndex)
        eigengIndex = (
            "0" * (self.site_num - indexlength) + eigengIndex
            if indexlength < self.site_num
            else eigengIndex
        )
        return sum((-1) ** int(i) * core_eig[ind] for ind, i in enumerate(eigengIndex))

    def get_eig(self):
        self.eigenval = []
        for i in range(2**self.site_num):
            self.eigenval.append(self.get_particular_eig(i))
        self.eigenval.sort()

    def Nmk(self, k):
        return self.site_num - k if k != self.site_num else self.site_num

    def getABlist(self, t, tag="All"):
        if tag == "A":
            return _np.array(
                [
                    _np.cos(2 * self.theta(k)) * _np.cos(2 * self.epsilon(k) * t)
                    - 1j * _np.sin(2 * self.epsilon(k) * t)
                    for k in range(1, self.site_num + 1)
                ]
            )
        elif tag == "B":
            return _np.array(
                [
                    _np.exp(-1j * self.kappa)
                    * _np.sin(2 * self.theta(k))
                    * _np.sin(2 * self.epsilon(k) * t)
                    for k in range(1, self.site_num + 1)
                ]
            )
        else:
            A = _np.array(
                [
                    _np.cos(2 * self.theta(k)) * _np.cos(2 * self.epsilon(k) * t)
                    - 1j * _np.sin(2 * self.epsilon(k) * t)
                    for k in range(1, self.site_num + 1)
                ]
            )
            B = _np.array(
                [
                    _np.exp(-1j * self.kappa)
                    * _np.sin(2 * self.theta(k))
                    * _np.sin(2 * self.epsilon(k) * t)
                    for k in range(1, self.site_num + 1)
                ]
            )
            return A, B

    def vacuum_evol(self, t_list, oper: str, par: int = 0):
        """计算真空态某个算符随时间的演化

        Args:
            t_list (list): _description_
            oper (string): 支持的算符包括：['sz_i','sz_tot','fdagf_ij','fdagf_i_tot']
            par (int, optional): 格点的位置.Defaults to 0.

        Returns:
            list: 演化
        """
        if self.omegalist is None:
            self.get_core_eig()
        res = []
        if oper == "Sz_i":  # 这个模型不同格点 Sz_i 的演化是完全相同的！（因为 c_jdagc_j 有周期条件）
            for t in t_list:
                cur_expect = sum(
                    _np.sin(2 * self.thetalist[k]) ** 2
                    * _np.sin(2 * self.omegalist[k] * t) ** 2
                    for k in range(self.site_num)
                )
                res.append(cur_expect / self.site_num - 0.5)
            return _np.array(res)
        elif oper == "fdag_f_i":  # 计算激发数 fdag_i f_i 的演化
            for t in t_list:
                cur_expect = sum(
                    _np.sin(2 * self.thetalist[k]) ** 2
                    * _np.sin(2 * self.omegalist[k] * t) ** 2
                    for k in range(self.site_num)
                )
                res.append(cur_expect / self.site_num)
            return _np.array(res)
        elif oper == "Sz_tot":  # 计算总磁矩 sz_tot 的演化
            for t in t_list:
                cur_expect = sum(
                    _np.sin(2 * self.thetalist[k]) ** 2
                    * _np.sin(2 * self.omegalist[k] * t) ** 2
                    for k in range(self.site_num)
                )
                res.append(cur_expect - 0.5 * self.site_num)
            return _np.array(res)
        elif oper == "fdagf_tot":  # 计算总激发数 fdagf_tot 的演化
            for t in t_list:
                cur_expect = sum(
                    _np.sin(2 * self.thetalist[k]) ** 2
                    * _np.sin(2 * self.omegalist[k] * t) ** 2
                    for k in range(self.site_num)
                )
                res.append(cur_expect)
            return _np.array(res)
        elif oper == "fdagf_iip1":  # 计算总激发数 fdagf_tot 的演化
            for t in t_list:
                cur_expect = sum(
                    _np.exp(-1j * 2 * _np.pi / self.site_num * (k + 1))
                    * _np.sin(2 * self.thetalist[k]) ** 2
                    * _np.sin(2 * self.omegalist[k] * t) ** 2
                    for k in range(self.site_num)
                )
                res.append(cur_expect / self.site_num)

                # Alist,Blist = self.getABlist(t)
                # Atildelist = _np.fft.ifft(Alist) * _np.sqrt(site_num)
                # Btildelist = _np.fft.ifft(Blist) * _np.sqrt(site_num)
                # res.append(sum(Atildelist[l%self.site_num]*Btildelist[(l-1)%self.site_num] for l in range(1, self.site_num))/self.site_num)
            return _np.array(res)
        elif oper == "ff_iip1":  # 计算总激发数 fdagf_tot 的演化
            for t in t_list:
                cur_expect = _np.exp(-1j * self.kappa) * sum(
                    _np.exp(-1j * 2 * _np.pi / self.site_num * (k + 1))
                    * (
                        _np.sin(2 * self.thetalist[k])
                        * _np.sin(4 * self.omegalist[k] * t)
                        - 1j
                        * _np.sin(4 * self.thetalist[k])
                        * _np.sin(2 * self.omegalist[k] * t) ** 2
                    )
                    for k in range(self.site_num)
                )
                res.append(cur_expect / self.site_num / 2)
            return _np.array(res)
        elif oper == "Sx_i_Sx_ip1":
            # 计算自旋 Sp_i (Sz_i+1 Sz_i+2 ... Sz_j-1) Sm_j 的演化
            for t in t_list:
                cur_expect = sum(
                    _np.exp(-1j * 2 * _np.pi / self.site_num * (k + 1))
                    * (
                        -2
                        * _np.sin(2 * self.thetalist[k]) ** 2
                        * _np.sin(2 * self.omegalist[k] * t) ** 2
                        + _np.exp(-1j * self.kappa)
                        * _np.sin(2 * self.thetalist[k])
                        * _np.sin(4 * self.omegalist[k] * t)
                        - 1j
                        * _np.exp(-1j * self.kappa)
                        * _np.sin(4 * self.thetalist[k])
                        * _np.sin(2 * self.omegalist[k] * t) ** 2
                    )
                    for k in range(self.site_num)
                )
                res.append(cur_expect.real / self.site_num / 4)
            return _np.array(res)
        elif oper == "Sy_i_Sy_ip1":
            # 计算自旋 Sp_i (Sz_i+1 Sz_i+2 ... Sz_j-1) Sm_j 的演化
            for t in t_list:
                cur_expect = sum(
                    _np.exp(-1j * 2 * _np.pi / self.site_num * (k + 1))
                    * (
                        -2
                        * _np.sin(2 * self.thetalist[k]) ** 2
                        * _np.sin(2 * self.omegalist[k] * t) ** 2
                        - _np.exp(-1j * self.kappa)
                        * _np.sin(2 * self.thetalist[k])
                        * _np.sin(4 * self.omegalist[k] * t)
                        + 1j
                        * _np.exp(-1j * self.kappa)
                        * _np.sin(4 * self.thetalist[k])
                        * _np.sin(2 * self.omegalist[k] * t) ** 2
                    )
                    for k in range(self.site_num)
                )
                res.append(cur_expect.real / self.site_num / 4)
            return _np.array(res)
        else:
            raise Exception("Not supported")

if __name__ == "__main__":
    pass
    # site_num = 5

    # Jx, Jy, hz, Jxy, Jyx = (
    #     _np.random.rand(),
    #     _np.random.rand(),
    #     _np.random.rand(),
    #     _np.random.rand(),
    #     _np.random.rand(),
    # )
    # λ = (Jx + Jy) / 4 - 1j / 4 * (Jyx - Jxy)
    # γ = (Jx - Jy) / 4 + 1j / 4 * (Jyx + Jxy)
    # h = hz / 2

    # # """测试 PBCFermionExactSol，site num = 1000000 以内算基态，site num = 5000 以内算演化还是比较快的"""
    # pbcff = PBCFermionExactSol(site_num, λ, γ, h)
    # print(pbcff.groundeng)
    # pbcff.get_firstexcitationeng()
    # print(pbcff.firstexcitationeng)
    # pbcff.get_core_eig()
    # t_list = _np.linspace(0, 3, 100)
    # res = pbcff.vacuum_evol(t_list, "Sz_i")
    # res = pbcff.vacuum_evol(t_list, "ff_iip1")
    # plt.plot(t_list, res.real)

    # """测试 FreeFerion，site num = 1500 以内算基态，site num = 500 以内还是比较快的"""
    # λ = λ.real
    # γ = 0
    # ff = FreeFermion(site_num, λ, γ, h, "obc")
    # ff.get_core_eig()
    # ff.get_firstexcitationeng()
    # print("ffee:", ff.firstexcitationeng)
    # print("ffgd:", ff.groundeng)
    # res2 = ff.vacuum_evol(t_list, "ff_ij", [0, 1])
    # plt.plot(t_list, res2.real)

    # """用老办法写矩阵 用 exp 做演化 验证用——费米子，site num = 18 以内算基态（用 get_mineigval），site num = 14 以内还是比较快的"""
    # H = sum(λ * pl.f(j) * pl.fdag((j + 1) % site_num) -
    #         _np.conj(λ) * pl.fdag(j) * pl.f((j + 1) % site_num) +
    #         γ * pl.f(j) * pl.f((j + 1) % site_num) -
    #         _np.conj(γ) * pl.fdag(j) * pl.fdag((j + 1) % site_num) +
    #         h * pl.fdag(j) * pl.f(j) - h * pl.f(j) * pl.fdag(j)
    #         for j in range(site_num))
    # Hmat = H.to_matrix(f_par=site_num)
    # eig = Hmat.eigenenergies()
    # print(eig[1]-eig[0])
    # print(H.get_mineigval())
    # init_state = pl.qtobj("0"*site_num)
    # result = qt.sesolve(Hmat, init_state, t_list, [(pl.f(0)*pl.f(1)).to_matrix(f_par=site_num)])
    # plt.plot(t_list,result.expect[0].real)

    # """用老办法写矩阵 用 exp 做演化 验证用——自旋，site num = 18 以内算基态，site num = 14 以内还是比较快的"""
    # H = sum(Jx * pl.sx(j) * pl.sx(j + 1) + Jy * pl.sy(j) * pl.sy(j + 1) +
    #         Jyx * pl.sy(j) * pl.sx(j + 1) + Jxy * pl.sx(j) * pl.sy(j + 1)
    #         for j in range(site_num - 1)) + sum(
    #             hz * pl.sz(j) for j in range(site_num)) + _np.prod([
    #                 pl.Z(i) for i in range(1, site_num - 1)
    #             ]) * (Jy * pl.sx(0) * pl.sx(site_num - 1) + Jx * pl.sy(0) *
    #                 pl.sy(site_num - 1) - Jyx * pl.sy(0) * pl.sx(site_num - 1) -
    #                 Jxy * pl.sx(0) * pl.sy(site_num - 1))
    # Hmat = H.to_matrix(s_par=site_num)
    # init_state = pl.qtobj("0"*site_num)
    # result = qt.sesolve(Hmat, init_state, t_list, [(pl.sz(0)).to_matrix(s_par=site_num)])
    # plt.plot(t_list,result.expect[0].real)

    # # """测试 OBCFermionExactSol"""
    # λ = λ.real
    # γ = 0
    # ff = OBCFermionExactSol(site_num, λ.real, h)
    # ff.get_firstexcitationeng()
    # print("obcffee", ff.firstexcitationeng)
    # ff.get_core_eig()
    # print("obcffge", ff.groundeng)
    # # ff = FreeFermion(site_num, λ, 0, h, "obc")
    # # ff.get_core_eig()
    # # print(ff.groundeng)

    # H = sum(λ * pl.f(j) * pl.fdag((j + 1) % site_num) -
    #         _np.conj(λ) * pl.fdag(j) * pl.f((j + 1) % site_num)
    #         for j in range(site_num - 1)) + sum(
    #             h * pl.fdag(j) * pl.f(j) - h * pl.f(j) * pl.fdag(j)
    #             for j in range(site_num))
    # Hmat = H.to_matrix(f_par=site_num)
    # eig = Hmat.eigenenergies()
    # print("exee", eig[1] - eig[0])
    # print("exge", eig[0])
