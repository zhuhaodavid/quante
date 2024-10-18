# Localization in Anderson’s Hopping Model

## 安德森模型的形式

安德森模型: 单激发一维链, 包含 onsite 能量, 与 hopping 能量:

$$
    H = \sum_{i}^{} T_{i} c_{i}^{\dagger} c_{i} + \sum_{ir}^{} W_{r} c_{i}^{\dagger} c_{i + r}
$$

周期性边界条件, 并且: ${ \sum_{i}^{} c_{i}^{\dagger} c_{i} = 1 }$, 如果用自旋算符来表示:
$$
\begin{align*}
    H &= \sum_{i}^{} T_{i} s_{i}^{ +} s_{i}^{ -} + \sum_{ij}^{} W_{r} s_{i}^{ +} s_{i + r}^{ -} \\
    &= \sum_{ir}^{} \frac{W_{r}}{2} (s^{x}_{i} s^{x}_{i + r} + s^{y}_{i} s^{y}_{i + r} + \mathrm{i}(s^{y}_{i} s^{x}_{i + r} - s^{x}_{i}s^{y}_{i + r})) + \sum_{i}^{} \frac{T_{i}}{2} s^{z}_{i} + \sum_{i}^{} \frac{T_{i}}{2}
\end{align*}
$$
并且总磁矩为 ${ \sum_{i}^{} s^{z}_{i} = L - 1 }$, 其中 ${ L }$ 是链长.

因为总激发数为 ${ 1 }$, 因而可以选择 ${ L }$ 个本征态作为基矢:
$$
    \lbrace \left| 0 \right>, \left| 1 \right>, \cdots , \left| L - 1 \right> \rbrace
$$
如果是自旋表示, 可以选择:
$$
    \lbrace \left| 100\cdots 0 \right>,  \left| 010\cdots 0 \right>, \left| 001\cdots 0 \right>, \cdots , \left| 000\cdots 1 \right>\rbrace
$$
在第一种基矢的形式下, 哈密顿量就可以写成是:
$$
    H = \sum_{i}^{} T_{i} \ket{i}\hspace{-1mm}\bra{i} + \sum_{ir}^{} W_{r} \ket{i}\hspace{-1mm}\bra{i + r} 
$$

这组基地下这个哈密顿量对应的矩阵就是:
$$
\begin{align*}
    \begin{bmatrix}
        T_0 + W_0 & W_1 & W_2 &  \cdots  & W_{L - 1}  \\
        W_{ - 1} & T_1 + W_0 & W_1 & \cdots & W_{L - 2}  \\
        W_{ - 2} & W_{ - 1} & T_2 + W_0 & \cdots & W_{L - 3}  \\
        \cdots  & \cdots  & \cdots  & \cdots & \cdots   \\
        W_{ -(L - 1)}  & W_{ - (L - 2)}  & W_{ -(L - 3)}  & \cdots & T_{L - 1} + W_0  \\
    \end{bmatrix}
\end{align*}
$$

设本征态为:
$$
    \left| \psi \right> = \sum_{m}^{} u_{m} \left| m \right>
$$
那么对应的本征方程就是:
$$
\begin{align*}
    (T_0 + W_0) u_0 + W_1 u_1 + \cdots + W_{L - 1}u_{L - 1} = E u_0 \\
    W_{ - 1} u_0 + (T_1 + W_0) u_1 + \cdots + W_{L - 2}u_{L - 1} = E u_{1}\\
    \cdots 
\end{align*}
$$
总结起来就是:
$$
    T_{m} u_{m} + \sum_{r = - L/2}^{L/2} W_{r} u_{r + m} = E u_{m}
$$
其中 ${ m }$ 从 ${ 0 }$ 取到 ${ L - 1 }$.

五种形式完全等价的, 并且从自旋角度看,这个模型是 free model.

## 周期安德森模型的性质

${ T_{m} }$ 有周期性, 周期为 ${ q }$:
$$
    T_{m + q} = T_{m}
$$

${ L }$ 可以被 ${ q }$ 整除 (周期势场). 那么这一模型的性质有一般的特点.

首先利用布洛赫定理:
$$
u_m = e^{i k m} \phi_m
$$

并且 ${ \phi_m }$ 周期为 ${ q }$:
$$
    \phi_{m + q} = \phi_{m}
$$
那么 Schrödinger 方程:
$$
T_{m} \phi_m + \sum_{r} W_{r} e^{i kr} \phi_{m+r} = E \phi_m
$$
利用傅里叶展开:
$$
\phi_m = \sum_{n'=0}^{q-1} \phi_{n'} e^{i \frac{2 \pi n' m}{q}},\quad\;\; T_m = \sum_{n''=0}^{q-1} T_{n''} e^{i \frac{2 \pi n'' m}{q}}
$$
得到:
$$
\sum_{n''=0}^{q-1} T_{n''} e^{i \frac{2 \pi n'' m}{q}} \sum_{n'=0}^{q-1} \phi_{n'} e^{i \frac{2 \pi n' m}{q}} + \sum_{r} W_r e^{i kr} \sum_{n'=0}^{q-1} \phi_{n'} e^{i \frac{2 \pi n' (m+r)}{q}} = E \sum_{n'=0}^{q-1} \phi_{n'} e^{i \frac{2 \pi n' m}{q}}
$$

两边同乘 ${ e^{ - i \frac{2 \pi n m}{q}} }$ 并对 ${ m }$ 求和得到:

$$
\sum_{n'}^{} T_{n - n'} \phi_{n'} +  \sum_{r} W_r \phi_{n} e^{i k r} e^{i \frac{2 \pi n r}{q}} = E \phi_n
$$


因此得到关于 ${ k }$ 的本征问题: ${ H(k)\phi = E\phi }$. 其中:
$$
H_{nn'}(k) = T_{n-n'} + \sum_{r} W_r e^{i k r} e^{i \frac{2 \pi nr}{q}} \delta_{n,n'}
$$

这是一个 ${ q \times q }$ 的方程, 有 ${ q }$ 个本征值. 其中 ${ k = 2\pi n/ L }$, ${ n = 0,1,\cdots ,L/q - 1 }$ 一共 ${ L/q }$ 个取值, 因此得到全部的 ${ L }$ 个本征值.

![](代码/continuous_bands.svg)


