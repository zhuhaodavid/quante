# 开边界 XXZ 链：纵向边界磁场下的 Bethe Ansatz 公式整理

本文整理三种情形：

1. $|\Delta|<1$，即 $\Delta=\cos\eta$；
2. $\Delta>1$，即 $\Delta=\cosh\gamma$；
3. $\Delta=1$，即 XXX 极限。

以下只考虑左右边界磁场沿 $z$ 方向的情形，也就是 diagonal $K$-matrices。如果边界场有 $x,y$ 分量，则 $K_\pm$ 非对角，普通 ABA 一般不能直接使用，需要 off-diagonal Bethe ansatz 或相关变换。

---

# 一、$|\Delta|<1$ 的情形

先考虑 $|\Delta|<1$，也就是

$$
\Delta=\cos\eta,\qquad 0<\eta<\pi,
$$

并且仍然使用

$$
a(u)=\sin(u+\eta),\qquad b(u)=\sin u,\qquad c=\sin\eta .
$$

如果左右边界磁场沿 $z$ 方向，即

$$
H
=
\sum_{k=1}^{L-1}
\left(
\sigma_k^x\sigma_{k+1}^x+
\sigma_k^y\sigma_{k+1}^y+
\Delta\sigma_k^z\sigma_{k+1}^z
\right)
+
h_- \sigma_1^z
+
h_+ \sigma_L^z ,
$$

那么对应的是 diagonal $K$-matrices。这种情况下 ABA 结构基本不变，只是本征值和 Bethe 方程多了边界反射因子。

---

## 1. Diagonal $K$-matrices

一种方便的归一化是取

$$
K_-(u)
=
\begin{pmatrix}
k_1^-(u)&0\\
0&k_2^-(u)
\end{pmatrix},
$$

其中

$$
\boxed{
k_1^-(u)
=
\frac{\sin(\xi_-+u)}{\sin\xi_-\,\cos u},
\qquad
k_2^-(u)
=
\frac{\sin(\xi_- -u)}{\sin\xi_-\,\cos u}.
}
$$

右边界取 dual diagonal solution：

$$
K_+(u)
=
\begin{pmatrix}
k_1^+(u)&0\\
0&k_2^+(u)
\end{pmatrix},
$$

其中

$$
\boxed{
k_1^+(u)
=
\frac{\sin(\xi_+-u-\eta)}{\cos(u+\eta)},
\qquad
k_2^+(u)
=
\frac{\sin(\xi_++u+\eta)}{\cos(u+\eta)}.
}
$$

这个归一化的好处是，当

$$
\xi_- = \xi_+ = \frac{\pi}{2}
$$

时，

$$
K_-(u)=K_+(u)=I,
$$

回到自由边界。

在这个 convention 下，边界场和 $\xi_\pm$ 的关系是

$$
\boxed{
h_- = \sin\eta\,\cot\xi_-,
\qquad
h_+ = -\sin\eta\,\cot\xi_+ .
}
$$

所以零边界场对应

$$
h_-=h_+=0
\quad\Longleftrightarrow\quad
\xi_- = \xi_+ = \frac{\pi}{2}.
$$

---

## 2. Bethe vector 还成立吗？

成立，但仍然是开边界的双行 Bethe vector：

$$
\boxed{
|\{u_j\}\rangle
=
\mathcal B(u_1)\mathcal B(u_2)\cdots \mathcal B(u_N)|0\rangle .
}
$$

这里

$$
\mathcal U_a(u)
=
M_a(u)K_{-,a}(u)\widehat M_a(u)
=
\begin{pmatrix}
\mathcal A(u)&\mathcal B(u)\\
\mathcal C(u)&\mathcal D_{\rm raw}(u)
\end{pmatrix}_a .
$$

只要 $K_\pm$ 是 diagonal，reference state

$$
|0\rangle=|\uparrow\uparrow\cdots\uparrow\rangle
$$

仍然是合适的伪真空态，所以标准 ABA 可以直接做。

---

## 3. Transfer matrix 本征值

和自由边界一样，定义

$$
Q(u)
=
\prod_{j=1}^N
\sin(u-u_j)\sin(u+u_j+\eta).
$$

本征值仍然是两项 T-Q 形式：

$$
\boxed{
\Lambda(u)
=
\Lambda_1(u)\frac{Q(u-\eta)}{Q(u)}
+
\Lambda_2(u)\frac{Q(u+\eta)}{Q(u)} .
}
$$

区别在于 $\Lambda_1,\Lambda_2$ 多了边界因子。

令

$$
r(u)=\frac{\sin\eta}{\sin(2u+\eta)}.
$$

则

$$
\boxed{
\Lambda_1(u)
=
\Bigl[
k_1^+(u)+r(u)k_2^+(u)
\Bigr]
k_1^-(u)\,
\sin^{2L}(u+\eta).
}
$$

$$
\boxed{
\Lambda_2(u)
=
k_2^+(u)
\Bigl[
k_2^-(u)-r(u)k_1^-(u)
\Bigr]
\sin^{2L}u .
}
$$

自由边界 $K_\pm=I$ 时，

$$
k_1^\pm=k_2^\pm=1,
$$

于是回到之前的

$$
\Lambda_1(u)
=
\left(1+\frac{\sin\eta}{\sin(2u+\eta)}\right)
\sin^{2L}(u+\eta),
$$

$$
\Lambda_2(u)
=
\left(1-\frac{\sin\eta}{\sin(2u+\eta)}\right)
\sin^{2L}u .
$$

---

## 4. 乘积型 Bethe 方程

Bethe 方程仍然来自 $\Lambda(u)$ 在 $u=u_j$ 处无极点：

$$
\Lambda_1(u_j)Q(u_j-\eta)
+
\Lambda_2(u_j)Q(u_j+\eta)
=0.
$$

整理后得到

$$
\boxed{
\left[
\frac{\sin(u_j+\eta)}{\sin u_j}
\right]^{2L}
R_-(u_j)R_+(u_j)
=
\prod_{k\neq j}
\frac{\sin(u_j-u_k+\eta)}
{\sin(u_j-u_k-\eta)}
\frac{\sin(u_j+u_k+2\eta)}
{\sin(u_j+u_k)} .
}
$$

其中左右边界反射因子是

$$
\boxed{
R_-(u)
=
\frac{\sin(\xi_-+u)}
{\sin(\xi_- -u-\eta)}
}
$$

和

$$
\boxed{
R_+(u)
=
\frac{\sin(\xi_+-u)}
{\sin(\xi_+ +u+\eta)} .
}
$$

这两个因子就是非零边界磁场对 Bethe 方程的修改。

当

$$
\xi_- = \xi_+ = \frac{\pi}{2}
$$

时，

$$
R_-(u)=R_+(u)=\frac{\cos u}{\cos(u+\eta)},
$$

于是就回到自由边界的方程：

$$
\left[
\frac{\sin(u_j+\eta)}{\sin u_j}
\right]^{2L}
\left[
\frac{\cos u_j}{\cos(u_j+\eta)}
\right]^2
=
\prod_{k\neq j}
\frac{\sin(u_j-u_k+\eta)}
{\sin(u_j-u_k-\eta)}
\frac{\sin(u_j+u_k+2\eta)}
{\sin(u_j+u_k)} .
$$

---

## 5. 实 rapidity 参数化

在 $|\Delta|<1$ 下继续取

$$
\boxed{
u_j=-\frac{\eta}{2}+\frac{i\alpha_j}{2},
\qquad \alpha_j\in\mathbb R.
}
$$

这样 Bethe 方程可以写成对数形式。

定义 bulk momentum 相位

$$
p_\eta(\alpha)
=
-i\log
\frac{\sin\left(-\frac{\eta}{2}+\frac{i\alpha}{2}+\eta\right)}
{\sin\left(-\frac{\eta}{2}+\frac{i\alpha}{2}\right)}.
$$

等价地，

$$
\boxed{
p_\eta(\alpha)
=
2\arctan
\left[
\frac{\tanh(\alpha/2)}{\tan(\eta/2)}
\right]
}
$$

差一个 $2\pi$ 分支 convention。

两体散射相位定义为

$$
\boxed{
\theta_\eta(\alpha)
=
-i\log
\frac{\sin\left(\frac{i\alpha}{2}+\eta\right)}
{\sin\left(\frac{i\alpha}{2}-\eta\right)} .
}
$$

也可以写成

$$
\boxed{
\theta_\eta(\alpha)
=
2\arctan
\left[
\frac{\tanh(\alpha/2)}{\tan\eta}
\right]
}
$$

同样需要按分支连续化处理。

边界相位定义为

$$
\boxed{
\varphi_-(\alpha)
=
-i\log R_-\left(-\frac{\eta}{2}+\frac{i\alpha}{2}\right),
}
$$

$$
\boxed{
\varphi_+(\alpha)
=
-i\log R_+\left(-\frac{\eta}{2}+\frac{i\alpha}{2}\right).
}
$$

更显式地，

$$
\boxed{
\varphi_-(\alpha)
=
-i\log
\frac{
\sin\left(\xi_- -\frac{\eta}{2}+\frac{i\alpha}{2}\right)
}{
\sin\left(\xi_- -\frac{\eta}{2}-\frac{i\alpha}{2}\right)
}
}
$$

因此

$$
\boxed{
\varphi_-(\alpha)
=
2\arctan
\left[
\cot\left(\xi_- -\frac{\eta}{2}\right)
\tanh\frac{\alpha}{2}
\right].
}
$$

同理，

$$
\boxed{
\varphi_+(\alpha)
=
-i\log
\frac{
\sin\left(\xi_+ +\frac{\eta}{2}-\frac{i\alpha}{2}\right)
}{
\sin\left(\xi_+ +\frac{\eta}{2}+\frac{i\alpha}{2}\right)
}
}
$$

所以

$$
\boxed{
\varphi_+(\alpha)
=
-2\arctan
\left[
\cot\left(\xi_+ +\frac{\eta}{2}\right)
\tanh\frac{\alpha}{2}
\right].
}
$$

这些 arctan 形式在数值上最好用 `atan2` 实现，以免分支跳跃。

---

## 6. 对数型 Bethe 方程

于是开边界、非零纵向边界场的 logarithmic BE 是

$$
\boxed{
2L\,p_\eta(\alpha_j)
+
\varphi_-(\alpha_j)
+
\varphi_+(\alpha_j)
-
\sum_{k\neq j}
\left[
\theta_\eta(\alpha_j-\alpha_k)
+
\theta_\eta(\alpha_j+\alpha_k)
\right]
=
2\pi I_j .
}
$$

这里 $I_j$ 仍然使用开边界 reduced convention：

$$
\boxed{
1\le I_1<I_2<\cdots<I_N.
}
$$

基态通常取

$$
\boxed{
I_j=j,\qquad j=1,\dots,N.
}
$$

非零边界磁场不会把 $I_j$ 改成 PBC 那种对称取值；它只是通过 $\varphi_\pm(\alpha)$ 改变 roots 的位置。

---

## 7. 能量公式

能量仍然由

$$
E
=
\sin\eta\,
\frac{d}{du}\log\Lambda(u)\bigg|_{u=0}
-
L\cos\eta
$$

给出。

在 $u=0$ 处，$\Lambda_2(u)\propto \sin^{2L}u$，所以一阶导数仍然不贡献。只需要看

$$
\Lambda_1(u)\frac{Q(u-\eta)}{Q(u)}.
$$

计算后得到

$$
\boxed{
E
=
(L-1)\cos\eta
+
\sin\eta(\cot\xi_- - \cot\xi_+)
-
\sum_{j=1}^N
\frac{4\sin^2\eta}
{\cosh\alpha_j-\cos\eta}.
}
$$

用边界场

$$
h_- = \sin\eta\,\cot\xi_-,
\qquad
h_+ = -\sin\eta\,\cot\xi_+
$$

表示，就是

$$
\boxed{
E
=
(L-1)\Delta
+
h_-+h_+
-
\sum_{j=1}^N
\frac{4(1-\Delta^2)}
{\cosh\alpha_j-\Delta}.
}
$$

注意边界场对能量有两种影响：

第一，显式给出边界参考能量

$$
h_-+h_+.
$$

第二，通过 Bethe 方程中的边界相位

$$
\varphi_-(\alpha),\qquad \varphi_+(\alpha)
$$

改变 Bethe roots $\alpha_j$，从而间接改变求和项。

---

## 8. 总结

非零纵向边界磁场的处理方式是：

$$
K_\pm=I
\quad\longrightarrow\quad
K_\pm(u)\ \text{取 diagonal reflection matrices}.
$$

本征值变成

$$
\boxed{
\Lambda(u)
=
\Lambda_1(u)\frac{Q(u-\eta)}{Q(u)}
+
\Lambda_2(u)\frac{Q(u+\eta)}{Q(u)}
}
$$

其中

$$
\boxed{
\Lambda_1(u)
=
\Bigl[
k_1^+(u)+\frac{\sin\eta}{\sin(2u+\eta)}k_2^+(u)
\Bigr]
k_1^-(u)\sin^{2L}(u+\eta),
}
$$

$$
\boxed{
\Lambda_2(u)
=
k_2^+(u)
\Bigl[
k_2^-(u)-\frac{\sin\eta}{\sin(2u+\eta)}k_1^-(u)
\Bigr]\sin^{2L}u.
}
$$

Bethe 方程变成

$$
\boxed{
\left[
\frac{\sin(u_j+\eta)}{\sin u_j}
\right]^{2L}
R_-(u_j)R_+(u_j)
=
\prod_{k\neq j}
\frac{\sin(u_j-u_k+\eta)}
{\sin(u_j-u_k-\eta)}
\frac{\sin(u_j+u_k+2\eta)}
{\sin(u_j+u_k)} .
}
$$

能量是

$$
\boxed{
E
=
(L-1)\Delta
+
h_-+h_+
-
\sum_{j=1}^N
\frac{4(1-\Delta^2)}
{\cosh\alpha_j-\Delta}.
}
$$

其中

$$
\boxed{
h_- = \sin\eta\,\cot\xi_-,
\qquad
h_+ = -\sin\eta\,\cot\xi_+ .
}
$$

---

# 二、$\Delta>1$ 的情形

现在考虑

$$
\Delta>1,\qquad \Delta=\cosh\gamma,\qquad \gamma>0,
$$

并且只考虑纵向边界磁场：

$$
H
=
\sum_{k=1}^{L-1}
\left(
\sigma_k^x\sigma_{k+1}^x
+
\sigma_k^y\sigma_{k+1}^y
+
\Delta\sigma_k^z\sigma_{k+1}^z
\right)
+
h_-\sigma_1^z
+
h_+\sigma_L^z .
$$

如果边界场是 $x,y$ 方向，$K_\pm$ 非对角，普通 ABA 会失效；这里先只讨论 diagonal $K_\pm$。

---

## 1. 参数化

$\Delta>1$ 使用双曲函数权重：

$$
\boxed{
a(u)=\sinh(u+\gamma),\qquad
b(u)=\sinh u,\qquad
c=\sinh\gamma .
}
$$

物理 roots 取为

$$
\boxed{
u_j=-\frac{\gamma}{2}+\frac{i\alpha_j}{2},
\qquad
\alpha_j\in\mathbb R .
}
$$

此时

$$
\frac{a(u_j)}{b(u_j)}
=
\frac{\sinh(u_j+\gamma)}{\sinh u_j}
=
\frac{\sin\frac{\alpha_j-i\gamma}{2}}
{\sin\frac{\alpha_j+i\gamma}{2}},
$$

是纯相位。

---

## 2. Diagonal $K$-matrices

取

$$
K_-(u)
=
\begin{pmatrix}
k_1^-(u)&0\\
0&k_2^-(u)
\end{pmatrix},
$$

其中

$$
\boxed{
k_1^-(u)=\sinh(\xi_-+u),
\qquad
k_2^-(u)=\sinh(\xi_--u).
}
$$

右边界取 dual solution：

$$
K_+(u)
=
\begin{pmatrix}
k_1^+(u)&0\\
0&k_2^+(u)
\end{pmatrix},
$$

其中

$$
\boxed{
k_1^+(u)=\sinh(\xi_+-u-\gamma),
\qquad
k_2^+(u)=\sinh(\xi_++u+\gamma).
}
$$

整体标量归一化可以任意乘，不影响 Bethe roots；只会改变 transfer matrix 的整体标量因子。为了和 Hamiltonian 常数项完全对齐，要和你定义 $T(u)$ 时的 $K_\pm$ 归一化保持一致。

边界场和 $\xi_\pm$ 的关系是

$$
\boxed{
h_- = \sinh\gamma\,\coth\xi_-,
\qquad
h_+ = -\sinh\gamma\,\coth\xi_+ .
}
$$

自由边界 $h_-=h_+=0$ 可以取

$$
\boxed{
\xi_- = \xi_+ = \frac{i\pi}{2}.
}
$$

因为

$$
\sinh\left(\frac{i\pi}{2}+u\right)=i\cosh u,
$$

所以这个时候 $K_\pm(u)$ 只差一个标量因子等价于 $I$。

如果想让边界场参数显式为实数，可以写

$$
\xi_\pm=\frac{i\pi}{2}+\zeta_\pm ,
$$

则

$$
\coth\left(\frac{i\pi}{2}+\zeta\right)=\tanh\zeta,
$$

因此

$$
\boxed{
h_- = \sinh\gamma\,\tanh\zeta_-,
\qquad
h_+ = -\sinh\gamma\,\tanh\zeta_+ .
}
$$

---

## 3. Bethe vector

只要 $K_\pm$ 是 diagonal，reference state 仍然是

$$
|0\rangle=|\uparrow\uparrow\cdots\uparrow\rangle .
$$

开边界 Bethe vector 仍然是双行形式：

$$
\boxed{
|\{u_j\}\rangle
=
\mathcal B(u_1)\mathcal B(u_2)\cdots\mathcal B(u_N)|0\rangle .
}
$$

其中

$$
\mathcal U_a(u)
=
M_a(u)K_{-,a}(u)\widehat M_a(u).
$$

---

## 4. 最大本征值 / Bethe 本征值

定义

$$
\boxed{
Q(u)
=
\prod_{j=1}^N
\sinh(u-u_j)\sinh(u+u_j+\gamma).
}
$$

本征值仍然是两项 T-Q 形式：

$$
\boxed{
\Lambda(u)
=
\Lambda_1(u)\frac{Q(u-\gamma)}{Q(u)}
+
\Lambda_2(u)\frac{Q(u+\gamma)}{Q(u)} .
}
$$

其中令

$$
r(u)=\frac{\sinh\gamma}{\sinh(2u+\gamma)}.
$$

则

$$
\boxed{
\Lambda_1(u)
=
\left[
k_1^+(u)+r(u)k_2^+(u)
\right]
k_1^-(u)\,
\sinh^{2L}(u+\gamma).
}
$$

$$
\boxed{
\Lambda_2(u)
=
k_2^+(u)
\left[
k_2^-(u)-r(u)k_1^-(u)
\right]
\sinh^{2L}u .
}
$$

这和 $|\Delta|<1$ 的公式完全平行，只是

$$
\sin\rightarrow\sinh,
\qquad
\eta\rightarrow\gamma .
$$

当 $\xi_\pm=i\pi/2$ 时，$K_\pm\sim I$，这就回到前面自由边界的

$$
\Lambda_1(u)
=
\left(1+\frac{\sinh\gamma}{\sinh(2u+\gamma)}\right)
\sinh^{2L}(u+\gamma),
$$

$$
\Lambda_2(u)
=
\left(1-\frac{\sinh\gamma}{\sinh(2u+\gamma)}\right)
\sinh^{2L}u
$$

的形式，差别只在无关的整体标量归一化。

---

## 5. 乘积型 Bethe 方程

Bethe 方程仍然来自

$$
\Lambda_1(u_j)Q(u_j-\gamma)
+
\Lambda_2(u_j)Q(u_j+\gamma)=0 .
$$

整理后得到

$$
\boxed{
\left[
\frac{\sinh(u_j+\gamma)}
{\sinh u_j}
\right]^{2L}
R_-(u_j)R_+(u_j)
=
\prod_{k\neq j}
\frac{\sinh(u_j-u_k+\gamma)}
{\sinh(u_j-u_k-\gamma)}
\frac{\sinh(u_j+u_k+2\gamma)}
{\sinh(u_j+u_k)} .
}
$$

其中边界反射因子为

$$
\boxed{
R_-(u)
=
\frac{\sinh(\xi_-+u)}
{\sinh(\xi_- -u-\gamma)}
}
$$

和

$$
\boxed{
R_+(u)
=
\frac{\sinh(\xi_+-u)}
{\sinh(\xi_+ +u+\gamma)} .
}
$$

自由边界时

$$
\xi_\pm=\frac{i\pi}{2},
$$

于是

$$
R_-(u)=R_+(u)
=
\frac{\cosh u}{\cosh(u+\gamma)}.
$$

所以自由边界 BE 退化为

$$
\left[
\frac{\sinh(u_j+\gamma)}
{\sinh u_j}
\right]^{2L}
\left[
\frac{\cosh u_j}{\cosh(u_j+\gamma)}
\right]^2
=
\prod_{k\neq j}
\frac{\sinh(u_j-u_k+\gamma)}
{\sinh(u_j-u_k-\gamma)}
\frac{\sinh(u_j+u_k+2\gamma)}
{\sinh(u_j+u_k)} ,
$$

这正是前面 $K_\pm=I$ 的结果。

---

## 6. 实 rapidity 形式

代入

$$
u_j=-\frac{\gamma}{2}+\frac{i\alpha_j}{2}.
$$

bulk momentum 因子为

$$
\frac{\sinh(u_j+\gamma)}{\sinh u_j}
=
\frac{
\sin\frac{\alpha_j-i\gamma}{2}
}{
\sin\frac{\alpha_j+i\gamma}{2}
}.
$$

两体散射因子为

$$
S_\gamma(\alpha)
=
\frac{
\sin\frac{\alpha-2i\gamma}{2}
}{
\sin\frac{\alpha+2i\gamma}{2}
}.
$$

边界因子变成

$$
R_-(\alpha)
=
R_-\left(-\frac{\gamma}{2}+\frac{i\alpha}{2}\right),
$$

即

$$
\boxed{
R_-(\alpha)
=
\frac{
\sinh\left(\xi_--\frac{\gamma}{2}+\frac{i\alpha}{2}\right)
}{
\sinh\left(\xi_--\frac{\gamma}{2}-\frac{i\alpha}{2}\right)
}.
}
$$

右边界为

$$
\boxed{
R_+(\alpha)
=
\frac{
\sinh\left(\xi_++\frac{\gamma}{2}-\frac{i\alpha}{2}\right)
}{
\sinh\left(\xi_++\frac{\gamma}{2}+\frac{i\alpha}{2}\right)
}.
}
$$

---

## 7. 对数型 Bethe 方程

定义

$$
\boxed{
p_\gamma(\alpha)
=
-i\log
\frac{
\sin\frac{\alpha-i\gamma}{2}
}{
\sin\frac{\alpha+i\gamma}{2}
}
}
$$

$$
\boxed{
\theta_\gamma(\alpha)
=
-i\log
\frac{
\sin\frac{\alpha-2i\gamma}{2}
}{
\sin\frac{\alpha+2i\gamma}{2}
}
}
$$

以及边界相位

$$
\boxed{
\varphi_-(\alpha)
=
-i\log R_-(\alpha),
\qquad
\varphi_+(\alpha)
=
-i\log R_+(\alpha).
}
$$

那么 logarithmic BE 是

$$
\boxed{
2L\,p_\gamma(\alpha_j)
+
\varphi_-(\alpha_j)
+
\varphi_+(\alpha_j)
-
\sum_{k\neq j}
\left[
\theta_\gamma(\alpha_j-\alpha_k)
+
\theta_\gamma(\alpha_j+\alpha_k)
\right]
=
2\pi I_j .
}
$$

这里仍然使用 open-chain reduced convention：

$$
\boxed{
1\le I_1<I_2<\cdots<I_N.
}
$$

基态一般取

$$
\boxed{
I_j=j,\qquad j=1,\dots,N.
}
$$

---

## 8. arctan / atan2 形式

bulk 相位可以写成

$$
\boxed{
p_\gamma(\alpha)
=
\pi+
2\arctan
\left[
\frac{\tan(\alpha/2)}{\tanh(\gamma/2)}
\right]
\quad \bmod 2\pi .
}
$$

去掉常数 $\pi$ 后，可以用

$$
\boxed{
\Theta_1(\alpha)
=
2\arctan
\left[
\frac{\tan(\alpha/2)}{\tanh(\gamma/2)}
\right].
}
$$

两体散射相位可以写成

$$
\boxed{
\Theta_2(\alpha)
=
2\arctan
\left[
\frac{\tan(\alpha/2)}{\tanh\gamma}
\right].
}
$$

边界相位如果令

$$
\xi_- = \frac{i\pi}{2}+\zeta_-,
\qquad
\xi_+ = \frac{i\pi}{2}+\zeta_+,
$$

则

$$
R_-(\alpha)
=
\frac{
\cosh\left(\zeta_--\frac{\gamma}{2}+\frac{i\alpha}{2}\right)
}{
\cosh\left(\zeta_--\frac{\gamma}{2}-\frac{i\alpha}{2}\right)
},
$$

$$
R_+(\alpha)
=
\frac{
\cosh\left(\zeta_++\frac{\gamma}{2}-\frac{i\alpha}{2}\right)
}{
\cosh\left(\zeta_++\frac{\gamma}{2}+\frac{i\alpha}{2}\right)
}.
$$

因此可以写成

$$
\boxed{
\varphi_-(\alpha)
=
2\arctan
\left[
\tan\frac{\alpha}{2}
\tanh\left(\zeta_- -\frac{\gamma}{2}\right)
\right],
}
$$

$$
\boxed{
\varphi_+(\alpha)
=
-2\arctan
\left[
\tan\frac{\alpha}{2}
\tanh\left(\zeta_+ +\frac{\gamma}{2}\right)
\right].
}
$$

数值上依然建议用 `atan2`，因为 $\tan(\alpha/2)$ 在 $\alpha=\pi$ 附近会跳支。

---

## 9. 能量公式

能量仍然由

$$
E
=
\sinh\gamma
\frac{d}{du}\log\Lambda(u)\bigg|_{u=0}
-
L\cosh\gamma
$$

给出。

在 $u=0$ 处，

$$
\Lambda_2(u)\propto \sinh^{2L}u,
$$

所以一阶导数仍然不贡献。只需要看

$$
\Lambda_1(u)\frac{Q(u-\gamma)}{Q(u)}.
$$

最终能量为

$$
\boxed{
E
=
(L-1)\cosh\gamma
+
\sinh\gamma\left(\coth\xi_- -\coth\xi_+\right)
-
\sum_{j=1}^{N}
\frac{4\sinh^2\gamma}
{\cosh\gamma-\cos\alpha_j}.
}
$$

用边界场

$$
h_- = \sinh\gamma\,\coth\xi_-,
\qquad
h_+ = -\sinh\gamma\,\coth\xi_+
$$

表示，就是

$$
\boxed{
E
=
(L-1)\Delta
+
h_-+h_+
-
\sum_{j=1}^{N}
\frac{4(\Delta^2-1)}
{\Delta-\cos\alpha_j}.
}
$$

这里有两类边界效应：

第一，显式参考态边界能量

$$
h_-+h_+.
$$

第二，边界相位

$$
\varphi_-(\alpha),\qquad \varphi_+(\alpha)
$$

改变 Bethe roots $\alpha_j$，从而间接改变求和项。

---

## 10. 最后总结

$\Delta>1$ 非零纵向边界场的主要结果是：

$$
\boxed{
\Delta=\cosh\gamma,
\qquad
u_j=-\frac{\gamma}{2}+\frac{i\alpha_j}{2}.
}
$$

乘积型 BE：

$$
\boxed{
\left[
\frac{\sinh(u_j+\gamma)}
{\sinh u_j}
\right]^{2L}
R_-(u_j)R_+(u_j)
=
\prod_{k\neq j}
\frac{\sinh(u_j-u_k+\gamma)}
{\sinh(u_j-u_k-\gamma)}
\frac{\sinh(u_j+u_k+2\gamma)}
{\sinh(u_j+u_k)} .
}
$$

其中

$$
\boxed{
R_-(u)
=
\frac{\sinh(\xi_-+u)}
{\sinh(\xi_- -u-\gamma)},
\qquad
R_+(u)
=
\frac{\sinh(\xi_+-u)}
{\sinh(\xi_+ +u+\gamma)} .
}
$$

对数型 BE：

$$
\boxed{
2L\,p_\gamma(\alpha_j)
+
\varphi_-(\alpha_j)
+
\varphi_+(\alpha_j)
-
\sum_{k\neq j}
\left[
\theta_\gamma(\alpha_j-\alpha_k)
+
\theta_\gamma(\alpha_j+\alpha_k)
\right]
=
2\pi I_j .
}
$$

能量：

$$
\boxed{
E
=
(L-1)\Delta
+
h_-+h_+
-
\sum_{j=1}^{N}
\frac{4(\Delta^2-1)}
{\Delta-\cos\alpha_j}.
}
$$

其中

$$
\boxed{
h_- = \sinh\gamma\,\coth\xi_-,
\qquad
h_+ = -\sinh\gamma\,\coth\xi_+ .
}
$$

---

# 三、$\Delta=1$ 的 XXX 情形

现在考虑 $\Delta=1$ 的 XXX 开链，并加入纵向边界磁场：

$$
H
=
\sum_{k=1}^{L-1}
\left(
\sigma_k^x\sigma_{k+1}^x+
\sigma_k^y\sigma_{k+1}^y+
\sigma_k^z\sigma_{k+1}^z
\right)
+
h_-\sigma_1^z
+
h_+\sigma_L^z .
$$

这里只讨论 $z$ 方向边界场，也就是 diagonal $K_\pm$。如果边界场有 $x,y$ 分量，就不是普通 diagonal ABA，需要 off-diagonal Bethe ansatz。

---

## 1. XXX 参数化

$\Delta=1$ 是 $\Delta=\cosh\gamma$ 的 $\gamma\to0$ rational limit。取

$$
a(u)=u+1,\qquad b(u)=u,\qquad c=1 .
$$

Bethe roots 取

$$
\boxed{
u_j=-\frac{1}{2}+\frac{i\alpha_j}{2},
\qquad \alpha_j\in\mathbb R .
}
$$

这样

$$
\frac{u_j+1}{u_j}
=
-\frac{1+i\alpha_j}{1-i\alpha_j}
$$

是纯相位。

---

## 2. Diagonal $K$-matrices

取左边界

$$
K_-(u)=
\begin{pmatrix}
k_1^-(u)&0\\
0&k_2^-(u)
\end{pmatrix},
$$

其中

$$
\boxed{
k_1^-(u)=\xi_-+u,
\qquad
k_2^-(u)=\xi_- -u .
}
$$

右边界取 dual solution：

$$
K_+(u)=
\begin{pmatrix}
k_1^+(u)&0\\
0&k_2^+(u)
\end{pmatrix},
$$

其中

$$
\boxed{
k_1^+(u)=\xi_+-u-1,
\qquad
k_2^+(u)=\xi_++u+1 .
}
$$

这个 convention 下，边界场和 $\xi_\pm$ 的关系是

$$
\boxed{
h_- = \frac{1}{\xi_-},
\qquad
h_+ = -\frac{1}{\xi_+}.
}
$$

自由边界对应

$$
\boxed{
\xi_-\to\infty,
\qquad
\xi_+\to\infty .
}
$$

这和 $|\Delta|<1$ 里某个有限 $\xi=\pi/2$ 对应自由边界不同；在 rational limit 里自由边界是 $\xi_\pm\to\infty$。

---

## 3. Bethe vector

只要 $K_\pm$ 是 diagonal，伪真空态仍然是

$$
|0\rangle=|\uparrow\uparrow\cdots\uparrow\rangle .
$$

Bethe vector 仍然是开边界双行形式：

$$
\boxed{
|\{u_j\}\rangle
=
\mathcal B(u_1)\mathcal B(u_2)\cdots \mathcal B(u_N)|0\rangle .
}
$$

这里

$$
\mathcal U_a(u)
=
M_a(u)K_{-,a}(u)\widehat M_a(u).
$$

---

## 4. Transfer matrix 本征值

定义

$$
\boxed{
Q(u)
=
\prod_{j=1}^{N}
(u-u_j)(u+u_j+1).
}
$$

则本征值仍然是两项 T-Q 形式：

$$
\boxed{
\Lambda(u)
=
\Lambda_1(u)\frac{Q(u-1)}{Q(u)}
+
\Lambda_2(u)\frac{Q(u+1)}{Q(u)} .
}
$$

其中

$$
\boxed{
\Lambda_1(u)
=
\left[
k_1^+(u)+\frac{1}{2u+1}k_2^+(u)
\right]
k_1^-(u)(u+1)^{2L},
}
$$

$$
\boxed{
\Lambda_2(u)
=
k_2^+(u)
\left[
k_2^-(u)-\frac{1}{2u+1}k_1^-(u)
\right]
u^{2L}.
}
$$

把 $k_i^\pm$ 展开，就是

$$
\boxed{
\Lambda_1(u)
=
\left[
\xi_+-u-1
+
\frac{\xi_++u+1}{2u+1}
\right]
(\xi_-+u)(u+1)^{2L},
}
$$

$$
\boxed{
\Lambda_2(u)
=
(\xi_++u+1)
\left[
\xi_- -u
-
\frac{\xi_-+u}{2u+1}
\right]
u^{2L}.
}
$$

当 $\xi_\pm\to\infty$ 时，忽略整体标量因子，就回到自由边界结果：

$$
\Lambda_1(u)
\propto
\left(1+\frac{1}{2u+1}\right)(u+1)^{2L},
$$

$$
\Lambda_2(u)
\propto
\left(1-\frac{1}{2u+1}\right)u^{2L}.
$$

---

## 5. 乘积型 Bethe 方程

Bethe 方程来自 $\Lambda(u)$ 在 $u=u_j$ 处无极点：

$$
\Lambda_1(u_j)Q(u_j-1)
+
\Lambda_2(u_j)Q(u_j+1)=0 .
$$

展开得到

$$
\boxed{
\left[
\frac{u_j+1}{u_j}
\right]^{2L}
R_-(u_j)R_+(u_j)
=
\prod_{k\neq j}
\frac{u_j-u_k+1}{u_j-u_k-1}
\frac{u_j+u_k+2}{u_j+u_k}.
}
$$

其中边界反射因子是

$$
\boxed{
R_-(u)
=
\frac{\xi_-+u}{\xi_- -u-1},
}
$$

$$
\boxed{
R_+(u)
=
\frac{\xi_+-u}{\xi_+ +u+1}.
}
$$

自由边界 $\xi_\pm\to\infty$ 时，

$$
R_-(u)R_+(u)\to1,
$$

所以回到自由开链 XXX 的 BE：

$$
\left[
\frac{u_j+1}{u_j}
\right]^{2L}
=
\prod_{k\neq j}
\frac{u_j-u_k+1}{u_j-u_k-1}
\frac{u_j+u_k+2}{u_j+u_k}.
$$

---

## 6. 实 rapidity 形式

代入

$$
u_j=-\frac{1}{2}+\frac{i\alpha_j}{2}.
$$

bulk 相位因子为

$$
\frac{u_j+1}{u_j}
=
-\frac{1+i\alpha_j}{1-i\alpha_j}.
$$

开边界中它出现 $2L$ 次幂，所以负号不影响方程。

边界反射因子变成

$$
R_-(\alpha)
=
\frac{\xi_- -\frac12+\frac{i\alpha}{2}}
{\xi_- -\frac12-\frac{i\alpha}{2}},
$$

$$
R_+(\alpha)
=
\frac{\xi_+ +\frac12-\frac{i\alpha}{2}}
{\xi_+ +\frac12+\frac{i\alpha}{2}}.
$$

因此实 rapidity 下的乘积 BE 是

$$
\boxed{
\left[
\frac{1+i\alpha_j}{1-i\alpha_j}
\right]^{2L}
R_-(\alpha_j)R_+(\alpha_j)
=
\prod_{k\neq j}
\frac{
1+\frac{i}{2}(\alpha_j-\alpha_k)
}{
1-\frac{i}{2}(\alpha_j-\alpha_k)
}
\frac{
1+\frac{i}{2}(\alpha_j+\alpha_k)
}{
1-\frac{i}{2}(\alpha_j+\alpha_k)
}.
}
$$

---

## 7. 对数型 Bethe 方程

定义

$$
\boxed{
p(\alpha)=2\arctan\alpha ,
}
$$

$$
\boxed{
\theta(\alpha)=2\arctan\frac{\alpha}{2}.
}
$$

边界相位定义为

$$
\varphi_-(\alpha)
=
-i\log R_-(\alpha),
\qquad
\varphi_+(\alpha)
=
-i\log R_+(\alpha).
$$

显式地，

$$
\boxed{
\varphi_-(\alpha)
=
2\arctan
\frac{\alpha}{2\xi_- -1},
}
$$

$$
\boxed{
\varphi_+(\alpha)
=
-2\arctan
\frac{\alpha}{2\xi_+ +1}.
}
$$

于是 logarithmic BE 是

$$
\boxed{
2L\,p(\alpha_j)
+
\varphi_-(\alpha_j)
+
\varphi_+(\alpha_j)
-
\sum_{k\neq j}
\left[
\theta(\alpha_j-\alpha_k)
+
\theta(\alpha_j+\alpha_k)
\right]
=
2\pi I_j .
}
$$

也就是

$$
\boxed{
4L\arctan\alpha_j
+
2\arctan\frac{\alpha_j}{2\xi_- -1}
-
2\arctan\frac{\alpha_j}{2\xi_+ +1}
-
2\sum_{k\neq j}
\left[
\arctan\frac{\alpha_j-\alpha_k}{2}
+
\arctan\frac{\alpha_j+\alpha_k}{2}
\right]
=
2\pi I_j .
}
$$

这里仍然是开边界 reduced convention，因此独立 roots 只取一侧，Bethe numbers 取正整数：

$$
\boxed{
1\le I_1<I_2<\cdots<I_N.
}
$$

基态通常取

$$
\boxed{
I_j=j,\qquad j=1,\dots,N.
}
$$

---

## 8. 能量公式

能量由

$$
E
=
\frac{d}{du}\log\Lambda(u)\bigg|_{u=0}
-
L
$$

给出。在 $u=0$ 处，

$$
\Lambda_2(u)\propto u^{2L},
$$

而且因为括号里还有一阶零点，所以一阶导数不贡献。仍然只需要看

$$
\Lambda_1(u)\frac{Q(u-1)}{Q(u)}.
$$

计算得到

$$
\frac{d}{du}\log\Lambda(u)\bigg|_{u=0}
=
2L-1
+
\frac{1}{\xi_-}
-
\frac{1}{\xi_+}
+
2\sum_{j=1}^N
\left[
\frac{1}{u_j}
-
\frac{1}{u_j+1}
\right].
$$

因此

$$
E
=
L-1
+
\frac{1}{\xi_-}
-
\frac{1}{\xi_+}
+
2\sum_{j=1}^{N}
\frac{1}{u_j(u_j+1)}.
$$

代入

$$
u_j=-\frac12+\frac{i\alpha_j}{2},
$$

有

$$
u_j(u_j+1)
=
-\frac{1+\alpha_j^2}{4}.
$$

所以

$$
\boxed{
E
=
L-1
+
\frac{1}{\xi_-}
-
\frac{1}{\xi_+}
-
\sum_{j=1}^{N}
\frac{8}{1+\alpha_j^2}.
}
$$

用边界场

$$
h_- = \frac{1}{\xi_-},
\qquad
h_+ = -\frac{1}{\xi_+}
$$

表示，就是

$$
\boxed{
E
=
L-1
+
h_-+h_+
-
\sum_{j=1}^{N}
\frac{8}{1+\alpha_j^2}.
}
$$

边界场的作用有两部分：一部分是显式的

$$
h_-+h_+,
$$

另一部分是通过 Bethe 方程中的边界相位 $\varphi_\pm(\alpha)$ 改变 roots $\alpha_j$。

---

## 9. 最简总结

$$
\boxed{
u_j=-\frac12+\frac{i\alpha_j}{2}}
$$

$$
\boxed{
\left[
\frac{u_j+1}{u_j}
\right]^{2L}
\frac{\xi_-+u_j}{\xi_- -u_j-1}
\frac{\xi_+-u_j}{\xi_+ +u_j+1}
=
\prod_{k\neq j}
\frac{u_j-u_k+1}{u_j-u_k-1}
\frac{u_j+u_k+2}{u_j+u_k}.
}
$$

对数形式：

$$
\boxed{
2L\,p(\alpha_j)
+
\varphi_-(\alpha_j)
+
\varphi_+(\alpha_j)
-
\sum_{k\neq j}
\left[
\theta(\alpha_j-\alpha_k)
+
\theta(\alpha_j+\alpha_k)
\right]
=
2\pi I_j .
}
$$

其中

$$
p(\alpha)=2\arctan\alpha,
\qquad
\theta(\alpha)=2\arctan\frac{\alpha}{2},
$$

$$
\varphi_-(\alpha)=2\arctan\frac{\alpha}{2\xi_- -1},
\qquad
\varphi_+(\alpha)=-2\arctan\frac{\alpha}{2\xi_+ +1}.
$$

能量：

$$
\boxed{
E
=
L-1+h_-+h_+
-
\sum_{j=1}^{N}\frac{8}{1+\alpha_j^2}.
}
$$
