整理 infinite xy 模型演化的关系参数

哈密顿量：
$$
    H = \sum_i J_x \sigma^x_i \sigma^x_{i+1} + J_y \sigma^y_i \sigma^y_{i+1} + \sum_i h_z \sigma^z_i
$$

参数变化：
$$
    \begin{cases}
        \lambda = (J_{x} + J_{y})/2 \\
        \gamma = (J_{x} - J_{y})/2
    \end{cases}
$$

以及：
$$
    \begin{cases}
        \theta_{k} = \operatorname{angle}(2 \gamma \sin k, h_{z} - 2 \lambda \cos k) \\
        \omega_{k} = 2\sqrt{4 \gamma^2 \sin^2 k + (h_{z} - 2\lambda \cos k)^2 }
    \end{cases}
$$
其中 ${ \operatorname{angle} }$ 表示角度，对应 python 中的 `np.arctan2`

定义了这些参数后，还需要将观测量和初态做对应变化：

$$
    \begin{cases}
        \sigma_{i}^{x} = F_{i}(c_{i} + c_{i}^{\dagger}) \\
        \sigma_{i}^{y} = F_{i}(c_{i} - c_{i}^{\dagger}) \\
        \sigma_{i}^{z} = 2 c_{i}^{\dagger} c_{i} - 1
    \end{cases}
$$
其中：
$$
    F_{n} = \prod_{i=1}^{n - 1} (2 c_{n}^{\dagger} c_{n} - 1)
$$

观测量：${ A }$ 一定是 ${ (\sigma_{i}^{x}, \sigma_{i}^{y}, \sigma_{i}^{z}) }$ 的函数，通过上述变换，可以转换为 ${ (c_{i}, c_{i}^{\dagger}) }$ 的函数，即：
$$
    A = A_{\sigma}(\lbrace \sigma_{i}^{x}, \sigma_{i}^{y}, \sigma_{i}^{z} \rbrace )  \mapsto A_{c}(\lbrace c_{i}, c_{i}^{\dagger} \rbrace )
$$

想要计算 ${ A(t) }$ 就只用通过 ${ c_{i}(t), c_{i}^{\dagger}(t) }$ 带入即可。

而 ${ c_{i}, c_{i}^{\dagger} }$ 可以通过：
$$
    c_{i} = \frac{1}{\sqrt{N}} \sum_{k}^{} \mathrm{e}^{ - \mathrm{i}nk} f_{k}
$$

并且：
$$
    f_{k}(t) = \alpha_{k}(t) f_{k}(0) + \beta_{k}(t) f_{ - k}^{\dagger} (0)
$$
其中：
$$
    \begin{cases}
        \alpha_{k}(t) = \cos^2 (\theta_{k}/2) \mathrm{e}^{ - \mathrm{i}\omega_{k}t} + \sin^2 (\theta_{k}/2) \mathrm{e}^{\mathrm{i}\omega_{k}t} \\
        \beta_{k}(t) = - \sin \theta_{k} \sin(\omega_{k}t)
    \end{cases}
$$

其中 ${ k }$ 从 ${ - \pi }$ 取到 ${ \pi }$，近似来说，对于非常大的 ${ N }$，等间隔的取，对于 python 的 `ks = np.linspace(-np.pi, np.pi, N, endpoint=False)`

目标是计算：${ \braket{ \psi \vert A(t) \vert \psi }  }$，第二步时要将初态 ${ \left| \psi \right> }$ 转换成：${ \left| \psi \right> = B_{\psi} \left| 0 \right> }$ 的形式，这样观测值就变成：
$$
    \braket{ \psi \vert A(t) \vert \psi } = \braket{ 0 \vert B_{\psi}^{\dagger} A(t) B_{\psi}\vert 0 }
$$

${ B_{\phi} }$ 同样要用 ${ \lbrace c_{i} \rbrace  }$ 展开，再用 ${ \lbrace f_{k} \rbrace  }$ 展开。

## Examples

计算算符 ${ \sigma^{z}_{n} }$ 的观察值。

首先 ${ \sigma^{z}_{n} = 2 c_{n}^{\dagger} c_{n} - 1 }$

然后：
$$
\begin{align*}
    \sigma_{n}^{z} &= 2 c_{n}^{\dagger} c_{n} - 1 \\
    &= 2 \left( \frac{1}{\sqrt{N}} \sum_{k}^{} \mathrm{e}^{ - \mathrm{i}nk} f_{k} \right)^{\dagger} \left( \frac{1}{\sqrt{N}} \sum_{k}^{} \mathrm{e}^{ - \mathrm{i}nk} f_{k} \right) - 1 \\
    &= \frac{2}{N} \sum_{k_1k_2}^{} \mathrm{e}^{\mathrm{i}nk_{1}} \mathrm{e}^{ - \mathrm{i}n k_{2}} f_{k_1}^{\dagger} f_{k_2} - 1
\end{align*}
$$

因此，演化为：
$$
\begin{align*}
    \sigma_{n}^{z}(t) &= \frac{2}{N} \sum_{k_1k_2}^{} \mathrm{e}^{\mathrm{i}nk_{1}} \mathrm{e}^{ - \mathrm{i}n k_{2}} f_{k_1}^{\dagger}(t) f_{k_2}(t) - 1
\end{align*}
$$
考虑真空态的演化，那么 ${ B_{\psi} }$ 就是单位矩阵，那么：
$$
\begin{align*}
    \braket{ \sigma_{n}^{z}(t) } = \frac{2}{N} \sum_{k_1k_2}^{} \mathrm{e}^{\mathrm{i}nk_{1}} \mathrm{e}^{ - \mathrm{i}n k_{2}} \braket{ f_{k_1}^{\dagger}(t) f_{k_2}(t) }  - 1
\end{align*}
$$
而：
$$
\begin{align*}
    \braket{ f_{k_1}^{\dagger}(t) f_{k_2}(t) } &= 
    \braket{ (\alpha_{k_1}(t) f_{k_1} + \beta_{k_1}(t) f_{ - k_1}^{\dagger} )^{\dagger} (\alpha_{k_2}(t) f_{k_2} + \beta_{k_2}(t) f_{ - k_2}^{\dagger}) } \\
    &= \braket{ (\alpha_{k_1}(t)^* f_{k_1}^{\dagger} + \beta^*_{k_1}(t) f_{ - k_1}) (\alpha_{k_2}(t) f_{k_2} + \beta_{k_2}(t) f_{ - k_2}^{\dagger}) } \\
    &= \beta_{k_1}^* \beta_{k_2} \braket{ f_{ - k_1} f_{ - k_2}^{\dagger}  } = \beta_{k_1}^* \beta_{k_2} \delta_{k_1,k_2}
\end{align*}
$$
因而：
$$
\begin{align*}
    \braket{ \sigma_{n}^{z}(t) } &= \frac{2}{N} \sum_{k}^{} \beta_{k}^* \beta_{k} - 1 \\
    &= \frac{2}{N} \sum_{k}^{} \sin^2 \theta_{k} \sin^2 (\omega_{k}t) - 1 \\
    &= \frac{1}{\pi} \Delta k  \sum_{k}^{} \sin^2 \theta_{k} \sin^2 (\omega_{k}t) - 1 \\
    &= \frac{1}{\pi} \int_{ - \pi}^{\pi} \mathrm{d}k \, \sin^2 \theta_{k} \sin^2 (\omega_{k}t) - 1
\end{align*}
$$
在 python 中的代码是`1/np.pi * spi.quad(lambda k: np.sin(theta(k))**2 * np.sin(omega(k) * t)**2, -np.pi, np.pi)[0] - 1`

试图考虑 ${ t \to \infty }$ ：
$$
\begin{align*}
    \braket{ \sigma_{n}^{z}(t) } &= 2 \int_{ - \pi}^{\pi} \mathrm{d}k \, \sin^2 \theta_{k} \sin^2 (\omega_{k}t) - 1
\end{align*}
$$
取 ${ t \to \infty }$ 那么替换：${ \sin^2 \to 1/2 }$，那么：
$$
\begin{align*}
    \braket{ \sigma_{n}^{z}(t \to \infty) } &= \int_{ - \pi}^{\pi} \mathrm{d}k \, \sin^2 \theta_{k} - 1
\end{align*}
$$

## 画图
```python
import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
jx = 1
jy = 0
hz = 2
λ = (jx + jy) / 2
γ = (jx - jy) / 2
theta = lambda k: np.arctan2(2*γ*np.sin(k),  hz - 2*λ*np.cos(k))
omega = lambda k: 2 * np.sqrt((2*γ*np.sin(k))**2 + (hz - 2*λ*np.cos(k))**2)
ts = np.linspace(0, 10, 500)
sigma_zs = np.zeros_like(ts, dtype=np.float64)
for i, t in enumerate(ts):
    sigma_zs[i] = 1/np.pi * spi.quad(lambda k: np.sin(theta(k))**2 * np.sin(omega(k) * t)**2, -np.pi, np.pi)[0] - 1
plt.plot(ts, sigma_zs)
```