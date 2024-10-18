# 自由费米子

## 平移不变系统

平移不变(PBC)自由费米子哈密顿量:
$$
    H = \sum_{j = 1}^{L} (\lambda c_{j}c_{j + 1}^{\dagger} - \lambda^* c_{j}^{\dagger} c_{j + 1}) + (\gamma c_{j}c_{j + 1} - \gamma^*  c_{j}^{\dagger} c_{j + 1}^{\dagger} ) + h \sum_{j = 1}^{L} (c_{j}^{\dagger} c_{j} - c_{j}c_{j}^{\dagger} ),
$$

通过 JW 变换:
$$
    c_{j} \mapsto \sigma_{j}^{ -}\prod_{n=1}^{j - 1}\sigma_{n}^{z},\quad\quad 
    \sigma_{j}^{ -} \mapsto c_{j} \prod_{n = 1}^{j - 1} (2c_{n}^{\dagger} c_{n} - 1) 
$$

可以变换为自旋模型:
$$
\begin{align*}
    H =& \sum_{j = 1}^{L - 1} (J_{x} s_{j}^{x}s_{j + 1}^{x} + J_{y} s_{j}^{y}s_{j + 1}^{y} + J_{yx} s_{j}^{y}s_{j + 1}^{x} + J_{xy} s_{j}^{x}s_{j + 1}^{y} )
    + h_{z} \sum_{j = 1}^{L} s_{j}^{z}
    \\ & +
    \left( 
        \prod_{n = 2}^{L - 1} \sigma_{n}^{z} 
     \right)
    (
        J_{y} s_{1}^{x}s_{L}^{x} 
        + J_{x} s_{1}^{y}s_{L}^{y}
        - J_{yx} s_{1}^{y} s_{L}^{x}
        - J_{xy} s_{1}^{x} s_{L}^{y}
    ),
\end{align*}
$$
其中:
$$
    \begin{cases}
        \lambda = \frac{1}{4} \left( J_{x} + J_{y} \right) + \frac{\mathrm{i}}{4}\left( J_{yx}-J_{xy} \right) \\
        \gamma = \frac{1}{4} \left( J_{x} - J_{y} \right) + \frac{\mathrm{i}}{4}\left( J_{yx} + J_{xy} \right) \\
        h = \frac{1}{2}h_{z}
    \end{cases}
    \text{   或者   }
    \begin{cases}
        J_{x} = 2 \Re (\lambda + \gamma) \\
        J_{y} = 2 \Re (\lambda - \gamma) \\
        J_{yx} = 2 \Im (\lambda + \gamma) \\
        J_{xy} = - 2\Im (\lambda - \gamma) \\
        h_{z} = 2h
    \end{cases}
$$

通过 Fourier 变换 和 Bogoliubov 变换解自由费米子系统:
$$
    H = \sum_{k = 1}^{L} 
    \epsilon_{k} (f_{k}^{\dagger} f_{k} - f_{k} f_{k}^{\dagger}),
$$
表示, 任何一个激发 ${ k }$ 贡献 ${ \epsilon_{k} }$ 或 ${ - \epsilon_{k} }$ 的能量. 其中, 激发能为:
$$
\begin{align*}
    \epsilon_{k} = & \underbrace{| \lambda | \sin \chi \sin \left( \frac{2\pi}{L} k \right)}_{\eta_{k}}
    +
    \underbrace{\sqrt{ \left( h - | \lambda | \cos \chi \cos \left( \frac{2\pi}{L} k \right) \right)^2 + \left( | \gamma |  \sin \left( \frac{2\pi}{L} k \right) \right)^2 }}_{\omega_{k}}
\end{align*}
$$

### 基态能

(${ \eta_{k} }$ 求和为 0):
$$
\begin{align*}
    E_{g} = & - \sum_{k = 1}^{L} 
    \sqrt{ \left( h - | \lambda | \cos \chi \cos \left( \frac{2\pi}{L} k \right) \right)^2 + \left( | \gamma |  \sin \left( \frac{2\pi}{L} k \right) \right)^2 }
\end{align*}
$$

${ L \to \infty }$ 下, 平均能量为:
$$
\begin{align*}
    \lim_{L \to \infty } \frac{E_{g}}{L} = - \int_{0}^{1} \mathrm{d}\alpha \, 
    \sqrt{(h - | \lambda | \cos\chi \cos (2\pi \alpha) )^2 + ( | \gamma |  \sin (2\pi \alpha))^2 },
\end{align*}
$$

一些特殊情况下:
$$
\begin{align*}
    \lim_{L \to \infty } \frac{E_{g}}{L} 
    = 
    \begin{cases}
        \frac{ | \lambda | }{\pi} E(1 - \frac{\gamma^2}{\lambda^2 }) + \frac{ | \gamma |  }{\pi} E(1 - \frac{\lambda^2}{\gamma^2 } ), & \gamma \neq 0, \lambda \neq 0, h = 0;\\
        2 | \gamma | / \pi, & \lambda = h = 0;\\
        2 | \lambda | / \pi, & \gamma = h = 0;\\
        - 4h/\pi, & \lambda = \gamma = h
    \end{cases}
\end{align*}
$$
其中 ${ E(k) }$ 是完全椭圆积分.

### 自由能

$$
\begin{align*}
    f &= - \frac{\ln Z}{\beta L}
    = - \frac{\ln \operatorname{tr} \mathrm{e}^{ - \beta H} }{\beta L} \\
    &= - \frac{\ln \operatorname{tr} \exp \left[ - \beta \left( \sum_{k = 1}^{L} 
    \epsilon_{k} (f_{k}^{\dagger} f_{k} - f_{k} f_{k}^{\dagger}) \right) \right]}{\beta L} \\
    &= - \frac{\sum_{k}^{} \ln \operatorname{tr} \exp \left[ - \beta 
    \epsilon_{k} (f_{k}^{\dagger} f_{k} - f_{k} f_{k}^{\dagger}) \right]}{\beta L} \\
    &= - \frac{\sum_{k}^{} \ln \left( \mathrm{e}^{\beta \epsilon_{k}} + \mathrm{e}^{ - \beta \epsilon_{k}}  \right)}{\beta L} = - \frac{\sum_{k}^{} \ln 2 \cosh \beta \epsilon_{k}}{\beta L} \\
    &= - \frac{\ln 2}{\beta} - \frac{\sum_{k}^{} \ln \cosh \beta \epsilon_{k}}{\beta L} \to  - \frac{\ln 2}{\beta} - \frac{1}{2\pi \beta} \int_{ - \pi}^{\pi} \mathrm{d}k \, \ln \cosh \beta \epsilon_{k}
\end{align*}
$$

### 内能

$$
\begin{align*}
    u &= \frac{\partial f\beta}{\partial \beta} = \frac{\partial}{\partial \beta} \left(  - \ln 2 - \frac{\sum_{k}^{} \ln \cosh \beta \epsilon_{k}}{L} \right) \\
    &= 
    - \frac{\sum_{k}^{} \epsilon_{k} \tanh \beta \epsilon_{k}}{L} \to - \frac{1}{2\pi} \int_{ - \pi}^{\pi} \mathrm{d}k \, \epsilon_{k}\tanh \beta \epsilon_{k}
\end{align*}
$$

### 比热

$$
\begin{align*}
    c_{v} = - \beta^2 \frac{\partial u}{\partial \beta} = - \beta^2 \frac{\sum_{k}^{} \epsilon_{k}^2  \operatorname{sech}^2 \beta \epsilon_{k}}{L} \to - \frac{\beta^2}{2\pi} \int_{ - \pi}^{\pi} \mathrm{d}k \, \epsilon_{k}^2 \operatorname{sech}^2 \beta \epsilon_{k}
\end{align*}
$$

## 开边界自由费米子(特殊)

开边界最近邻费米子:

$$
\begin{align*}
    H = \sum_{j = 1}^{L - 1} \lambda ( c_{j}c_{j + 1}^{\dagger} - c_{j}^{\dagger} c_{j + 1}) + h \sum_{j = 1}^{L} (c_{j}^{\dagger} c_{j} - c_{j}c_{j}^{\dagger} ),
\end{align*}
$$

通过同样 JW 变换, 可以得到自旋模型:
$$
\begin{align*}
    H =& \sum_{j = 1}^{L - 1} J (s_{j}^{x}s_{j + 1}^{x} + s_{j}^{y}s_{j + 1}^{y} )
    + h_{z} \sum_{j = 1}^{L} s_{j}^{z},
\end{align*}
$$
其中
$$
    \begin{cases}
        J = 2\lambda \\
        h_{z} = 2h
    \end{cases}
$$

这个哈密顿量可以通过正弦变换:
$$
    f_{s}^{\dagger} = \sqrt{ \frac{2}{L + 1} } \sum_{r = 1}^{L} c_{r} \sin \left( \frac{\pi}{L + 1} rs  \right)
$$
对角化:
$$
\begin{align*}
    H = \sum_{k = 1}^{L} \epsilon_{k} (f_{k}^{\dagger} f_{k} - f_{k} f_{k}^{\dagger} )
\end{align*}
$$
其中激发能为:
$$
\begin{align*}
    \epsilon_{k} = \lambda \cos \left( \frac{\pi}{L + 1}k - h \right)
\end{align*}
$$

因而基态能为:
$$
    E_{g} = - \sum_{k = 1}^{L} \Bigl| \lambda \cos \Bigl( \frac{\pi}{L + 1} k \Bigr) - h \Bigr|  
$$

## 开边界自由费米子(一般)

一般自由费米子系统:
$$
\begin{align*}
    H  &= \sum_{k>j}^{} 2(\lambda_{jk} c_{j}c_{k}^{\dagger} - \lambda_{jk}^* c_{j}^{\dagger} c_{k}) +  2(\gamma_{jk} c_{j}c_{k} - \gamma_{jk}^* c_{j}^{\dagger} c_{k}^{\dagger} ) - \sum_{j = 1}^{L} \lambda_{j} (c_{j}^{\dagger} c_{j} - c_{j}c_{j}^{\dagger} ), \\
    &=
    \begin{bmatrix}
        c_1^{\dagger}  \\ c_2^{\dagger}  \\ \cdots  \\ c_{L}^{\dagger}  \\ c_{1} \\ c_2 \\ \cdots  \\ c_{L}  \\
    \end{bmatrix}^{T}
    \begin{bmatrix}
        \begin{array}{cccc:cccc}
            -\lambda_{1} & -\lambda_{12}^* & \cdots  & -\lambda_{1L}^*  & 0 & -\gamma_{12}^*  & \cdots  & -\gamma_{1L}^*   \\
            -\lambda_{12}  & -\lambda_{2} & \cdots  & -\lambda_{2L} & \gamma_{12}^*  & 0 & \cdots  & -\gamma_{2L}^*   \\
            \cdots & \cdots  & \cdots  & \cdots  & \cdots  & \cdots  & \cdots  & \cdots   \\
            -\lambda_{1L}  & -\lambda_{2L} & \cdots  & -\lambda_{L} &  \gamma_{1L}^*  & \gamma_{2L}^*  & \cdots  & 0 \\
            \hdashline 
            0 & \gamma_{12} & \cdots  & \gamma_{1L} & \lambda_{1} & \lambda_{12} & \cdots  & \lambda_{1L}  \\
            -\gamma_{12} & 0 & \cdots  & \gamma_{2L} & \lambda_{12}^*  & \lambda_{2} & \cdots  & \lambda_{2L}  \\
            \cdots & \cdots  & \cdots  & \cdots  & \cdots  & \cdots  & \cdots  & \cdots   \\
            -\gamma_{1L} & -\gamma_{2L} & \cdots  & 0 & \lambda_{1L}^*  & \lambda_{2L}^* & \cdots  & \lambda_{L}  \\
         \end{array}
    \end{bmatrix}
    \begin{bmatrix}
         c_1 \\
         c_2 \\
         \cdots  \\
         c_{L} \\
         \hdashline
         c_1^{\dagger}  \\
         c_2^{\dagger}  \\
         \cdots  \\
         c_{L}^{\dagger}  \\
    \end{bmatrix}
\end{align*}
$$

简记为:
$$
    H = 
    \begin{pmatrix} c^{\dagger T} , c^{T} \end{pmatrix}
    \begin{pmatrix}
        -\lambda^*  &  - \gamma ^*  \\
        \gamma   &  \lambda  \\
    \end{pmatrix}
    \begin{pmatrix} c\\ c^{\dagger} \end{pmatrix}
$$

对角化得到:
$$
    H = \sum_{i}^{} \epsilon_{i} (f_{i}^{\dagger}  f_{i}  - f_{i} f_{i}^{\dagger} )
$$

其中 ${ \epsilon_{i} }$, 是矩阵 ${ \begin{pmatrix}-\lambda^*  &  - \gamma ^*  \\ \gamma   &  \lambda  \\ \end{pmatrix} }$ 的本征值

若只包含最近邻项:
$$
\begin{align*}
    H  &= \sum_{j = 1}^{L - 1} 2(\lambda_{j} c_{j}c_{j + 1}^{\dagger} - \lambda_{j}^* c_{j}^{\dagger} c_{j + 1}) +  2(\gamma_{j} c_{j}c_{j + 1} - \gamma_{j}^* c_{j}^{\dagger} c_{j + 1}^{\dagger} ) + \sum_{j = 1}^{L} h_{j} (c_{j}^{\dagger} c_{j} - c_{j}c_{j}^{\dagger} ), \\
    &=
    \begin{bmatrix}
        c_1^{\dagger}  \\ c_2^{\dagger}  \\ \cdots  \\ c_{L}^{\dagger}  \\ c_{1} \\ c_2 \\ \cdots  \\ c_{L}  \\
    \end{bmatrix}^{T}
    \begin{bmatrix}
        \begin{array}{cccc:cccc}
            h_1 & -\lambda_{1}^* & \cdots  & 0 & 0 & -\gamma_{1}^*  & \cdots  & 0   \\
            -\lambda_{1}  & h_2 & \cdots  & 0 & \gamma_{1}^*  & 0 & \cdots  & 0 \\
            \cdots & \cdots  & \cdots  & \cdots  & \cdots  & \cdots  & \cdots  & \cdots   \\
            0  & 0 & \cdots  & h_{L} &  0  & 0 & \cdots  & 0 \\
            \hdashline 
            0 & \gamma_{1} & \cdots  & 0 & - h_1 & \lambda_{1} & \cdots  & 0  \\
            -\gamma_{1} & 0 & \cdots  & 0 & \lambda_{1}^*  & - h_2 & \cdots  & 0  \\
            \cdots & \cdots  & \cdots  & \cdots  & \cdots  & \cdots  & \cdots  & \cdots   \\
            0 & 0 & \cdots  & 0 & 0 & 0 & \cdots  & - h_{L}  \\
         \end{array}
    \end{bmatrix}
    \begin{bmatrix}
         c_1 \\
         c_2 \\
         \cdots  \\
         c_{L} \\
         \hdashline
         c_1^{\dagger}  \\
         c_2^{\dagger}  \\
         \cdots  \\
         c_{L}^{\dagger}  \\
    \end{bmatrix}
\end{align*}
$$

通过 JW 变换, 其对应的自旋模型为:
$$
\begin{align*}
    H =& \sum_{j = 1}^{L - 1} (J^{x}_{j} s_{j}^{x}s_{j + 1}^{x} + J^{y}_{j} s_{j}^{y}s_{j + 1}^{y} + J^{yx}_{j} s_{j}^{y}s_{j + 1}^{x} + J^{xy}_{j} s_{j}^{x}s_{j + 1}^{y}) 
    + \sum_{j = 1}^{L} h^{z}_{j} s_{j}^{z},
\end{align*}
$$
其中:
$$
    \begin{cases}
        \lambda_{j} = \frac{1}{8} \left( J^{x}_{j} + J^{y}_{j} \right) + \frac{\mathrm{i}}{8}\left( J^{yx}_{j} - J^{xy}_{j} \right) \\
        \gamma_{j} = \frac{1}{8} \left( J^{x}_{j} - J^{y}_{j} \right) + \frac{\mathrm{i}}{8}\left( J^{yx}_{j} + J^{xy}_{j} \right) \\
        h_{j} = \frac{1}{2} h^{z}_{j}
    \end{cases}
    \text{   或者   }
    \begin{cases}
        J^{x}_{j} = 4 \Re (\lambda_{j} + \gamma_{j}) \\
        J^{y}_{j} = 4 \Re (\lambda_{j} - \gamma_{j}) \\
        J^{yx}_{j} = 4 \Im (\lambda_{j} + \gamma_{j}) \\
        J^{xy}_{j} = - 4\Im (\lambda_{j} - \gamma_{j}) \\
        h^{z}_{j} = 2h_{j}
    \end{cases}
$$