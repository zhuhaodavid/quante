# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-08-28 16:37:49
# @Last Modified by:   hzhu
# @Last Modified time: 2025-08-29 02:29:03

from scipy.linalg import get_lapack_funcs

def permuteschur(T, Q, order):
    """
    Reorder complex Schur decomposition (T, Q) according to `order`.
    Using LAPACK trexc directly (like Julia).

    #todo: rewrite
    """
    trexc, = get_lapack_funcs(('trexc',), (T,))

    n = T.shape[0]
    p = list(order)  # copy, will be modified

    T_new, Q_new = T.copy(), Q.copy()

    for i in range(len(p)):
        ifirst = int(p[i]) + 1   # LAPACK uses 1-based indices
        ilast = i + 1

        if ifirst != ilast:
            T_new, Q_new, info = trexc(T_new, Q_new, ifirst, ilast)
            if info != 0:
                raise RuntimeError(f"trexc failed with info={info}")

            # Update permutation indices (same as Julia code)
            for k in range(i + 1, len(p)):
                if p[k] < p[i]:
                    p[k] += 1

    return T_new, Q_new


# function schur2eigvals(T::AbstractMatrix{<:BlasReal}, which::AbstractVector{Int})
#     n = checksquare(T)
#     which2 = unique(which)
#     length(which2) == length(which) ||
#         throw(ArgumentError("which should contain unique values"))
#     D = zeros(Complex{eltype(T)}, length(which2))
#     for k in 1:length(which)
#         i = which[k]
#         if i < n && !iszero(T[i + 1, i])
#             halftr = (T[i, i] + T[i + 1, i + 1]) / 2
#             diff = (T[i, i] - T[i + 1, i + 1]) / 2
#             d = diff * diff + T[i, i + 1] * T[i + 1, i]  # = hafltr*halftr - det
#             D[i] = halftr + im * sqrt(-d)
#         elseif i > 1 && !iszero(T[i, i - 1])
#             halftr = (T[i, i] + T[i - 1, i - 1]) / 2
#             diff = -(T[i, i] - T[i - 1, i - 1]) / 2
#             d = diff * diff + T[i, i - 1] * T[i - 1, i]  # = hafltr*halftr - det
#             D[i] = halftr - im * sqrt(-d)
#         else
#             D[i] = T[i, i]
#         end
#     end
#     return D
# end