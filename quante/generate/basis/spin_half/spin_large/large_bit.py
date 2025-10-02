# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-10-01 22:03:50
# @Last Modified by:   hzhu
# @Last Modified time: 2025-10-02 02:46:53

import numpy as np
from numba import njit
from quante.generate.basis.spin_half.bitsoperation import count_tot_down_64


@njit('Tuple((uint64[:], i8))(b1[:])')
def bigint_from_array(arr):
    """ Convert a list of True/False to a big integer represented as an array of uint64. """
    n = len(arr)
    nwords = (n + 63) // 64
    res = np.zeros(nwords, dtype=np.uint64)
    one = np.uint64(1)
    for i in range(n):
        if arr[n-i-1]:
            res[nwords-(i//64)-1] |= one << np.uint64(i % 64)
    return res, n

@njit('i8(uint64[:], i8, i8)')
def bigint_at(arr, k, nbits):
    """ Get the k-th bit of the big integer. """
    pos_r2l = nbits - k - 1
    word_index = len(arr) - pos_r2l // 64 - 1
    bit_index = pos_r2l % 64
    return (arr[word_index] >> bit_index) & np.uint64(1)

@njit('string(uint64[:], i8)')
def bigint_to_str(arr, nbits):
    """ Print the big integer in binary format up to nbits. """
    left_len = nbits % 64
    res = []
    for i in range(left_len):
        res.append('1' if (arr[0] >> (left_len - i - 1)) & 1 else '0')
    for i in range(1, len(arr)):
        for j in range(64):
            res.append('1' if (arr[i] >> (63 - j)) & 1 else '0')
    return ''.join(res)

@njit('Tuple((uint64[:], i8))(string)')
def bigint_from_str(s):
    """ Convert a binary string to a big integer represented as an array of uint64. """
    n = len(s)
    nwords = (n + 63) // 64
    res = np.zeros(nwords, dtype=np.uint64)
    one = np.uint64(1)
    for i in range(n):
        if s[n-i-1]=='1':
            res[nwords-(i//64)-1] |= one << np.uint64(i % 64)
    return res, n

@njit('b1(uint64[:], uint64[:])')
def bigint_equal(arr1, arr2):
    """ Compare two big integers represented as arrays of uint64. Return True if arr1 == arr2. """
    for i in range(len(arr1)):
        if arr1[i] != arr2[i]:
            return False
    return True
 
@njit('i8(uint64[:])')
def bigint_bitcount(arr):
    """ Count the number of set bits in the big integer. """
    count = 0
    for word in arr:
        count += count_tot_down_64(word)
    return count

@njit('b1(uint64[:], uint64[:])')
def bigint_larger_than(arr1, arr2):
    """ Compare two big integers represented as arrays of uint64. Return True if arr1 > arr2. """
    l = len(arr1)
    for i in range(l):
        if arr1[i] > arr2[i]:
            return True
        elif arr1[i] < arr2[i]:
            return False
    return False  # equal

@njit('void(uint64[:], i8, i8)')
def bigint_flip_at(arr, k, nbits):
    """ Flip the k-th bit of the big integer. """
    pos_r2l = nbits - k - 1
    word_index = len(arr) - pos_r2l // 64 - 1
    bit_index = pos_r2l % 64
    arr[word_index] ^= np.uint64(1) << np.uint64(bit_index)


@njit('b1(uint64[:], i8)')
def bigint_add_one(arr, nbits):
    """ Add one to the big integer represented as an array of uint64. """
    carry = np.uint64(1)
    overflow = False
    for i in range(len(arr)-1, -1, -1):
        if carry == 0:
            break
        new_val = arr[i] + carry
        if new_val < arr[i]:  # overflow
            carry = np.uint64(1)
        else:
            carry = np.uint64(0)
        arr[i] = new_val
    left_len = nbits % 64
    if (arr[0] >> left_len) > 0 or carry != 0:
        overflow = True
    return overflow

@njit('uint64[:](uint64[:])')
def bigint_copy(arr):
    res = np.empty_like(arr)
    for i in range(len(arr)):
        res[i] = arr[i]
    return res

#######################################
# Gosper’s hack for large int
#######################################
# @njit('b1(uint64[:], i8)')
# def next_combination_bigint(arr, nbits):
#     """
#     在 arr (uint64[:]) 表示的 nbits 位组合上生成下一个组合（in-place）。
#     返回 True 表示成功生成下一个组合，False 表示已经到达末尾（如 111..1000..0 的最后一个）。
#     位索引采用你现有的约定：k in [0, nbits-1]，0 为最左（MSB 侧）。
#     算法（左索引表示法）：
#       从右端（nbits-1）向左扫描，寻找位置 p (1..nbits-1) 满足：
#          bit(p) == 1 and bit(p-1) == 0
#       找到后：
#          - 把 bit(p) 置 0，bit(p-1) 置 1（相当于将右侧的一个 1 向左移动一位）
#          - 计算 p 右边（索引 p+1 .. nbits-1）有多少个 1，清空那部分
#          - 在最右端重新设置相同数量的 1（将这些 1 都放到最右边）
#     该实现用到了 bigint_at / bigint_flip_at（你已有），逐位操作但完全 numba 兼容。
#     """
#     # 从右往左找“10”模式（在你的左索引表示里，对应 bit(p)==1, bit(p-1)==0）
#     for p in range(nbits - 1, 0, -1):
#         if bigint_at(arr, p, nbits) == 1 and bigint_at(arr, p - 1, nbits) == 0:
#             # 把 p 处的 1 -> 0；把 p-1 处的 0 -> 1
#             # 这里使用 flip（因为我们已知当前位的值），flip 两次就完成替换
#             bigint_flip_at(arr, p, nbits)       # 原来是1，变为0
#             bigint_flip_at(arr, p - 1, nbits)   # 原来是0，变为1

#             # 统计 p 右边 (p+1 .. nbits-1) 的 1 的个数
#             ones = 0
#             if p + 1 <= nbits - 1:
#                 for idx in range(p + 1, nbits):
#                     ones += bigint_at(arr, idx, nbits)

#             # 清空 p 右边的所有位（p+1 .. nbits-1）
#             if p + 1 <= nbits - 1:
#                 for idx in range(p + 1, nbits):
#                     # 如果是1就 flip 成0；如果是0就跳过
#                     if bigint_at(arr, idx, nbits) == 1:
#                         bigint_flip_at(arr, idx, nbits)

#             # 在最右端（从 nbits-1 向左）放置 ones 个 1
#             # 例如 ones=3，则设置 positions nbits-1, nbits-2, nbits-3 为 1
#             for j in range(ones):
#                 target = nbits - 1 - j
#                 # 如果目标位当前为0，则 flip 为1（若已是1，flip会把它变0 — 因此先检查）
#                 if bigint_at(arr, target, nbits) == 0:
#                     bigint_flip_at(arr, target, nbits)

#             return True

#     # 没找到，说明已经是最后一个组合
#     return False

@njit
def _ctz(x):
    """Count trailing zeros of a uint64 word"""
    c = 0
    mask = np.uint64(1)
    while (x & mask) == 0:
        x >>= np.uint64(1)
        c += 1
    return c

@njit
def _find_lower_bit_array(arr):
    """Find global index of lowest set bit in uint64 array (small endian)"""
    l = len(arr)
    for i in range(l-1,-1,-1):
        if arr[i] != 0:
            return (l-i-1)*64 + _ctz(arr[i])
    return -1  # all zero

@njit
def uint64_add(a, b):
    """Add two uint64 arrays of same length, return result, in-place add a+b->res"""
    n = len(a)
    res = np.empty(n, dtype=np.uint64)
    carry = np.uint64(0)
    for i in range(n-1,-1,-1):
        tmp = a[i] + b[i] + carry
        if tmp < a[i] or (carry==1 and tmp == a[i]):
            carry = np.uint64(1)
        else:
            carry = np.uint64(0)
        res[i] = tmp
    return res

@njit
def uint64_xor(a, b):
    n = len(a)
    res = np.empty(n, dtype=np.uint64)
    for i in range(n):
        res[i] = a[i] ^ b[i]
    return res

@njit
def bigint_shift_right(arr, k):
    """Right shift array by k bits"""
    n = len(arr)
    res = np.zeros(n, dtype=np.uint64)
    word_shift = k // 64
    bit_shift = k % 64
    for i in range(n-1,word_shift-1,-1):
        j = i - word_shift
        res[i] |= arr[j] >> np.uint64(bit_shift)
        if bit_shift and j > 0:
            res[i] |= arr[j-1] << np.uint64(64 - bit_shift)
    return res

@njit
def next_combination_bigint_gosper(arr):
    """Big integer Gosper's hack in uint64 array (small endian), in-place update arr"""
    n = len(arr)
    k = _find_lower_bit_array(arr)
    # 1. u = 2^k
    u = np.zeros(n, dtype=np.uint64)
    word_idx = n - k // 64 - 1
    bit_idx = k % 64
    u[word_idx] = np.uint64(1) << bit_idx

    # 2. v = arr + u
    v = uint64_add(arr, u)

    # 3. t = (v ^ arr) // u -> shift_right by k
    t = uint64_xor(v, arr)
    t = bigint_shift_right(t, k)

    # 4. t >>= 2
    t = bigint_shift_right(t, 2)
    # print(bigint_to_str(t, 100))

    # 5. res = v + t
    res = uint64_add(v, t)

    return res

for i in range(100):
    a1 = np.array([False]*100)
    a1[i] = True
    bigint1, nbits = bigint_from_array(a1)
    r = bigint_at(bigint1, i, nbits)
    assert r==1
    res = bigint_to_str(bigint1, nbits)
    assert res[i] == '1'
    bigint2, nbits2 = bigint_from_str(res)
    assert bigint_equal(bigint1, bigint2)
    r = bigint_bitcount(bigint1)
    assert r==1

arr = np.array([False]*100)
arr[-1] = True
b1, n = bigint_from_array(arr)
arr = np.array([False]*100)
arr[-2] = True
b2, n  = bigint_from_array(arr)
assert bigint_larger_than(b2, b1)
assert not bigint_larger_than(b1, b2)

arr = np.array([False]*100)
arr[-1] = True
arr[0] = True
b1, n = bigint_from_array(arr)
s1 = bigint_to_str(b1, n)
# 100 bits, only last two bits are 1
arr = np.array([False]*100)
arr[-1] = True
b1, n = bigint_from_array(arr)
bigint_flip_at(b1, 0, n)
s2 = bigint_to_str(b1, n)  
# 100 bits, only last two bits are 1
assert s1 == s2

arr = np.array([True]*100)
arr[0] = False
b1, n = bigint_from_array(arr)
overflow = bigint_add_one(b1, n)
assert not overflow
assert bigint_to_str(b1, n) == '1'+'0'*99


arr = np.array([True]*100)
b1, n = bigint_from_array(arr)
overflow = bigint_add_one(b1, n)
assert overflow
assert bigint_to_str(b1, n) == '0'*100

arr = np.array([False]*100)
arr[-1] = True
arr[-3] = True
b1, n = bigint_from_array(arr)

# for i in range(100):
#     b1 = next_combination_bigint_gosper(b1)
#     print(bigint_to_str(b1, n))

