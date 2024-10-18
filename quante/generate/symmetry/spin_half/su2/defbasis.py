# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-19 15:51:39
# @Last Modified by:   hzhu
# @Last Modified time: 2024-09-19 18:11:32

from ..bitsoperation import next_combination, count_tot_down
from .....linalg.usenumba.numba_settings import njit, config, numba_cache_dir, pvectorize
import numpy as np

# 1/2 ----- 0 ----- 1/2 ----- 0 ...
#     \                \
#      \                 ---- 1 ...
#       \
#        -- 1 ----- 1/2 ----- 0 ...
#            \         \
#             \          -----1 ...
#              \
#                -- 3/2 ----- 1 ...
#                      \
#                        ---- 2 ...
#
# 2 ....... 4 ...... 8 ...... 16  .....


config.CACHE_DIR = numba_cache_dir
@njit
def get_jlist(L, j):
    """
    从 1/2, 经过 L-1 个 1/2 自旋的耦合，得到总自旋 j 的路径
    
    净增加为 (j-1/2)，也就是这 L-1 个 1/2 自旋的总自旋为 (j-1/2)
    
    假设有 x 个向上的自旋，那么有 L-x-1 个向下的自旋，总磁矩为 x/2 - (L-x-1)/2 = x - L/2 + 1/2
    
    因而 L-1 个自旋中，自旋向上的自旋数为 (j-1/2) + L/2 - 1/2 = L/2 + j - 1
    
    自旋向下的总数为 L - (j + L/2 - 1) - 1 = L/2 - j
    """
    num_one = int(L/2 - j)
    init_state = (1<<num_one) - 1
    if init_state == 0:
        return np.array([0])
    j_list = []
    while init_state < (1 << (L-1)):
        # 检查叠加过程中是否有非法的路径（负数净磁矩）
        for i in range(1,L-1):
            # 可以数出其中自旋向下的个数为 x，自旋向上的个数为 L-1-x-i
            # 这样净磁矩就是 -x/2 + (L-1-x-i)/2 = -x + L/2 - 1/2 - i/2
            # 不能小于 -1/2，因此 -x + L/2 - 1/2 - i/2 > -1/2
            # 也就是 2x < L-i
            if 2 * count_tot_down(init_state >> i) > L - i:
                break
        else:
            j_list.append(init_state)
        init_state = next_combination(init_state)
    return np.array(j_list)


@pvectorize("f8(i8,i8,i8)")
def get_ci(jpath, mpath, L):
    """calculate cg coeffiect by
    < j1, m1, 1/2, m2 | j, m > =
    
                  |           m2 = 1/2               |           m2 = -1/2            |
     j = j1 + 1/2 |       sqrt( (j + m)/2j )         |       sqrt( (j - m)/2j )       |
     j = j1 - 1/2 | - sqrt( (j - m + 1)/(2j + 2) )   |  sqrt( (j + m + 1)/(2j + 2) )  |

    s.t.
    | j, m > = | j1, m-1/2, 1/2, 1/2 > < j1, m-1/2, 1/2, 1/2 | j, m > 
                     + | j1, m+1/2, 1/2, - 1/2 > < j1, m+1/2, 1/2, -1/2 | j, m >

    jpath: 代表 |j,m> 的路径
    mpath: 中的 i 个位置，代表 mi 的是 1/2 还是 -1/2
    """
    coefficient = 1.
    
    for i in range(L-2,-1,-1):
        # 从 jpath 中提取出地 i 位置的 j
        # 初始值为 1/2，jpath 中自旋向下的个数是 x，自旋向上的个数是 L-1-x-i
        # 因此 j = 1/2 - x/2 + (L-1-x-i)/2 = (L - i)/2 - x
        ji = (L - i)/2 - count_tot_down(jpath>>i)
        
        # 从 mpath 中提取出第 i 个位置的 m
        # 若向下的个数是 x，那么向上的个数就是 L - x - i
        # 这样总磁矩就是：-x/2 + (L-x-i)/2 = (L-i)/2 - x
        mi = (L - i)/2 - count_tot_down(mpath>>i)
        
        # 根据第 i 个位置的 m 是 1/2 还是 -1/2，来决定系数
        idf = 1 << i  # 定位到第 i 个位置
        m2_is_up = mpath & idf == 0
        j_is_up = jpath & idf == 0
        
        if m2_is_up and j_is_up:
            coefficient *= np.sqrt((ji + mi)/(2*ji))
        elif m2_is_up and not j_is_up:
            coefficient *= -np.sqrt((ji - mi + 1)/(2*ji + 2))
        elif not m2_is_up and j_is_up:
            coefficient *= np.sqrt((ji - mi)/(2*ji))
        else:
            coefficient *= np.sqrt((ji + mi + 1)/(2*ji + 2))
        
        if coefficient == 0:
            break
            
    return coefficient
