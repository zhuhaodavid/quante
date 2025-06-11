# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-11 22:47:58
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-11 22:48:24

import os
import functools

__all__ = [
    "njit",
    "pnjit",
    "vectorize",
    "pvectorize",
    "get_thread_pool",
    "parallel_reduce",
    "numba_cache_dir",
    "typed",
]


#####################################
#       设置使用的核心数
#####################################
def detect_number_of_cores():
    """
    获得计算机最大的核心数
    """
    import subprocess
    
    # Linux, Unix and MacOS:
    if hasattr(os, "sysconf"):
        if "SC_NPROCESSORS_ONLN" in os.sysconf_names:
            # Linux & Unix:
            ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
            if isinstance(ncpus, int) and ncpus > 0:
                return ncpus
        else:  # OSX:
            return int(subprocess.check_output(["sysctl", "-n", "hw.ncpu"]))
    # Windows:
    try:
        ncpus = int(os.environ.get("NUMBER_OF_PROCESSORS", ""))
        if ncpus > 0:
            return ncpus
    except ValueError:
        pass
    return 1  # Default

# 设置核心数

# if 'NUMEXPR_MAX_THREADS' not in os.environ:
#     os.environ['NUMEXPR_MAX_THREADS'] = str(detect_number_of_cores())
if "OMP_NUM_THREADS" in os.environ:
    _NUM_THREAD_WORKERS = int(os.environ["OMP_NUM_THREADS"])
else:
    import psutil
    _NUM_THREAD_WORKERS = psutil.cpu_count(logical=False)
# if "NUMBA_NUM_THREADS" in os.environ:
#     if int(os.environ["NUMBA_NUM_THREADS"]) != _NUM_THREAD_WORKERS:
#         pass
# else:
#     os.environ["NUMBA_NUM_THREADS"] = str(_NUM_THREAD_WORKERS)


#####################################
#     修改 numba 的缓存目录
#####################################

import numba  # !! 必须在设置核心数之后引用
# 设置 numba 的缓存目录
try:
    # 默认存到 site-packages/quante/numba_cache 中
    import sysconfig
    global_site_packages = sysconfig.get_paths()["purelib"]
    numba_cache_dir = os.path.join(global_site_packages, 'numba_quante_cache')
    os.makedirs(numba_cache_dir, exist_ok=True)
except:
    # 如果没有权限,那么就存到 ~/quante/numba_cache 中
    import site
    userbase_dir = site.getuserbase()  # 用户主目录下的python包路径
    numba_cache_dir = os.path.join(userbase_dir, 'numba_quante_cache')
    os.makedirs(numba_cache_dir, exist_ok=True)

numba.config.CACHE_DIR = numba_cache_dir
config = numba.config

_NUMBA_CACHE = True
_NUMBA_PAR = True

njit = functools.partial(numba.njit, cache=_NUMBA_CACHE)
pnjit = functools.partial(numba.njit, cache=_NUMBA_CACHE, parallel=_NUMBA_PAR)
prange = numba.prange
vectorize = functools.partial(numba.vectorize, cache=_NUMBA_CACHE)
pvectorize = functools.partial(
    numba.vectorize,
    cache=_NUMBA_CACHE,
    target="parallel",
)
from numba import types, typed

#####################################
#     修改 numba 的缓存目录
#####################################

if "OMP_NUM_THREADS" in os.environ:
    _NUM_THREAD_WORKERS = int(os.environ["OMP_NUM_THREADS"])
else:
    import psutil
    _NUM_THREAD_WORKERS = psutil.cpu_count(logical=False)

# 获取进程池,为了实现 parallel_reduce
class CacheThreadPool(object):
    def __init__(self, func):
        self._settings = "__UNINITIALIZED__"
        self._pool_fn = func

    def __call__(self, num_threads=None):
        # convert None to default so caches the same
        if num_threads is None:
            num_threads = _NUM_THREAD_WORKERS
        # first call
        if self._settings == "__UNINITIALIZED__":
            self._pool = self._pool_fn(num_threads)
            self._settings = num_threads
        # new type of pool requested
        elif self._settings != num_threads:
            self._pool.shutdown()
            self._pool = self._pool_fn(num_threads)
            self._settings = num_threads
        return self._pool

@CacheThreadPool
def get_thread_pool(num_workers=None):
    from concurrent.futures import ThreadPoolExecutor as PoolExecutor
    # from concurrent.futures import ProcessPoolExecutor as PoolExecutor
    return PoolExecutor(num_workers)

# 指定并行分配任务的方法
try:
    import cytoolz
    partition_all = cytoolz.partition_all
except ImportError:
    import toolz
    partition_all = toolz.partition_all
    pass

def parallel_reduce(fn, seq, nthreads=_NUM_THREAD_WORKERS):
    """
    并行的实现 fn(seq[0], seq[1], seq[2], ...)
    """
    if nthreads == 1: return functools.reduce(fn, seq)
    pool = get_thread_pool(nthreads)  # cached
    def _sfn(x):
        if len(x) == 1: return x[0]
        return fn(*x)
    def _inner_preduce(x):
        if len(x) <= 2: return _sfn(x)
        paired_x = partition_all(2, x)
        new_x = tuple(pool.map(_sfn, paired_x))
        return _inner_preduce(new_x)
    return _inner_preduce(tuple(seq))

