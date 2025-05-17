.. quante documentation master file, created by
   sphinx-quickstart on Wed Oct 30 21:12:57 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

quante documentation
====================

quante - 一个用于量子物理的Python库


用于处理张量网络、精确对角化、保存工具、日志工具等的工具库。

提供功能：
  1. 更好的日志记录、保存（hdf5）、随机工具
  2. 自适应特征值、SVD方法
  3. 量子态、算符、张量网络生成器

关于 python 计算效率的说明
  1. 首先要确认是否真的需要调优，不要在没有意义的优化上花费时间，优先保证简单易用！
  2. 80/20法则：百分之二十的代码产生了百分之八十的时间消耗。
  3. 对于简单数据处理的调优使用 numba，cython。


.. toctree::
    :caption: 安装 & 使用
    :maxdepth: 1

    ./installation/installation
    ./installation/usage


.. toctree::
    :caption: 函数 API
    :maxdepth: 1
    
    ./funcsAPI/basicfun
    ./funcsAPI/linalg
    ./funcsAPI/generate
    ./funcsAPI/solvable_models
    ./funcsAPI/tensor
    ./funcsAPI/torch_utils
    ./funcsAPI/quantity
