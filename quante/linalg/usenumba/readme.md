
quante.linalg.usenumba 这个文件中储存了对 numba 的设置和使用到 numba 的函数

这个文件不能通过 qt.linalg. 代码提示出来(因为 `quante.linalg.__init__.py` 中使用了 from xxx import * 的保护), 因此这个文件中是函数只能被包里的其他函数调用.

如果想在包外调用 quante.linalg.usenumba 中的函数, 唯一的方式是 from quante.linalg.usenumba.xxx import xxx, 这非常不常用, 所以这个文件夹中的文件都不需要加 `__all__` 来保护

文件夹中的文件:

- numba_settings:

    用来调整程序可以调用的最高核心数, 

    改变 numba 缓存储存位置(默认在 site-packages/quante/numba_cache 中)

    返回 parallel_reduce 可以并行的执行 f(a,b,c,...) 这样的函数

- eig_modified_numba

    本征分解用到的 numba 函数

- operations_numba

    operations 用到的 numba 函数
