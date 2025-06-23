# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-11 22:13:41
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-23 15:26:52
 
import os as _os
import ast as _ast
import sys as _sys
import numpy as _np
import time as _time
import inspect as _inspect
import logging as _logging
import traceback as _traceback

from types import FunctionType
from typing import Union

import ctypes as _ctypes
import platform as _platform
import difflib

__all__ = [
    "get_free_space",
    "create_folder",
    "check_file_exists",
    "set_logging",
    "println",
    "info",
    "send_email",
    "clear_numba_cache",
    "logger",
    "set_show",
]

# ===================================
#     系统层面的函数，如建立文件夹
# ===================================

def get_free_space(folder: str) -> float:
    """获取磁盘剩余空间。
    
    Parameters
    ----------
    folder : str
        磁盘路径，例如 "D:\\"。
    
    Returns
    -------
    float
        剩余空间，单位为 GB。

    Examples
    --------
    >>> get_free_space("D:\\")
    """
    if _platform.system() == "Windows":
        free_bytes = _ctypes.c_ulonglong(0)
        _ctypes.windll.kernel32.GetDiskFreeSpaceExW(
            _ctypes.c_wchar_p(folder), None, None, _ctypes.pointer(free_bytes)
        )
        return free_bytes.value / (1024 ** 3)
    else:
        import shutil as _shutil
        usage = _shutil.disk_usage(folder)
        return usage.free / (1024 ** 3)
    

def create_folder(path1:str, path2: Union[None, str]=None) -> str:
    """创建一个文件夹（包括路径中指定的所有父文件夹）。
    
    Parameters
    ----------
    path1 : str
        主路径，可以是类似 "xxx/xxx/..." 的格式。
    path2 : Union[None, str], optional
        子路径，必须为 None 或者类似 "xxx" 的字符串。
    
    Returns
    -------
    str
        创建的完整路径，以斜杠结尾。
    
    Examples
    --------
    >>> import quante as qt
    >>> path1 = "data/entanglement"
    >>> path2 = "XXZ"
    >>> create_folder(path1, path2)
    '.../data/entanglement/XXZ/'
    """
    whole_path = _os.path.abspath(path1).replace("\\", "/") + "/"  # 规范化主路径
    whole_path = _os.path.join(whole_path, path2).replace("\\", "/") if path2 else whole_path  # 如果提供了子路径，则加入到主路径中
    _os.makedirs(whole_path, exist_ok=True)  # 创建文件夹（包括所有父文件夹）
    return whole_path if whole_path.endswith("/") else whole_path+"/"   # 确保返回路径以斜杠结尾
 

def check_file_exists(filename):
    """检查文件是否存在"""
    if not _os.path.exists(filename):
        directory = _os.path.dirname(filename)
        if not directory:
            directory = _os.getcwd()
        basename, ext = _os.path.splitext(_os.path.basename(filename))
        all_files_folders = _os.listdir(directory)
        all_files = [f for f in all_files_folders if _os.path.isfile(_os.path.join(directory, f))]
        similar_files = difflib.get_close_matches(basename, [f.split('.')[0] for f in all_files])
        import textwrap
        if similar_files:
            similar_files_with_ext = [f for f in all_files if f.split('.')[0] in similar_files]
            wrapped_filename = textwrap.fill("   ".join(similar_files_with_ext), width=80)
            raise FileNotFoundError(
                f"file\n    {filename} \ndoes not exist, did you mean:\n"
                f"    {wrapped_filename}\n"
                )
        else:
            wrapped_filename = textwrap.fill("   ".join(all_files), width=80)
            all_folder = [f for f in all_files_folders if _os.path.isdir(_os.path.join(directory, f))]
            if all_folder:
                wrapped_foldername = textwrap.fill("   ".join(all_folder), width=80)
                raise FileNotFoundError(
                    f"file\n    {filename} \ndoes not exist, available files are\n"
                    f"    {wrapped_filename} \n"
                    f"folders are \n"
                    f"    {wrapped_foldername}\n"
                    )
            else:
                raise FileNotFoundError(
                    f"file\n    {filename} \ndoes not exist, available files are\n"
                    f"    {wrapped_filename}\n"
                    )


def clear_numba_cache():
    """清除 numba 的缓存目录"""
    from .utils_numba import numba_cache_dir, numba
    import shutil
    shutil.rmtree(numba_cache_dir, ignore_errors=True)

   
#############################################################
#  日志工具
#############################################################

# 创建自定义的日志记录器
logger = _logging.getLogger('quante_logger')

def set_logging(level: int = 1, savelog: bool = False, filenameTime: bool = False, logtime: bool = False, showlevel=False):
    """配置日志记录功能.
    
    Parameters
    ----------
    level : int, optional
        日志记录的级别，可以填 -1, 1, 2, 3, 4，分别对应 debug, info, warning, error, critical，默认为 1。
    savelog : bool, optional
        是否将日志保存到文件中，默认为 `False`。
    filenameTime : bool, optional
        是否在日志文件名中包含时间戳，默认为 `False`。
    logtime : bool, optional
        是否在日志消息中包含时间戳，默认为 `False`。
    
    Returns
    -------
    None: 该函数无返回值。
    
    Examples
    --------
    >>> set_logging(savelog=True, logtime=True)
    """
    # 清除已有的处理器，防止重复添加
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    _format = ""
    if logtime:  # 根据 `logtime` 参数设置日志格式
        _format += "%(asctime)s"
    if showlevel:
        _format += " %(levelname)s"
    if _format:
        _format += ": %(message)s"
    else:
        _format += "%(message)s"

    if savelog:
        filename = "log/"
        create_folder("log/")  # 创建日志目录
        filename += _os.path.basename(_sys.argv[0])[:-3]  # 根据运行的脚本文件名生成日志文件名
        if filenameTime:  # 如果 `filenameTime` 为 `True`，在文件名中添加时间戳
            now = "_" + _time.strftime("%Y-%m-%d-%H_%M_%S", _time.localtime(_time.time()))
            filename += now
        filename += '.log'
        file_handler = _logging.FileHandler(filename, mode="w", encoding='utf-8')
        file_handler.setFormatter(_logging.Formatter(_format))
        logger.addHandler(file_handler)
        try:
            println.use_color = False
        except:
            pass
    else:
        console_handler = _logging.StreamHandler()
        console_handler.setFormatter(_logging.Formatter(_format))
        logger.addHandler(console_handler)
        try:
            println.use_color = True
        except:
            pass
    
    # 设置日志记录级别
    assert level in [-1, 1, 2, 3, 4], "Invalid log level, should be in [-1, 1, 2, 3, 4]"
    logger.setLevel({-1:_logging.DEBUG, 1:_logging.INFO, 2:_logging.WARNING, 3:_logging.ERROR, 4:_logging.CRITICAL}[level])
    logger.propagate = False  # 防止日志消息传播到root日志记录器

set_logging()  # 使用默认日志记录器

def custom_exception_handler(exc_type, exc_value, exc_traceback):
    # logger.debug("发生错误！这是固定的提示信息。")
    tb_str = "".join(_traceback.format_exception(exc_type, exc_value, exc_traceback))
    logger.error(tb_str)  # 输出格式化的traceback字符串

# 下面这句话可以让 logging 记录所有的报错信息
_sys.excepthook = custom_exception_handler

class COLOR:
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    DEFAULT = '\033[39m'

def set_color(s: str, color: str) -> str:
    return f"{color}{s}{COLOR.DEFAULT}"

def objstr(obj, color=True, compact=False, indent=0):
    """如果是 ndarray，那么输出它的 __str__，否则用 pprint 得到字符串"""
    import pprint
    isobject = False
    if type(obj) == _np.ndarray:
        if compact:
            s = _np.array2string(obj, formatter={'float_kind': lambda x: "%.2f" % x}, max_line_width=None, threshold=4)
        else:
            s = obj.__str__()
    elif isinstance(obj, list):
        s = pprint.pformat(obj, compact=True)
    elif isinstance(obj, FunctionType):
        return f"<function {obj.__name__}>", False
    else:
        if (obj.__class__.__str__ is not object.__str__ or obj.__class__.__repr__ is not object.__repr__):
            s = pprint.pformat(obj)
        else:
            isobject = True
            s = _get_custom_object_str(obj, color)
    s = s.replace("\\n", "\n")  # Preserve string newlines in output.
    s = '\n'.join(v if i == 0 else " "*indent + v for i, v in enumerate(s.split('\n')))
    return s, isobject

def _get_custom_object_str(obj, color=True, ):
    import inspect
    # 首先拿到 header, footer
    obj_type = type(obj)
    header = f"<{obj_type.__name__} {hex(id(obj))}"
    footer = ">"
    if color:
        header = set_color(header, COLOR.CYAN)
        footer = set_color(footer, COLOR.CYAN)
    
    # 拿到所有的属性
    attrs = []
    attr_pattern: str = r"(?!_).*"
    import re
    for attr in dir(obj):
        if re.fullmatch(attr_pattern, attr):
            try:
                attr_val = getattr(obj, attr)
            except AttributeError:
                continue
            if inspect.ismethod(attr_val) or inspect.isbuiltin(attr_val):
                continue
            else:
                attrs.append(attr)
    
    # 将属性组装起来
    elems = ""
    for key in sorted(attrs):
        val = getattr(obj, key)
        indent = len(key) + 5
        coloredkey = '.' + key
        if color:
            coloredkey = set_color(coloredkey, COLOR.GREEN)
        elems += f" {coloredkey} = {objstr(val,compact=True, indent=indent, color=color)[0]}\n"
    
    return f"{header}\n{elems}{footer}"

def _isLiteral(s):
    if _isFormatStr(s): return True
    try:
        import ast as _ast
        _ast.literal_eval(s)
    except Exception:
        return False
    return True

def _isFormatStr(s):
    import re
    # 检查是否是 f-string
    pattern = r'^(f|rf|fr|Fr|fR|FR)([\'"])(.*?)\2'
    match = re.fullmatch(pattern, s)
    if bool(re.search(pattern, s)): return True
    
    # 检查是否是 % 格式化
    pattern = r'^([\'"])(.*?)%[sdxf](.*?)\1\s*%'
    if bool(re.search(pattern, s)): return True
    
    # 检查是否是 .format() 格式化
    pattern = r'^([\'"])(.+?)\1.format\((.*?)\)\s*$'
    match = re.fullmatch(pattern, s)
    if bool(match): return True
    
    return False

def unparse_tgs(node):
    res = []
    for elt in node.elts:
        if isinstance(elt, _ast.Tuple):
            res.append(unparse_tgs(elt))
        elif isinstance(elt, _ast.Name):
            res.append(elt.id)
        else:
            res.append(_ast.unparse(elt))
    return tuple(res)

def get_last_lv(call_frame) -> tuple | str | None:
    from objprint.executing import Source
    # check if call_frame is None
    if call_frame is None:
        return None

    # get node
    node = Source.executing(call_frame).node
    if node is None:
        return None
    
    # initialize stack of statements
    last_stmt = None
    lineno = _inspect.getlineno(call_frame)
    statement_node = Source.for_frame(call_frame).statements_at_line(lineno)
    if len(statement_node) > 1:
        last_stmt = sorted(statement_node, key=lambda x: x.col_offset)[-2]
    
    if last_stmt is None:
        # if no statement found, go back to the previous line
        for _ in range(10):
            lineno -= 1
            statement_node = Source.for_frame(call_frame).statements_at_line(lineno)
            if len(statement_node) > 0:
                last_stmt = sorted(statement_node, key=lambda x: x.col_offset)[-1]
                break

    if last_stmt is None:
        # return None if no statement found
        return None
    
    # find left values
    left_values = None
    if isinstance(last_stmt, _ast.Assign):
        tgs = last_stmt.targets[-1]
        if isinstance(tgs, _ast.Name):
            left_values = tgs.id
        elif isinstance(tgs, _ast.Tuple):
            left_values = unparse_tgs(tgs)
        else:
            left_values = _ast.unparse(tgs)

    return left_values

def get_lv(call_frame) -> tuple | str | None:
    from objprint.executing import Source
    # check if call_frame is None
    if call_frame is None:
        return None

    # get node
    node = Source.executing(call_frame).node
    if node is None:
        return None
    
    # initialize stack of statements
    lineno = _inspect.getlineno(call_frame)
    statement_node = Source.for_frame(call_frame).statements_at_line(lineno)
    stmt = sorted(statement_node, key=lambda x: x.col_offset)[-1] # todo: 这里可能会有问题，可能不是最后一个
    
    # find left values
    left_values = None
    if isinstance(stmt, _ast.Assign):
        tgs = stmt.targets[-1]
        if isinstance(tgs, _ast.Name):
            left_values = tgs.id
        elif isinstance(tgs, _ast.Tuple):
            left_values = unparse_tgs(tgs)
        else:
            left_values = _ast.unparse(tgs)

    return left_values

def get_vals(last_lv, call_frame) -> list:
    """获取变量的值"""
    local_vars = call_frame.f_locals
    global_vars = call_frame.f_globals
    vals = [eval(lv, global_vars, local_vars) for lv in last_lv]
    return vals

def flatten_tuple(t):
    if isinstance(t, str):
        return (t, )
    result = []
    for item in t:
        result.extend(flatten_tuple(item))  # 递归处理嵌套元组
    return tuple(result)

def get_args(frame):
    import tokenize
    import io
    if frame is None:
        return None
    func_call_str = get_executing_function_call_str(frame)
    if func_call_str is None:
        func_call_str = get_executing_function_call_str2(frame)
    if func_call_str is None:
        return None
    func_call_io = io.StringIO(func_call_str)
    depth = 0
    args = []
    curr_arg = ""
    last_pos = (0, 0)
    for token in tokenize.generate_tokens(func_call_io.readline):
        if depth == 0 and token.string == "(":
            depth = 1
        elif depth == 1 and token.string == ")":
            args.append(curr_arg.strip())
            break
        elif depth == 1 and token.string == ",":
            args.append(curr_arg.strip())
            curr_arg = ""
        elif depth >= 1:
            if token.string in "([{":
                depth += 1
            elif token.string in ")]}":
                depth -= 1
            if depth >= 1 and token.type != tokenize.NL:
                if token.start[0] != last_pos[0] or token.start[1] - last_pos[1] > 0:
                    curr_arg += f" {token.string}"
                else:
                    curr_arg += token.string
        last_pos = token.end
    return args

def get_executing_function_call_str(frame):
    from objprint.executing import Source
    node = Source.executing(frame).node
    if node is None:
        return None

    try:
        module = _inspect.getmodule(frame)
        if module is None:
            return None
        source = _inspect.getsource(module)
    except (OSError, TypeError):
        return None

    return _ast.get_source_segment(source, node)

def get_executing_function_call_str2(frame):
    from objprint.executing import Source
    node = Source.executing(frame).node
    if node is None:
        return None
    lineno = _inspect.getlineno(frame)
    statement_node = Source.for_frame(frame).statements_at_line(lineno)
    stmt = sorted(statement_node, key=lambda x: x.col_offset)[-1]
    return _ast.unparse(stmt)

class Show:
    """
    打印输入变量的名称和值到日志中。
    
    任何以 f" 或 rf" 开头的字符串会被识别为 f-string，而不会被解析为变量名称。
    
    可以通过在 builtins.pyi 中添加：

    >>> def Timer(self, *str_or_funcs, output_unit: float|None = None, save=False) -> None: ...
    >>> def show(*values) -> None: ...

    使得 ide 可以识别 show 和 Timer 函数。

    Examples
    --------
    >>> from quante.basicfun import println as show
    >>> a = "this is a test"
    >>> show(a)
    a: this is a test
    >>> a, (b, c) = 1, (2, 3)
    >>> show()
    a: 1; b: 2; c: 3
    """
    def __init__(self, use_color=True):
        self.use_color = use_color
        self.arg_name = True
    
    def __call__(self, *ipt, level=1, end=None):
        if level == 0:
            return None
        
        if self.arg_name and end is None:
            call_frame = _inspect.currentframe()  # 获取调用函数的栈帧
            if call_frame is not None:
                call_frame = call_frame.f_back
            
            if len(ipt) == 0:
                last_lv = get_last_lv(call_frame)
                if last_lv is None:
                    return ""
                last_lv = flatten_tuple(last_lv)
                vals = get_vals(last_lv, call_frame)
                out: str = self._constructArgumentOutput(last_lv, vals)
            else:
                args = get_args(call_frame)
                if args is None:
                    args = ["Unknown Arg" for _ in range(len(ipt))]
                if len(args) != len(ipt):
                    out = "\n".join(map(str, ipt))
                else:
                    out: str = self._constructArgumentOutput(args, ipt)
        else:
            if end is None:
                end = "\n"
            out = end.join(map(str, ipt))

        if level == 1:
            logger.info(out)
        elif level == -1:
            logger.debug(out)
        elif level == 2:
            logger.warning(out)
        elif level == 3:
            logger.error(out)
        else:
            logger.critical(out)
       
        logger.handlers[0].flush()  # 立即刷新日志

    def _constructArgumentOutput(self, paraname, inputargs):
        if len(paraname) == 1 and _isLiteral(paraname[0]):
            if isinstance(inputargs[0], str):
                return inputargs[0] # 这样不会显示引号
            return objstr(inputargs[0], color=self.use_color)[0]
        
        pairs = [(arg, *objstr(val, color=self.use_color)) 
                 for arg, val in zip(paraname, inputargs)]
        
        pairStrs = []
        for arg, val, isobject in pairs:
            if _isLiteral(arg):
                pairStrs.append(val)
            else:
                coloredarg = f"{arg}: "
                if self.use_color:
                    coloredarg = set_color(coloredarg, COLOR.RED)
                pairStrs.append(coloredarg + val)

        seperator = set_color("; ", COLOR.RED) if self.use_color else "; "
        allArgsOnOneLine = seperator.join(pairStrs)
        
        multilineArgs = len(allArgsOnOneLine.splitlines()) > 1
        firstLineTooLong = len(allArgsOnOneLine.splitlines()[0]) > 70
        
        if multilineArgs or firstLineTooLong:
            lines = []
            for i, (arg, value, isobject) in enumerate(pairs):
                prefixTooLong =  (len(pairs[i][0]) > 8) or isobject
                lines.append(self._format_pair(arg, value, prefixTooLong))
        else:
            lines = [allArgsOnOneLine]
        
        return "\n".join(lines)

    def _format_pair(self, arg, value, prefixTooLong=False):
        arg_lines = self._indented_lines("", arg)
        value_prefix = arg_lines[-1] + ": "
        
        # str如果分行，那么整体退一格
        looksLikeAString = value[0] + value[-1] in ["''", '""']
        if looksLikeAString:  # Align the start of multiline strings.
            lines = value.splitlines(True)
            for i in range(1, len(lines)):
                lines[i] = " " + lines[i]
            value = "".join(lines)

        value_lines = self._indented_lines(value_prefix, value, prefixTooLong=prefixTooLong)
        lines = arg_lines[:-1] + value_lines
        return "\n".join(lines)

    def _indented_lines(self, prefix, string, prefixTooLong=False):
        lines = string.splitlines()
        prefixlen = len(prefix)
        
        if bool(prefix.strip()) and self.use_color:
            prefix = set_color(prefix, COLOR.RED)
            
        if prefixTooLong:
            lth = 3
            return [prefix + '\n' + ' '*lth + lines[0]] + [" " * lth + line for line in lines[1:]]
        else:
            return [prefix + lines[0]] + [" " * prefixlen + line for line in lines[1:]]

println = show = Show()
def set_show(use_color=None, arg_name=None) -> None:
    if use_color is not None:
        println.use_color = use_color
    if arg_name is not None:
        println.arg_name = arg_name

info = Show()
info.arg_name = False # 不显示参数名称

def send_email(
    subject: str, 
    body: str, 
    to_email: str, 
    from_email: str, 
    smtp_server: str, 
    smtp_port: int, 
    login: str, 
    password: str
):
    r"""发送程序执行完成的邮件
    
    Parameters
    ----------
    subject : str
        邮件主题
    body : str
        邮件正文
    to_email : str
        收件人邮箱地址
    from_email : str
        发件人邮箱地址
    smtp_server : str
        SMTP 服务器地址（如'smtp.gmail.com'）
    smtp_port : int
        SMTP 端口号（通常 587 或 465）
    login : str
        SMTP 登录用户名（通常是发件人邮箱）
    password : str
        SMTP 登录密码或授权码
        
    Returns
    -------
    None

    Example
    -------
    >>> # 配置 SMTP 信息和邮件内容
    >>> smtp_server = "smtp.xxx.edu.cn"  # 例如 Gmail 的 SMTP 服务器
    >>> smtp_port = 25  # Gmail 使用的 TLS 端口
    >>> from_email = "xxxxxxxxxx@xxxx.edu.cn"
    >>> to_email = "xxxxxxxxxxx@outlook.com"
    >>> login = "xxxxxxxxxxx@xxx.edu.cn"
    >>> password = "xxxxxxxxxxxx"  # 应用专用密码或邮箱密码
    >>> 
    >>> # 自定义邮件内容
    >>> subject = "程序执行完成通知"
    >>> body = "您的程序已经成功运行完成！"
    >>> 
    >>> try:
    >>>     pass  # 程序运行代码
    >>>     body += "\n\n程序运行成功！"
    >>> except Exception as e:
    >>>     body += f"\n\n出现错误：\n{e}"
    >>> finally:
    >>>     send_email(subject, body, to_email, from_email, smtp_server, smtp_port, login, password)
    """
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    import traceback

    try:
        # 创建邮件
        message = MIMEMultipart()
        message["From"] = from_email
        message["To"] = to_email
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain"))  # 可以将 "plain" 替换为 "html" 发送 HTML 格式邮件

        # 连接 SMTP 服务器并发送邮件
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # 启用 TLS 加密
            server.login(login, password)
            server.sendmail(from_email, to_email, message.as_string())
            print(f"邮件已发送至 {to_email}")
    except Exception as e:
        print("发送邮件时发生错误：")
        traceback.print_exc()

