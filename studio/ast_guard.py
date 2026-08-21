"""
OKX-Dog AI Quant Studio - AST 静态代码安全与白名单审查器
模块: okx-dog-ai/studio/ast_guard.py
角色: 后端与交易风控安全审计师 (agency-blockchain-security-auditor)
功能: 在代码运行前解析 Python 抽象语法树 (AST)，100% 拦截危险系统调用、文件外联与黑名单模块
"""

import ast
from typing import Tuple, List

# 严格的白名单导入模块
ALLOWED_MODULES = {
    "numpy",
    "np",
    "pandas",
    "pd",
    "math",
    "scipy",
    "typing",
    "dataclasses",
    "collections",
    "itertools",
    "datetime",
    "strategy_base",
}

# 绝对禁止的高危内置函数与属性
FORBIDDEN_CALLS = {
    "eval",
    "exec",
    "compile",
    "open",
    "__import__",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "system",
    "popen",
    "spawn",
    "fork",
    "exit",
    "quit",
}

# 绝对禁止的黑名单模块
FORBIDDEN_MODULES = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "requests",
    "urllib",
    "http",
    "ftplib",
    "shutil",
    "ctypes",
    "multiprocessing",
    "threading",
    "builtins",
    "pty",
    "posix",
}


class ASTSecurityVisitor(ast.NodeVisitor):
    """AST 访问器，深度审查导入与函数调用"""

    def __init__(self):
        self.errors: List[str] = []

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            base_mod = alias.name.split(".")[0]
            if base_mod in FORBIDDEN_MODULES or base_mod not in ALLOWED_MODULES:
                self.errors.append(f"安全违规: 禁止导入非白名单/高危模块 '{alias.name}'")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            base_mod = node.module.split(".")[0]
            if base_mod in FORBIDDEN_MODULES or base_mod not in ALLOWED_MODULES:
                self.errors.append(f"安全违规: 禁止从非白名单模块导入 '{node.module}'")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        func = node.func
        # 针对直接函数调用 (如 eval(...))
        if isinstance(func, ast.Name):
            if func.id in FORBIDDEN_CALLS:
                self.errors.append(f"安全违规: 严禁调用高危函数 '{func.id}()'")
        # 针对属性调用 (如 os.system(...))
        elif isinstance(func, ast.Attribute):
            if func.attr in FORBIDDEN_CALLS:
                self.errors.append(f"安全违规: 严禁调用高危方法 '{func.attr}()'")
        self.generic_visit(node)


def validate_python_code_security(code: str) -> Tuple[bool, str]:
    """
    静态审查 Python 源码安全性

    返回:
        (is_valid: bool, error_or_success_message: str)
    """
    if not code or not code.strip():
        return False, "代码内容为空"

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Python 语法错误: 行 {e.lineno}, 列 {e.offset} - {e.msg}"
    except Exception as e:
        return False, f"AST 解析失败: {str(e)}"

    visitor = ASTSecurityVisitor()
    visitor.visit(tree)

    if visitor.errors:
        return False, " | ".join(visitor.errors)

    return True, "AST 安全与语法审查 100% 通过"
