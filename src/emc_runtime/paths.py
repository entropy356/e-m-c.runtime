"""路径参数的安全收口。

``--config``、scenario 文件、``--output`` 这些路径可能由外部输入或模型输出
构造，统一在这里规范化：展开 ``~``、消除 ``.``/``..``、解析符号链接，
并拒绝空字节与明显不合理的形态。

这样既消除了 path traversal 的歧义（Sonar ``pythonsecurity:S8707``），
也让所有读写入口的校验规则只有一份。
"""

from __future__ import annotations

from pathlib import Path


class PathSecurityError(ValueError):
    """路径不满足安全约束。"""


def _reject_unsafe(raw: str | Path) -> str:
    text = str(raw).strip()
    if not text:
        raise PathSecurityError("路径为空")
    if "\x00" in text:
        raise PathSecurityError("路径包含空字节")
    return text


def normalize_path(raw: str | Path) -> Path:
    """展开 ``~``、消除 ``.`` 与 ``..``、解析符号链接，返回绝对路径。"""
    return Path(_reject_unsafe(raw)).expanduser().resolve()


def readable_file(raw: str | Path) -> Path:
    """读取入口（配置、剧本）：必须是已存在的普通文件。"""
    path = normalize_path(raw)
    if not path.exists():
        raise PathSecurityError(f"路径不存在: {path}")
    if not path.is_file():
        raise PathSecurityError(f"不是文件: {path}")
    return path


def writable_file(raw: str | Path) -> Path:
    """写入入口（``--output``）：不能指向已存在的目录。"""
    path = normalize_path(raw)
    if path.exists() and path.is_dir():
        raise PathSecurityError(f"目标是目录: {path}")
    return path
