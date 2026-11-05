"""hermes-context-engine：Hermes Agent 的上下文引擎插件（注册入口）。

独立插件目录：``hermes plugins install rays1d/hermes-context-engine``（或把本目录放进
``<hermes-home>/plugins/``），再在 config.yaml 设 ``context.engine: context-engine`` 即
启用——无需 ``plugins.enabled``（context engine 有专用发现通道）。

模块布局：
    __init__.py     注册入口（本文件）
    engine.py       HermesContextEngine — pass 编排 / 重建许可 / compress / LKG
    units.py        单元识别：身份键、工具弧配对、文件 part（纯函数）
    render.py       请求副本渲染：前缀 / 占位符 / 移除手术 / 时间标记（纯函数）
    estimate.py     单一 token 估算函数 + 校准因子
    store.py        SQLite 存储（schema 栅栏 / 单事务 / 会话锁）
    tools.py        ctx_reduce / ctx_expand — schema 与四态文案契约
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _build_engine():
    from .engine import HermesContextEngine

    return HermesContextEngine()


def register_context_engine(ctx) -> None:
    """Context engine discovery entry (dedicated channel + general plugin system).

    Called once per load; each call yields a fresh instance so per-agent cloning
    starts from clean mutable state while all durable state stays in the store.
    """
    ctx.register_context_engine(_build_engine())


def register(ctx) -> None:
    """General plugin system entry (plugin.yaml path)."""
    register_context_engine(ctx)
