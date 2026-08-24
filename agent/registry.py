"""
OKX-Dog 量化智能体角色可插拔注册中心与分层系统 (Pluggable Role Registry)
模块: okx-dog-ai/agent/registry.py
角色: 多智能体交易协同架构师 (agency-multi-agent-systems-architect)

设计模式与架构目标:
1. 像对冲基金企业“招聘新员工”一样简单：未来新增任何分析专家（如期权波动率、推特社媒舆情、MEV防夹），
   只需继承 BaseSpecialist 并使用 @register_specialist 装饰器；
2. 自动拓扑装配：LangGraph 图编译器自动检索已注册的感知层专家，并构建动态并行 Fan-Out / Fan-In 边；
3. 动态状态隔离与无损挂载：新角色的输出自动通过 Reducer 归入 specialist_outputs[name] 中，无需频繁修改全局 State 模型。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Coroutine, Dict, List, Optional, Type

from .state import QuantTraderState

logger = logging.getLogger("okx_dog.ai.agent.registry")


class BaseSpecialist(ABC):
    """
    量化感知/分析专家标准抽象基类
    """
    name: str = "base_specialist"
    stage_name: str = "基础分析专家"
    layer: str = "perception"  # perception (感知层), adversarial (对抗层), risk (风控层)
    description: str = "量化分析专家基类"

    @abstractmethod
    async def analyze(self, state: QuantTraderState) -> Dict[str, Any]:
        """
        执行专业量化分析并返回该专家的状态更新字典。
        必须包含:
        - 专家的核心分析数据字典 (如 macro_regime, onchain_analysis 等)
        - thinking_steps: [ThinkingStep, ...] 供思维链实时追踪
        """
        pass

    def get_node_function(self) -> Callable[[QuantTraderState], Coroutine[Any, Any, Dict[str, Any]]]:
        """将专家实例包装为 LangGraph 兼容的异步 Node 函数"""
        async def _node_wrapper(state: QuantTraderState) -> Dict[str, Any]:
            logger.info("执行专家节点: [%s] (%s)", self.name, self.stage_name)
            result = await self.analyze(state)
            
            # 将当前专家的产物作为独立分片，利用 StateGraph 的 merge_dict_reducer 进行并发合并
            result["specialist_outputs"] = {
                self.name: {k: v for k, v in result.items() if k not in ["thinking_steps", "specialist_outputs"]}
            }
            return result

        _node_wrapper.__name__ = f"{self.name}_node"
        _node_wrapper.__doc__ = self.description
        return _node_wrapper


class AgentRoleRegistry:
    """
    智能体角色单例注册中心
    """
    _specialists: Dict[str, BaseSpecialist] = {}
    _specialist_classes: Dict[str, Type[BaseSpecialist]] = {}

    @classmethod
    def register(cls, specialist_cls: Type[BaseSpecialist]) -> Type[BaseSpecialist]:
        """类装饰器：注册一个专业专家角色"""
        instance = specialist_cls()
        cls._specialists[instance.name] = instance
        cls._specialist_classes[instance.name] = specialist_cls
        logger.info("成功注册智能体专家角色: [%s] (层级: %s, 阶段: %s)", instance.name, instance.layer, instance.stage_name)
        return specialist_cls

    @classmethod
    def get(cls, name: str) -> Optional[BaseSpecialist]:
        """获取指定名称的专家实例"""
        return cls._specialists.get(name)

    @classmethod
    def get_specialists_by_layer(cls, layer: str = "perception") -> List[BaseSpecialist]:
        """获取指定层级的所有专家实例"""
        return [s for s in cls._specialists.values() if s.layer == layer]

    @classmethod
    def get_all_specialists(cls) -> Dict[str, BaseSpecialist]:
        """获取所有已注册的专家实例"""
        return dict(cls._specialists)

    @classmethod
    def clear(cls) -> None:
        """清空注册中心 (供单元测试使用)"""
        cls._specialists.clear()
        cls._specialist_classes.clear()


def register_specialist(specialist_cls: Type[BaseSpecialist]) -> Type[BaseSpecialist]:
    """专家注册便捷装饰器"""
    return AgentRoleRegistry.register(specialist_cls)
