"""
PDE Neural Operator Research Agent - 启动脚本

使用方式：
  1. 首次运行，生成配置文件：
     python run_agent.py --init-config

  2. 编辑 config.yaml，填入 LLM API Key

  3. 运行 Agent：
     python run_agent.py --task task1
     python run_agent.py --task task2
     python run_agent.py --task task3

  4. 或直接通过环境变量/命令行传入：
     OPENAI_API_KEY=sk-xxx python run_agent.py --task task1

说明：
  - task1/task2/task3 的数据目录由 Agent 内部固定解析为 PDEAgent/data/task{N}
  - config.yaml 仅保留 llm 与 research 流程配置；模型训练超参不再由该文件驱动
"""
import sys
import os

# 确保可以导入 agent 包
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.main import main

if __name__ == "__main__":
    main()
