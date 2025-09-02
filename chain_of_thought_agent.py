from typing import List, Dict, Any, Optional, Type, Union
from enum import Enum
from pydantic import BaseModel, Field
import json
from dotenv import load_dotenv
import os
from openai import OpenAI

# 加载环境变量
load_dotenv()

# 初始化OpenAI客户端
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ToolType(str, Enum):
    """工具类型"""
    CALCULATOR = "calculator"
    SEARCH = "search"

class ToolCall(BaseModel):
    """工具调用"""
    name: str
    arguments: Dict[str, Any]

class Thought(BaseModel):
    """思考过程"""
    thought: str
    reasoning: str
    plan: List[str]

class Observation(BaseModel):
    """观察结果"""
    content: str
    tool_calls: Optional[List[ToolCall]] = None

class Action(BaseModel):
    """行动"""
    tool: str
    tool_input: Dict[str, Any]

class AgentState(BaseModel):
    """Agent状态"""
    thoughts: List[Thought] = Field(default_factory=list)
    observations: List[Observation] = Field(default_factory=list)
    actions: List[Action] = Field(default_factory=list)
    final_answer: Optional[str] = None

class BaseTool(BaseModel):
    """工具基类"""
    name: str
    description: str
    parameters: Dict[str, Any]
    
    def run(self, **kwargs) -> str:
        raise NotImplementedError

class CalculatorTool(BaseTool):
    """计算器工具"""
    def __init__(self):
        super().__init__(
            name="calculator",
            description="用于执行数学计算",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的数学表达式，例如：3 + 5 * 2"
                    }
                },
                "required": ["expression"]
            }
        )
    
    def run(self, expression: str) -> str:
        try:
            # 安全地计算数学表达式
            result = eval(expression, {"__builtins__": {}}, {})
            return f"计算结果: {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"

class SearchTool(BaseTool):
    """搜索工具"""
    def __init__(self):
        super().__init__(
            name="search",
            description="用于搜索信息",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词或问题"
                    }
                },
                "required": ["query"]
            }
        )
    
    def run(self, query: str) -> str:
        # 这里可以接入实际的搜索API
        # 为了简化，我们返回模拟数据
        return f"这是关于'{query}'的搜索结果。在实际应用中，这里会返回真实的搜索结果。"

class CotAgent:
    """思维链Agent"""
    
    def __init__(self, tools: List[BaseTool] = None):
        self.tools = tools or [CalculatorTool(), SearchTool()]
        self.state = AgentState()
        self.max_iterations = 5
        
    def _get_system_prompt(self) -> str:
        """获取系统提示"""
        return """你是一个智能助手，使用思维链(Chain of Thought)来解决问题。
        请按照以下步骤进行思考：
        1. 分析问题并制定解决计划
        2. 如果需要使用工具，请明确说明要使用哪个工具以及输入参数
        3. 根据工具的返回结果进行观察和分析
        4. 最终给出明确的答案
        
        你可以使用的工具：
        {tools}
        
        请始终按照以下格式输出：
        ```
        思考：<你的思考过程>
        计划：
        - 第一步
        - 第二步
        ...
        
        行动：
        ```json
        {{
            "tool": "工具名称",
            "tool_input": {{"参数名": "参数值"}}
        }}
        ```
        
        观察：<工具返回的结果>
        
        最终答案：<你的最终回答>
        """.format(
            tools="\n".join([f"- {tool.name}: {tool.description}" for tool in self.tools])
        )
    
    def _extract_action(self, text: str) -> Optional[Dict[str, Any]]:
        """从文本中提取行动"""
        try:
            action_start = text.find("```json\n") + 8
            action_end = text.find("\n```", action_start)
            if action_start < 8 or action_end < 0:
                return None
                
            action_json = text[action_start:action_end].strip()
            return json.loads(action_json)
        except Exception as e:
            print(f"解析行动时出错: {e}")
            return None
    
    def _extract_thought(self, text: str) -> str:
        """从文本中提取思考过程"""
        thought_start = text.find("思考：") + 3
        thought_end = text.find("\n\n", thought_start)
        if thought_start < 3:
            return ""
        if thought_end < 0:
            thought_end = len(text)
            
        return text[thought_start:thought_end].strip()
    
    def _extract_final_answer(self, text: str) -> Optional[str]:
        """从文本中提取最终答案"""
        if "最终答案：" not in text:
            return None
            
        answer_start = text.find("最终答案：") + 5
        return text[answer_start:].strip()
    
    def _run_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """运行工具"""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool.run(**tool_input)
        return f"错误：找不到工具 '{tool_name}'"
    
    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """调用语言模型"""
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"调用语言模型时出错: {str(e)}"
    
    def run(self, query: str) -> str:
        """运行Agent"""
        print(f"\n{'='*50}\n开始处理查询: {query}\n{'='*50}")
        
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": query}
        ]
        
        for i in range(self.max_iterations):
            print(f"\n--- 第 {i+1} 轮思考 ---")
            
            # 1. 生成思考
            response = self._call_llm(messages)
            print(f"\n思考过程：\n{response}")
            
            # 2. 提取思考
            thought = self._extract_thought(response)
            if thought:
                self.state.thoughts.append(Thought(
                    thought=thought,
                    reasoning="",
                    plan=[]
                ))
            
            # 3. 检查是否有最终答案
            final_answer = self._extract_final_answer(response)
            if final_answer:
                self.state.final_answer = final_answer
                print(f"\n✅ 最终答案: {final_answer}")
                return final_answer
            
            # 4. 提取并执行行动
            action = self._extract_action(response)
            if not action:
                print("⚠️ 未找到有效的行动")
                continue
                
            tool_name = action.get("tool")
            tool_input = action.get("tool_input", {})
            
            print(f"\n🛠️ 执行工具: {tool_name}")
            print(f"   输入: {tool_input}")
            
            # 记录行动
            self.state.actions.append(Action(
                tool=tool_name,
                tool_input=tool_input
            ))
            
            # 执行工具
            observation = self._run_tool(tool_name, tool_input)
            print(f"👀 观察结果: {observation}")
            
            # 记录观察结果
            self.state.observations.append(Observation(
                content=observation,
                tool_calls=[ToolCall(name=tool_name, arguments=tool_input)]
            ))
            
            # 更新消息历史
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"观察：{observation}"})
        
        return "达到最大迭代次数，未能找到答案。"

if __name__ == "__main__":
    # 创建并运行Agent
    agent = CotAgent()
    
    # 示例查询
    queries = [
        "计算一下 15 的平方加上 25 的平方等于多少？",
        "搜索一下人工智能的最新发展"
    ]
    
    for query in queries:
        print("\n" + "="*50)
        print(f"处理查询: {query}")
        print("="*50)
        result = agent.run(query)
        print(f"\n处理完成。最终结果: {result}")
        
        # 重置Agent状态
        agent.state = AgentState()
