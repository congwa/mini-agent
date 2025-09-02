import json
from typing import Dict, List, Optional, Any, Generator, Tuple
from enum import Enum
from pydantic import BaseModel, Field
from openai import OpenAI
import os
from dotenv import load_dotenv

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

class Message(BaseModel):
    """消息"""
    role: str  # 'system', 'user', 'assistant', 'tool'
    content: str
    tool_calls: List[ToolCall] = Field(default_factory=list)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

class Tool(BaseModel):
    """工具基类"""
    name: str
    description: str
    parameters: Dict[str, Any]
    
    def execute(self, **kwargs) -> str:
        raise NotImplementedError

class CalculatorTool(Tool):
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
    
    def execute(self, expression: str) -> str:
        try:
            # 安全地计算数学表达式
            result = eval(expression, {"__builtins__": {}}, {})
            return f"计算结果: {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"

class SearchTool(Tool):
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
    
    def execute(self, query: str) -> str:
        # 这里可以接入实际的搜索API
        # 为了简化，我们返回模拟数据
        return f"这是关于'{query}'的搜索结果。在实际应用中，这里会返回真实的搜索结果。"

class FunctionCallAgent:
    """函数调用Agent"""
    
    def __init__(self, tools: List[Tool] = None):
        self.tools = tools or [CalculatorTool(), SearchTool()]
        self.conversation_history: List[Message] = []
        self.max_iterations = 5
        
    def _get_system_prompt(self) -> str:
        """获取系统提示"""
        return """你是一个智能助手，可以使用工具来帮助用户解决问题。
        你可以使用的工具：
        {tools}
        
        当需要调用工具时，请按照工具定义的格式提供参数。
        工具调用完成后，我会将结果返回给你。
        """.format(
            tools="\n".join([f"- {tool.name}: {tool.description}" for tool in self.tools])
        )
    
    def _get_tools_schema(self) -> List[Dict]:
        """获取工具的模式定义"""
        return [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
        } for tool in self.tools]
    
    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """执行工具调用"""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool.execute(**arguments)
        return f"错误：找不到工具 '{tool_name}'"
    
    def _process_tool_calls(self, tool_calls: List[Dict]) -> List[Dict]:
        """处理工具调用"""
        tool_responses = []
        for tool_call in tool_calls:
            function = tool_call.function
            tool_name = function.name
            try:
                arguments = json.loads(function.arguments)
                print(f"\n🛠️ 执行工具: {tool_name}")
                print(f"   输入: {arguments}")
                
                # 执行工具
                result = self._execute_tool(tool_name, arguments)
                print(f"👀 工具结果: {result}")
                
                # 记录工具响应
                tool_responses.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": tool_name,
                    "content": result
                })
                
            except Exception as e:
                error_msg = f"执行工具 {tool_name} 时出错: {str(e)}"
                print(f"❌ {error_msg}")
                tool_responses.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": tool_name,
                    "content": error_msg
                })
        
        return tool_responses
    
    def run(self, query: str) -> Generator[str, None, None]:
        """运行Agent"""
        print(f"\n{'='*50}\n开始处理查询: {query}\n{'='*50}")
        
        # 初始化对话历史
        self.conversation_history = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": query}
        ]
        
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n--- 第 {iteration} 轮处理 ---")
            
            # 1. 调用模型
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=self.conversation_history,
                tools=self._get_tools_schema(),
                tool_choice="auto",
                stream=True
            )
            
            # 2. 处理流式响应
            response_content = ""
            tool_calls = []
            
            for chunk in response:
                delta = chunk.choices[0].delta
                
                # 收集文本内容
                if delta.content:
                    response_content += delta.content
                    yield delta.content
                
                # 收集工具调用
                if delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        if len(tool_calls) <= tool_call.index:
                            tool_calls.append({
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": ""
                                }
                            })
                        
                        # 更新工具调用参数
                        if tool_call.function.arguments:
                            tool_calls[tool_call.index]["function"]["arguments"] += tool_call.function.arguments
            
            # 3. 更新对话历史
            if response_content:
                self.conversation_history.append({
                    "role": "assistant", 
                    "content": response_content
                })
            
            # 4. 处理工具调用
            if tool_calls:
                # 将工具调用添加到对话历史
                self.conversation_history.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": tc["type"],
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"]
                            }
                        } for tc in tool_calls
                    ]
                })
                
                # 执行工具调用
                tool_responses = self._process_tool_calls([
                    type('obj', (), {'id': tc['id'], 'function': type('func', (), tc['function'])})
                    for tc in tool_calls
                ])
                
                # 将工具响应添加到对话历史
                for response in tool_responses:
                    self.conversation_history.append(response)
                
                # 继续下一轮处理
                continue
            
            # 5. 如果没有工具调用，处理完成
            if response_content:
                print(f"\n💡 最终回答: {response_content}")
                return
        
        print("\n⚠️ 达到最大迭代次数，处理结束")

if __name__ == "__main__":
    # 创建并运行Agent
    agent = FunctionCallAgent()
    
    # 示例查询
    queries = [
        "计算一下 15 的平方加上 25 的平方等于多少？",
        "搜索一下人工智能的最新发展"
    ]
    
    for query in queries:
        print("\n" + "="*50)
        print(f"处理查询: {query}")
        print("="*50)
        
        print("\nAgent回复: ", end="")
        for chunk in agent.run(query):
            print(chunk, end="", flush=True)
        
        print("\n\n对话历史:")
        for msg in agent.conversation_history:
            role = msg["role"]
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])
            
            if role == "user":
                print(f"\n用户: {content}")
            elif role == "assistant":
                if content:
                    print(f"\n助手: {content}")
                for tc in tool_calls:
                    print(f"\n助手调用工具: {tc['function']['name']}")
                    print(f"参数: {tc['function']['arguments']}")
            elif role == "tool":
                print(f"\n工具({msg['name']}): {content}")
        
        print("\n" + "-"*50 + "\n")
