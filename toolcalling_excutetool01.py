import warnings
from pprint import pprint

from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage

warnings.filterwarnings("ignore")

#llm 모델 추가
llm = ChatOpenAI(model = "gpt-4o-mini")

# Tavily 검색 도구 초기화 (최대 2개의 결과 반환)
web_search = TavilySearchResults(max_results=2)

# 웹 검색 도구를 직접 LLM에 바인딩
llm_with_tools = llm.bind_tools([web_search])

# 도구 호출이 필요 없는 LLM 호출을 수행
query = "스테이크와 어울리는 와인을 추천해주세요."
ai_msg = llm_with_tools.invoke(query)

### 방법 1: 직접 도구 호출 처리
# 이 방법은 AI 메시지에서 첫 번째 도구 호출을 가져와 직접 처리
# 'args'를 사용하여 도구를 호출하고 결과를 얻는다.

tool_call = ai_msg.tool_calls[0]
tool_output = web_search.invoke(tool_call["args"])
print(f"{tool_call['name']} 호출 결과:")
print("-" * 100)
print(tool_output)

### 방법 2: ToolMessage 객체 생성
# 이 방법은 도구 호출 결과를 사용하여 ToolMessage 객체를 생성
# 도구 호출의 ID와 이름을 포함하여 더 구조화된 메시지를 만든다.
tool_message = ToolMessage(
    content=tool_output,
    tool_call_id=tool_call["id"],
    name=tool_call["name"],
)
print("-" * 100)
print(tool_message)

### 방법 3: 도구 직접 호출하여 바로 ToolMessage 객체 생성

# 이 방법은 도구를 직접 호출하여 ToolMessage 객체를 생성
# 가장 간단하고 직관적인 방법으로, Langchain의 추상화를 활용한다.
print("-" * 100)
tool_message = web_search.invoke(tool_call)
print(tool_message)
print("-" * 100)
pprint(tool_message.content)
print("-" * 100)
pprint(tool_message.tool_call_id)
print("-" * 100)
pprint(tool_message.name)