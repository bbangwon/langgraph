import warnings
from pprint import pprint

from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI

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

# LLM의 전체 출력 결과 출력
pprint(ai_msg)
print("-" * 100)

# 메시지 content 속성 (텍스트 출력)
pprint(ai_msg.content)
print("-" * 100)

# LLM이 호출한 도구 정보 출력
pprint(ai_msg.tool_calls)
print("-" * 100)