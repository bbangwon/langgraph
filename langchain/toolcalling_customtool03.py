from langchain_community.tools import TavilySearchResults
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from typing import List
from pprint import pprint

# Tool 정의
@tool
def search_web(query: str) -> List[str]:
    """Searches the internet for information that does not exist in the database or for the latest information."""

    tavily_search = TavilySearchResults(max_results=2)
    docs = tavily_search.invoke(query) 

    # 검색결과는 딕셔너리 객체를 가지고 있는의 리스트
    # 이후 프롬프트 전달시 문자열 포매팅을 해야 함

    formatted_docs = "\n---\n".join([   # 구분선을 이용해 각각의 검색결과를 구분
        f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
        for doc in docs
    ])

    if(len(formatted_docs) > 0):
        return formatted_docs
    
    return "관련 정보를 찾을 수 없습니다."


llm = ChatOpenAI(model = "gpt-4o-mini")

# LLM에 도구를 바인딩
llm_with_tool = llm.bind_tools([search_web])

query = "스테이크와 어울리는 와인을 추천해주세요."
ai_msg = llm_with_tool.invoke(query)

# LLM의 전체 출력 결과 출력
pprint(ai_msg)
print("-"*100)

# 메시지 content 속성 (텍스트 출력)
pprint(ai_msg.content)
print("-"*100)

# LLM이 호출한 도구 정보 출력
pprint(ai_msg.tool_calls)
print("-"*100)
