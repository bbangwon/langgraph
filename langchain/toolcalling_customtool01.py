from langchain_community.tools import TavilySearchResults
from langchain_core.tools import tool
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

# 도구 속성
print("자료형 : ")
print(type(search_web))
print("-"*100)

print("도구 이름 : ")
print(search_web.name)
print("-"*100)

print("도구 설명 : ")
print(search_web.description)
print("-"*100)

print("도구 인자 : ")
pprint(search_web.args_schema.model_json_schema())
print("-"*100)