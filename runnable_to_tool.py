from langchain_community.document_loaders import WikipediaLoader
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field
from typing import List
from textwrap import dedent
from pprint import pprint
from langchain_openai import ChatOpenAI
from langchain_community.tools import TavilySearchResults
from langchain_core.tools import tool

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


# WikipediaLoader를 사용하여 여러개의 위키피디아 문서를 검색하는 함수
def search_wiki(input_data: dict) -> List[Document]:
    """Search Wikipedia documents based on user input (query) and return k documents"""
    query = input_data["query"]
    k = input_data.get("k", 2)
    wiki_loader = WikipediaLoader(query=query, load_max_docs=k, lang="ko")
    wiki_docs = wiki_loader.load()
    return wiki_docs

# 도구 호출에 사용할 입력 스키마 정의
class WilkiSearchSchema(BaseModel):
    """Input schema for Wikipedia search."""
    query: str = Field(..., description="The query to search for in Wikipedia")
    k: int = Field(2, description="The number of documents to return (default is 2)")

# RunnableLamda 함수를 사용하여 위키피디아 문서 로더를 Runnable로 변환
# 입력 스키마를 설정해주면 도구의 성능을 높일수 있음
runnable = RunnableLambda(search_wiki)
wiki_search = runnable.as_tool(
    name="wiki_search",
    description=dedent("""
        Use this tool when you need to search for information on Wikipedia.
        It searches for Wikipedia articles related to the user's query and returns
        a specified number of documents. This tool is useful when general knowledge
        or background information is required.
    """),
    args_schema=WilkiSearchSchema,
)

# # 도구 속성 출력
# print("자료형 : ")
# print(type(wiki_search))
# print("-"*100)

# print("name: ")
# print(wiki_search.name)
# print("-"*100)

# print("description: ")
# pprint(wiki_search.description)
# print("-"*100)

# print("args_schema: ")
# pprint(wiki_search.args_schema.model_json_schema())
# print("-"*100)

# 위키 검색 실행
# query = "파스타의 유래"
# wiki_results = wiki_search.invoke({"query": query})

# # 검색 결과 출력
# for result in wiki_results:
#     print(result)
#     print("-"*100)

llm = ChatOpenAI(model="gpt-4o-mini")

# LLM에 도구를 바인딩
llm_with_tools = llm.bind_tools(tools=[search_web, wiki_search])

# 도구 호출이 필요한 LLM 호출을 수행
query = "서울 강남의 유명한 파스타 맛집은 어디인가요? 그리고 파스타의 유래를 알려주세요."
ai_msg = llm_with_tools.invoke(query)

#LLM의 전체 출력 결과 출력
pprint(ai_msg)
print("-"*100)

# 메시지 content 속성 (텍스트 출력)
pprint(ai_msg.content)
print("-"*100)

# LLM이 호출한 도구 정보 출력
pprint(ai_msg.tool_calls)
print("-"*100)
