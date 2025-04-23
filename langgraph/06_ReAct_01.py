from typing import List
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI

embeddings_model = OllamaEmbeddings(model="bge-m3")

# Chroma 인덱스 로드
vector_db = Chroma(
     embedding_function=embeddings_model,
     collection_name="restaurant_menu",
     persist_directory="./chroma_db",  # Chroma DB가 저장된 경로
)

# Tool 정의
@tool
def search_menu(query: str) -> List[str]:
    """ 레스트랑 메뉴에서 정보를 검색합니다. """
    docs = vector_db.similarity_search(query, k=2)

    formatted_docs = "\n\n---\n\n".join(
        [
            f'<document source="{doc.metadata["source"]}"/>\n{doc.page_content}\n</Document>'
            for doc in docs
        ]
    )

    if len(docs) > 0:
        return formatted_docs
    
    return "관련 메뉴 정보를 찾을 수 없습니다."

@tool
def search_web(query: str) -> List[str]:
    """ 데이터베이스에 존재하지 않는 정보 또는 최신 정보를 인터넷에서 검색합니다."""

    tavily_search = TavilySearchResults(max_results=3)
    docs = tavily_search.invoke(query)

    formatted_docs = "\n\n---\n\n".join(
        [
            f'<document source="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in docs
        ]
    )

    if len(docs) > 0:
        return formatted_docs
    
    return "관련 웹 정보를 찾을 수 없습니다."

# LLM 모델
llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)

# 도구 목록
tools = [
    search_menu,
    search_web
]

# 모델에 도구를 바인딩
llm_with_tools = llm.bind_tools(tools)

# 도구 호출
tool_call = llm_with_tools.invoke([HumanMessage(content="스테이크 메뉴의 가격은 얼마인가요?")])

# 결과 출력
print(tool_call)

# 도구 호출 (일반호출)
tool_call = llm_with_tools.invoke([HumanMessage(content="LangGraph는 무엇인가요?")])

# 결과 출력
print(tool_call)

# 도구 호출 (덧셈 연산)
tool_call = llm_with_tools.invoke([HumanMessage(content="3 + 3은 얼마인가요?")])

# 결과 출력
print(tool_call)
