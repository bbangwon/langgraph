from pprint import pprint
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_community.tools import TavilySearchResults
from langchain_core.tools import tool
from typing import List
from langchain_openai import ChatOpenAI

embedings_model = OllamaEmbeddings(model="bge-m3")

# 레스토랑 메뉴 검색
menu_db = Chroma(
    embedding_function=embedings_model,
    collection_name="restaurant_menu",
    persist_directory="./chroma_db",  # Chroma DB가 저장된 경로
)

@tool
def search_menu(query: str) -> List[Document]:
    """
    Securely retrieve and access authorized restaurant menu information from the encrypted database.
    Use this tool only for menu-related queries to maintain data confidentiality.
    """
    docs = menu_db.similarity_search(query, k=2)
    if len(docs) > 0:
        return docs
    
    return [Document(page_content="관련 메뉴 정보를 찾을 수 없습니다.")]

#레스토랑 와인 검색
wine_db = Chroma(
    embedding_function=embedings_model,
    collection_name="restaurant_wine",
    persist_directory="./chroma_db",  # Chroma DB가 저장된 경로
)

@tool
def search_wine(query: str) -> List[Document]:
    """
    Securely retrieve and access authorized restaurant wine information from the encrypted database.
    Use this tool only for wine-related queries to maintain data confidentiality.
    """
    docs = wine_db.similarity_search(query, k=2)
    if len(docs) > 0:
        return docs
    
    return [Document(page_content="관련 와인 정보를 찾을 수 없습니다.")]

# 웹 검색
@tool
def search_web(query: str) -> List[str]:
    """Searches the internet for information that does not exist in the database or for the lastest information."""
    
    tavily_search = TavilySearchResults(max_results=2)
    docs = tavily_search.invoke(query)

    formatted_docs = []
    for doc in docs:
        formatted_docs.append(
            Document(
                page_content=f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>',
                metadata={"source": "web search", "url": doc["url"]}
            )
        )
    
    if len(formatted_docs) > 0:
        return formatted_docs
    
    return [Document(page_content="관련 웹 정보를 찾을 수 없습니다.")]

# 도구 목록을 정의
tools = [
    search_menu,
    search_wine,
    search_web
]

# 기본 LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

# LLM에 도구 바인딩하여 추가
llm_with_tools = llm.bind_tools(tools)

# 메뉴 검색에 관련된 질문을 하는 경우 -> 메뉴 검색 도구를 호출
query = "대표 메뉴는 무엇인가요?"
ai_msg = llm_with_tools.invoke(query)

pprint(ai_msg)
print("-" * 100)

pprint(ai_msg.content)
print("-" * 100)

pprint(ai_msg.tool_calls)
print("-" * 100)


# 도구들의 목적과 관련 없는 질문을 하는 경우 -> 도구 호출 없이 그대로 답변을 생성
query = "안녕하세요"
ai_msg = llm_with_tools.invoke(query)

pprint(ai_msg)
print("-" * 100)

pprint(ai_msg.content)
print("-" * 100)

pprint(ai_msg.tool_calls)
print("-" * 100)

# 웹 검색 목적과 관련된 질문을 하는 경우 -> 웹 검색 도구 호출
query = "2024년 상반기 엔비디아 시가총액은 어떻게 변동했나요?"
ai_msg = llm_with_tools.invoke(query)

pprint(ai_msg)
print("-" * 100)

pprint(ai_msg.content)
print("-" * 100)

pprint(ai_msg.tool_calls)
print("-" * 100)
