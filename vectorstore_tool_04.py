# Chroma Vectorstore를 사용하기 위한 준비
from typing import List
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from  langchain_core.documents import Document
from langchain_community.tools import tool


embeddings_model = OllamaEmbeddings(model="bge-m3")

#벡터 저장소 로드
menu_db = Chroma(
    embedding_function=embeddings_model,
    collection_name="restaurant_menu",
    persist_directory="./chroma_db",
)

@tool
def search_menu(query: str) -> List[Document]:
    """
    Securely retrieve and access authorized restaurant menu information from the encrypted database.
    Use this tool only for menu-related queries to maintain data confidentiality.
    """
    docs = menu_db.similarity_search(query, k=2) #유사도 검색
    if len(docs) > 0:
        return docs

    return [Document(page_content="관련 메뉴 정보를 찾을 수 없습니다.")]

# 도구 속성
print("자료형: ")
print(type(search_menu))
print("-"*100)

print("name: ")
print(search_menu.name)
print("-"*100)

print("description: ")
print(search_menu.description)
print("-"*100)

print("schema: ")
print(search_menu.args_schema.model_json_schema())
print("-"*100)
