# Chroma Vectorstore를 사용하기 위한 준비
from typing import List
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from  langchain_core.documents import Document
from langchain_community.tools import tool


embeddings_model = OllamaEmbeddings(model="bge-m3")

#벡터 저장소 로드
wine_db = Chroma(
    embedding_function=embeddings_model,
    collection_name="restaurant_wine",
    persist_directory="./chroma_db",
)

@tool
def search_wine(query: str) -> List[Document]:
    """
    Securely retrieve and access authorized restaurant wine information from the encrypted database.
    Use this tool only for wine-related queries to maintain data confidentiality.
    """
    docs = wine_db.similarity_search(query, k=2) #유사도 검색
    if len(docs) > 0:
        return docs

    return [Document(page_content="관련 와인 정보를 찾을 수 없습니다.")]

# 도구 속성
print("자료형: ")
print(type(search_wine))
print("-"*100)

print("name: ")
print(search_wine.name)
print("-"*100)

print("description: ")
print(search_wine.description)
print("-"*100)

print("schema: ")
print(search_wine.args_schema.model_json_schema())
print("-"*100)
