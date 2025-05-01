from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

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

# 지식 정제를 위한 클래스
class RefinedKnowledge(BaseModel):
    """
    Represensts a refined piece of knowledge extracted form a document.
    """

    knowledge_strip: str = Field(description="A refiened piece of knowledge extracted from a document.")
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

# LLM 모델 초기화 및 구조화된 출력 설정
llm = ChatOpenAI(model="gpt-4o", temperature=0) # 2가지 복합적인 작업때문에 gpt-4o 모델 사용
structured_llm_refiner = llm.with_structured_output(RefinedKnowledge)

# 지식 정제를 위한 프롬프트
system_prompt = """
    You are an expert in knowledge refinement. Your task is to extract key information from the given document related to the provided question and evaluate its relevance.

    [Instructions]
    1. 질문과 문서를 주의 깊게 읽습니다.
    2. 질문에 답하는 데 관련된 문서 내의 주요 정보를 식별합니다.
    3. 각 주요 정보에 대해:
        a. 간결하게 추출하고 요약합니다 (정보당 1-2문장을 목표로 합니다).
        b. 질문에 대한 관련성을 'yes' (관련 있음) 또는 'no' (관련 없음)로 평가합니다.
    4. 각 정보를 다음 형식으로 새 줄에 제시합니다:
        [추출된 정보] (yes/no)

    [Example Output]
    AI 시스템은 학습 데이터에 존재하는 편향을 나타낼 수 있습니다. (yes)
    의사 결정에 AI를 사용하는 것은 프라이버시 문제를 제기합니다. (yes)
    기계 학습 모델은 상당한 컴퓨팅 자원을 필요로 합니다. (no)

    [Note]
    Focus on extracting factual and objective information. Avoid personal opinions or speculations. Aim to provide 3-5 key pieces of information, but you may include more if the document is particularly rich in relevant content.
    """

# 지식정제를 위한 프롬프트 템플릿 생성
refine_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "[Document]\n{document}\n\n[User question]\n{question}"),
])

# Knowledge Refiner 파이프라인 구성
knowledge_refiner = refine_prompt | structured_llm_refiner

# 지식 정제 실행
question = "대표 메뉴는 무엇인가요?"
retrieved_docs = search_menu.invoke(question)
print(f"검색된 문서 수: {len(retrieved_docs)}")

for test_chunk in retrieved_docs:
    print("문서: ", test_chunk.page_content)

    refiend_knowledge = knowledge_refiner.invoke({"question": question, "document": test_chunk})
    print(f"정제된 지식: {refiend_knowledge.knowledge_strip}")
    print(f"정제된 지식 평가: {refiend_knowledge.binary_score}")
    print("="*50)

