from typing import List, Literal
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

# 문서 관련성 평가 결과를 위한 데이터 모델 정의
class MultiGradeDocuments(BaseModel):
    """Three-class score for relevance check on retrieved documents."""

    relevance_score: Literal["correct", "incorrect", "ambiguous"] = Field(
        description="Document relevance to the question: 'correct', 'incorrect', or 'ambiguous'"
    )

# LLM 모델 초기화 및 구조화된 출력 설정
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_grader = llm.with_structured_output(MultiGradeDocuments)

# 문서 관련성 평가를 위한 시스템 프롬프트 정의
system_prompt = """
You are an expert evaluator tasked with assessing the relevance of retrieved documents to a user's question. Your role is crucial in enhancing the quality of information retrieval systems.

[평가 기준]
1. 키워드 관련성 : 문서가 질문의 주요 단어나 유사어를 포함하는지 확인
2. 의미적 관련성 : 문서의 전반적인 주제가 질문의 의도와 일치하는지 평가
3. 부분 관련성 : 질문의 일부를 다루거나 맥락 정보를 제공하는 문서도 고려
4. 답변 가능성: 직접적인 답이 아니더라도 답변 형성에 도움될 정보 포함 여부 평가

[점수 체계]
- 'Correct;: 문서가 명확히 관련 있고, 질문에 답하는 데 필요한 정보를 포함함.
- 'Incorrect' : 문서가 명확히 무관하거나, 질문에 도움이 되지 않는 정보를 포함함.
- 'Ambiguous' : 문서의 관련성이 불분명하거나, 일부 관련 정보는 있지만 유용성이 확실하지 않음, 혹은 질문과 약간만 관련 있음.

[주의사항]
- 단순 단어 매칭이 아닌 질문의 전체 맥락을 고려하세요
- 완벽한 답변이 아니어도 유용한 정보가 있다면 관련 있다고 판단하세요

Your evaluation plays a critical role in improving the overall performance of the information retrieval system. Strive for balanced and thoughtful assessments.
"""

# 채점 프롬프트 템플릿 생성
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Document: \n\n {document} \n\n Question: {question}"),
])

# Retrieval Grade 파이프라인 구성
retrieval_grader_multi = grade_prompt | structured_llm_grader

# 관련성 평가 실행
question = "비건 메뉴가 있나요?"
retrieved_docs = search_menu.invoke(question)
print(f"검색된 문서 수: {len(retrieved_docs)}")

for test_chunk in retrieved_docs:
    print("문서:", test_chunk.page_content)
    print("-"*50)

    relevance = retrieval_grader_multi.invoke({"question": question, "document": test_chunk.page_content})
    print(f"문서 관련성: {relevance.relevance_score}")
    print("="*50)
