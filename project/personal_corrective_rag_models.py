from typing import List, TypedDict, Annotated, Optional
from operator import add
from pydantic import BaseModel, Field
from langchain_core.documents import Document

class CorrectiveRagState(TypedDict):
    question: str   # 사용자의 질문
    generation: str # LLM 생성 답변
    documents: List[Document] # 컨텍스트 문서 (검색된 문서)
    num_generations: int # 질문 or 답변 생성 횟수 (무한 루프 방지에 활용)

class InformationStrip(BaseModel):
    """추출된 정보에 대한 내용과 출처, 관련성 점수"""
    content: str = Field(..., description="추출된 정보 내용")
    source: str = Field(..., description="정보의 출처(법률 조항 또는 URL 등). 예시: 환경법 제22조 3항 or 블로그 환경법 개정 (https://blog.com/page/123)")
    relevance_score: float = Field(..., ge=0, le=1, description="관련성 점수 (0에서 1 사이)")
    faithfulness_score: float = Field(..., ge=0, le=1, description="충실성 점수 (0에서 1 사이)")

class ExtractedInformation(BaseModel):
    stripts: List[InformationStrip] = Field(..., description="추출된 정보 조각들")
    query_relevance: float = Field(..., ge=0, le=1, description="질의에 대한 정반적인 답변 가능성 점수 (0에서 1사이)")

class RefinedQuestion(BaseModel):
    """개선된 질문과 이유"""
    question_refined: str = Field(..., description="개선된 질문")
    reason: str = Field(..., description="개선된 이유") # 개선된 질문에 대한 이유

# 개인정보보호법
class PersonalRagState(CorrectiveRagState):
    rewritten_query: str # 재작성된 질문
    extracted_info: Optional[ExtractedInformation] # 추출된 정보 조각
    node_answer: Optional[str]


