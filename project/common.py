from typing import List, TypedDict
from pprint import pprint
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker

#문서 임베딩 모델
embedings_model = OllamaEmbeddings(model="bge-m3")

# Re-rank 모델
rerank_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
cross_reranker = CrossEncoderReranker(model=rerank_model, top_n=2)

# 개인정보보호법 검색
personal_db = Chroma(
    embedding_function=embedings_model,
    collection_name="personal_law",
    persist_directory="./chroma_db"
)

personal_db_retriever = ContextualCompressionRetriever(
    base_compressor=cross_reranker,
    base_retriever=personal_db.as_retriever(search_kwargs={"k": 5}),
)

@tool
def personal_law_search(query: str) -> List[Document]:
    """개인정보보호법 법률 조항을 검색합니다."""
    docs = personal_db_retriever.invoke(query)

    if len(docs) > 0:
        return docs

    return [Document(page_content="관련 정보를 찾을 수 없습니다.")]

# 근로기준법 검색
labor_db = Chroma(
    embedding_function=embedings_model,
    collection_name="labor_law",
    persist_directory="./chroma_db"
)

labor_db_retriever = ContextualCompressionRetriever(
    base_compressor=cross_reranker,
    base_retriever=labor_db.as_retriever(search_kwargs={"k": 5}),
)

@tool
def labor_law_search(query: str) -> List[Document]:
    """근로기준법 법률 조항을 검색합니다."""
    docs = labor_db_retriever.invoke(query)

    if len(docs) > 0:
        return docs

    return [Document(page_content="관련 정보를 찾을 수 없습니다.")]

#주택임대차보호법 검색
housing_db = Chroma(
    embedding_function=embedings_model,
    collection_name="housing_law",
    persist_directory="./chroma_db"
)

housing_db_retriever = ContextualCompressionRetriever(
    base_compressor=cross_reranker,
    base_retriever=housing_db.as_retriever(search_kwargs={"k": 5}),
)

@tool
def housing_law_search(query: str) -> List[Document]:
    """주택임대차보호법 법률 조항을 검색합니다."""
    docs = housing_db_retriever.invoke(query)

    if len(docs) > 0:
        return docs

    return [Document(page_content="관련 정보를 찾을 수 없습니다.")]

# 웹 검색
web_retriever = ContextualCompressionRetriever(
    base_compressor=cross_reranker,
    base_retriever=TavilySearchAPIRetriever(k=10)
)

@tool
def web_search(query: str) -> List[str]:
    """데이터베이스에 없는 정보 또는 최신 정보를 웹에서 검색합니다."""
    docs = web_retriever.invoke(query)

    formatted_docs = []
    for doc in docs:
        formatted_docs.append(
            Document(
                page_content=f'<Document href="{doc.metadata["source"]}"/>\n{doc.page_content}\n</Document>',
                metadata={"source": "web search", "url": doc.metadata["source"]}
            )
        )

    if len(formatted_docs) > 0:
        return formatted_docs

    return [Document(page_content="관련 정보를 찾을 수 없습니다.")]

tools = [
    personal_law_search,
    labor_law_search,
    housing_law_search,
    web_search
]

# 기본 LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

# LLM에 도구 바인딩
llm_with_tools = llm.bind_tools(tools)

# 문서 검색 테스트 함수
def test(query: str):
    ai_msg = llm_with_tools.invoke(query)

    pprint(ai_msg)
    print("-" * 100)

    pprint(ai_msg.content)
    print("-"*100)

    pprint(ai_msg.tool_calls)
    print("-"*100)


# test("연차휴가 부여 기준에 대해서 설명해주세요.")
# test("안녕하세요?")
# test("연차휴가 부여 기준에 대해서 설명해주세요. 2023년 연차휴가 사용 비율은 어느 정도인가요?") #벡터 검색과 웹 검색이 모두 필요한 경우
# test("전월세 직거래 시에 유의사항은 무엇인가요?") # 벡터 검색과 웹 검색이 모두 필요한 경우

# 공통으로 사용할 모델
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
