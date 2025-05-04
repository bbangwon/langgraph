from typing import List
from personal_models import PersonalRagState
from personal_graph import personal_law_agent
from labor_models import LaborRagState
from labor_graph import labor_law_agent
from housing_models import HousingRagState
from housing_graph import housing_law_agent
from search_models import SearchRagState
from search_graph import search_law_agent
from route_models import ResearchAgentState
from route_llm import question_tool_router
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from common import llm


#질문 라우팅 노드
def analyze_question_tool_search(state: ResearchAgentState):
    question = state["question"]
    result = question_tool_router.invoke({"question": question})
    datasources = [tool.tool for tool in result.tools]
    return {"datasources": datasources}

def route_datasources_tool_search(state: ResearchAgentState) -> List[str]:
    datasources = set(state['datasources']) # 중복 제거
    valid_sources = {"search_personal", "search_labor", "search_housing", "search_web"}

    if datasources.issubset(valid_sources): # 모든 데이터 소스가 유효한 경우
        return list(datasources)

    return list(valid_sources)

# 노드 정의
def personal_rag_node(state: PersonalRagState, input=ResearchAgentState) -> ResearchAgentState:
    print("--- 개인정보보호법 전문가 에이전트 시작 ---")
    question = state["question"]
    answer = personal_law_agent.invoke({"question": question})
    return {"answers": [answer["node_answer"]]}

def labor_rag_node(state: LaborRagState, input=ResearchAgentState) -> ResearchAgentState:
    print("--- 근로기준법 전문가 에이전트 시작 ---")
    question = state["question"]
    answer = labor_law_agent.invoke({"question": question})
    return {"answers": [answer["node_answer"]]}

def housing_rag_node(state: HousingRagState, input=ResearchAgentState) -> ResearchAgentState:
    print("--- 주택임대차보호법 전문가 에이전트 시작 ---")
    question = state["question"]
    answer = housing_law_agent.invoke({"question": question})
    return {"answers": [answer["node_answer"]]}

def web_rag_node(state: SearchRagState, input=ResearchAgentState) -> ResearchAgentState:
    print("--- 인터넷 검색 전문가 에이전트 시작 ---")
    question = state["question"]
    answer = search_law_agent.invoke({"question": question})
    return {"answers": [answer["node_answer"]]}

def answer_final(state: ResearchAgentState) -> ResearchAgentState:
    """
    Generate answer using the retrieved_documents
    """
    print("---최종 답변---")

    # RAG 프롬프트 정의
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """YOu are an assistant answering questions based on provided documents. Follow these guidelines:

        1. Use only information from the given documents.
        2. If the document lacks relevant info, say "제공된 정보로는 충분한 답변을 할 수 없습니다."
        3. Cite the source of information for each sentence in your answer. Use the following format:
            - For legal articles: "법률명 제X조 Y항"
            - For web sources: "출처 제목 (URL)"
        4. Don't speculate or add information not in the documents.
        5. Keep answers concise and clear.
        6. Omit irrelevant information.
        7. If multiple sources provide the same information, cite all relevant sources.
        8. If information comes from multiple sources, combine them coherently while citing all sources.

        Example of citation usage:
        "부동산 거래 시 계약서에 거래 금액을 명시해야 합니다 (부동산 거래신고 등에 관한 법률 제3조 1항). 또한, 계약 체결일로부터 30일 이내에 신고해야 합니다 (부동산 거래 신고 안내 블로그, https://example.com/realestate)."
        """),
        ("human", "Answer the following question using these documents:\n\n{documents}\n\n[Question]\n{question}"),
    ])

    question = state["question"]
    documents = state.get("answers", [])
    if not isinstance(documents, list):
        documents = [documents]

    # 문서 내용을 문자열로 결합
    document_text = "\n\n".join(documents)
    # RAG generation
    rag_chain = rag_prompt | llm | StrOutputParser()
    generation = rag_chain.invoke({"documents": document_text, "question": question})

    return {"final_answer": generation, "question":question}



def llm_fallback(state: ResearchAgentState) -> ResearchAgentState:
    """
    Generate answer using the LLM without context
    """
    print("---Fallback 답변---")

    # LLM Fallback 프롬프트 정의
    fallback_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AI assistant helping with various topics. Follow these guidelines:

        1. Provide accurate and helpful information to the best of your ability.
        2. Express uncertainty when unsure; avoid speculation.
        3. Keep answers concise yet informative.
        4. Respond ethically and constructively.
        5. Mention reliable general sources when applicable."""),
        ("human", "{question}"),
    ])

    question = state["question"]
    llm_chain = fallback_prompt | llm | StrOutputParser()
    answer = llm_chain.invoke({"question": question})
    return {"final_answer": answer, "question":question}
