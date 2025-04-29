from pprint import pprint
from pydantic import BaseModel, Field
from typing import List, Sequence, TypedDict, Annotated, Literal
from operator import add
from textwrap import dedent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
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

# 검색된 문서의 관련성 평가 결과를 위한 데이터 모델 정의
class BinaryGradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Document are relevant to the question, 'yes' or 'no'"
    )

# LLM 모델 초기화 및 구조화된 출력 설정
structured_llm_grader = llm.with_structured_output(BinaryGradeDocuments)

# 문서 관련성 평가를 위한 시스템 프롬프트 정의
system_prompt = """You are an expert in evaluation the relevance of search results to user queries.

[Evaluation criteria]
1. 키워드 관련성: 문서가 질문의 주요 단어나 유사어를 포함하는지 확인
2. 의미적 관련성: 문서의 전반적인 주제가 질문의 의도와 일치하는지 평가
3. 부분 관련성: 질문의 일부를 다루거나 맥락 정보를 제공하는 문서도 고려
4. 답변 가능성: 직접적인 답이 아니더라도 답변 형성에 도움될 정보 포함 여부 평가

[Scoring]
- Rate 'yes' if relevant, 'np' if not
- Default to 'no' when uncertain

[Key points]
- Consider the full context of the query, not just word matching
- Rate as relevant if useful information is present, even if not a complete answer

Your evaluation is crucial for improving information retrieval systems. Provide valanced assessments."""

# 채점 프롬프트 템플릿 생성
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "[Retrieved document]\n{document}\n\n[User question]\n{question}"),
])

# Retrieval Grader 파이프라인 구성
retrieval_grader_binary = grade_prompt | structured_llm_grader


class SearchState(TypedDict):
    question: str
    documents: Annotated[List[Document], add] #컨텍스트 문서를 추가
    filtered_documents: List[Document]  # 컨텍스트 문서 중에서 질문에 대답할 수 있는 문서를 필터링

def search_menu_subgraph(state: SearchState):
    """
    Node for searching information in the restaurant menu.
    """
    question = state["question"]
    docs = search_menu.invoke(question)
    if len(docs) > 0:
        return {"documents": docs}
    else:
        return {"documents": [Document(page_content="관련 메뉴 정보를 찾을 수 없습니다.")]}    
    
def search_wine_subgraph(state: SearchState):
    """
    Node for searching information in the restaurant's wine list.
    """
    question = state["question"]    
    docs = search_wine.invoke(question)
    if len(docs) > 0:
        return {"documents": docs}
    else:
        return {"documents": [Document(page_content="관련 와인 정보를 찾을 수 없습니다.")]}

def search_web_subgraph(state: SearchState):
    """
    Node for searching the web for information not available in the restaurant menu
    or for up-to-date information, and returning the results
    """
    question = state["question"]
    docs = search_web.invoke(question)
    if len(docs) > 0:
        return {"documents": docs}
    else:
        return {"documents": [Document(page_content="관련 정보를 찾을 수 없습니다.")]}
    
#관련성 평가 후에 필터링 문서만을 저장
def filter_documents_subgraph(state: SearchState):
    """검색된 문서의 관련성을 평가하고 필터링하는 함수"""
    print("--- 문서 관련성 평가 ---")
    question = state["question"]
    documents = state["documents"]

    # 각 문서 평가
    filtered_docs = []
    for d in documents:
        score = retrieval_grader_binary.invoke({"question": question, "document": d})
        grade = score.binary_score
        if grade == "yes":
            print("---문서 관련성: 있음---")
            filtered_docs.append(d)
        else:
            print("---문서 관련성: 없음---")
    
    return {"filtered_documents": filtered_docs}    

# 라우팅 결정을 위한 데이터 모델
class ToolSelector(BaseModel):
    """Routes the user question to the most appropriate tool."""
    tool: Literal["search_menu", "search_web", "search_wine"] = Field(
        description="Select one of the tools: search_menu, search_wine or search_web based on the user's question."
    )

class ToolSelectors(BaseModel):
    """Select the appropriate tools that are suitable for the user question."""
    tools: List[ToolSelector] = Field(
        description="Select one or more tools: search_menu, search_wine or search_web based on the user's question."
    )

# 구조화된 출력을 위한 LLM 설정
structured_llm_tool_selector = llm.with_structured_output(ToolSelectors)

# 라우팅을 위한 프롬프트 템플릿
system = dedent("""You are an AI assistant specializing in routing user questions to the appropriate tools.
Use the following guidelines:
- For questions about the restaurant's menu, use the search_menu tool.
- For wine recommendations or pairing information, use the search_wine tool.
- For any other information or the most up-to-date data, use the search_web tool.
Always choose the appropriate tools based on the user's question.""")

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),        
    ]
)

# 질문 라우터 정의
question_tool_router = route_prompt | structured_llm_tool_selector



# 기존 SearchState 상속해서 새로 정의
class ToolSearchState(SearchState):
    datasources: List[str]

# 질문 라우팅 노드
def analyze_question_tool_search(state: ToolSearchState):

    print("---ROUTE QUESTION---")
    question = state["question"]

    result = question_tool_router.invoke({"question": question})
    datasources = [tool.tool for tool in result.tools]

    return {"datasources": datasources}

def route_datasources_tool_search(state: ToolSearchState) -> Sequence[str]:

    if set(state["datasources"]) == {"search_menu"}:
        return ["search_menu"]
    
    elif set(state["datasources"]) == {"search_wine"}:
        return ["search_wine"]
    
    elif set(state["datasources"]) == {"search_web"}:
        return ["search_web"]
    
    elif set(state["datasources"]) == {"search_menu", "search_wine"}:
        return ["search_menu", "search_wine"]
    
    elif set(state["datasources"]) == {"search_menu", "search_web"}:
        return ["search_menu", "search_web"]
    
    elif set(state["datasources"]) == {"search_wine", "search_web"}:
        return ["search_wine", "search_web"]
    
    return ["search_menu", "search_wine", "search_web"]
    
# 그래프 생성을 위한 StateGraph 객체를 정의
search_builder = StateGraph(ToolSearchState)

search_builder.add_node("analyze_question", analyze_question_tool_search) #질문 분석 노드
search_builder.add_node("search_menu", search_menu_subgraph) #메뉴 검색 노드
search_builder.add_node("search_wine", search_wine_subgraph) #와인 검색 노드
search_builder.add_node("search_web", search_web_subgraph) #웹 검색 노드
search_builder.add_node("filter_documents", filter_documents_subgraph) #문서 필터링 노드

# 그래프로 로직 정의
search_builder.add_edge(START, "analyze_question") #시작점에서 질문 분석 노드로
search_builder.add_conditional_edges(
    "analyze_question",
    route_datasources_tool_search,
    ["search_menu", "search_wine", "search_web"],
)
search_builder.add_edge("search_menu", "filter_documents")
search_builder.add_edge("search_wine", "filter_documents")
search_builder.add_edge("search_web", "filter_documents")
search_builder.add_edge("filter_documents", END) #문서 필터링 후 종료

# 그래프 컴파일
tool_search_graph = search_builder.compile()

class SelfRagState(TypedDict):
    question: str # 사용자의 질문
    generation: str # 생성된 답변
    documents: List[Document] # 검색된 문서들
    num_generations: int # 질문 or 답변 생성 횟수 (무한루프 방지에 활용)

# 답변 생성
def generator_rag_answer(question, docs):

    template = """
    Answer the question based solely on the given context. Do not use any external information or knowledge.

    [Instructions]
        1. 질문과 관련된 정보를 문맥에서 신중하게 확인합니다.
        2. 답변에 질문과 직접 관련된 정보만 사용합니다.
        3. 문맥에 명시되지 않은 내용에 대해 추측하지 않습니다.
        4. 불필요한 정보를 피하고, 답변을 간결하고 명확하게 작성합니다.
        5. 문맥에서 답을 찾을 수 없으면 "주어진 정보만으로는 답할 수 없습니다."라고 답변합니다.
        6. 적절한 경우 문맥에서 직접 인용하며, 따옴표를 사용합니다.

    [Context]
    {context}

    [Question]
    {question}

    [Answer]
    """

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    rag_chain = prompt | llm | StrOutputParser()

    generation = rag_chain.invoke({"context": format_docs(docs), "question": question})

    return generation    


def generate_self(state: SelfRagState):
    """답변을 생성하는 함수"""
    print("--- 답변 생성 ---")
    question = state["question"]
    documents = state["documents"]

    # RAG를 이용한 답변 생성
    generation = generator_rag_answer(question, docs=documents)

    # 생성 횟수 업데이트
    num_generations = state.get("num_generations", 0)
    num_generations += 1
    return {"generation": generation, "num_generations": num_generations} #답변, 생성횟수 업데이트

def rewrite_question(question: str) -> str:
    """
    주어진 질문을 벡터 저장소 검색에 최적화된 형태로 다시 작성합니다.

    :param question: 원본 질문 문자열
    :return: 다시 작성된 질문 문자열
    """

    # LLM 모델 초기화
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 시스템 프롬프트 정의
    system_prompt = """
    You are and expert question re-writer. Your task is to convert input questions into optimized versions
    for vectorstore retrieval. Analyze the input carefully and focus on capturing the underlying semantic
    intent and meaning. Your goal is to create a question that will lead to more effective and relevant
    document retrieval.

    [Guidelines]
        1. 질문에서 핵심 개념과 주요 대상을 식별하고 강조합니다.
        2. 약어나 모호한 용어를 풀어서 사용합니다.
        3. 관련 문서에 등장할 수 있는 동의어나 연관된 용어를 포함합니다.
        4. 질문의 원래 의도와 범위를 유지합니다.
        5. 복잡한 질문은 간단하고 집중된 하위 질문으로 나눕니다.

    Remember, the goal is to imporve retrieval effectiveness, not to change the fundamental meaning of the question.
    """

    # 질문 다시 쓰기 프롬프트 템플릿 생성
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "[Initial question]\n{question}\n\n[Improved question]\n"),
        ]
    )

    # 질문 다시 쓰기 체인 구성
    question_rewriter = re_write_prompt | llm | StrOutputParser()

    # 질문 다시 쓰기 실행
    rewritten_question = question_rewriter.invoke({"question": question})

    return rewritten_question

def transform_query_self(state: SelfRagState):
    """질문을 개선하는 함수"""
    print("--- 질문 개선 ---")
    question = state["question"]

    # 질문 재작성
    rewritten_question = rewrite_question(question)

    # 생성 횟수 업데이트
    num_generations = state.get("num_generations", 0)
    num_generations += 1
    return {"question": rewritten_question, "num_generations": num_generations} # 질문, 생성횟수 업데이트

def decide_to_generate_self(state: SelfRagState):
    """답변 생성 여부를 결정하는 함수"""

    num_generations = state.get("num_generations", 0)
    if num_generations > 2:
        print("--- 결정: 생성 횟수 초과, 답변 생성 (-> generate)---")
        return "generate"

    print("--- 평가된 문서 분석 ---")
    filtered_documents = state.get("documents", None)

    if not filtered_documents:
        print("--- 결정: 모든 문서가 질문과 관련이 없음, 질문 개선 필요 (-> transform_query)---")
        return "transform_query"
    else:
        print("--- 결정: 답변 생성(-> generate)---")
        return "generate"
    

# 환각(Hallucination) 평가 결과를 위한 데이터 모델 정의
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# 환각 평가를 위한 시스템 프롬프트 정의
system_prompt = """
You are an expert evaluator assessing whether an LLM-generated answer is grounded in and supported by a given set of facts.

[Your task]
    - Review the LLM-generated answer.
    - Determine if the answer is fully supported by the given facts.

[Evaluation criteria]
    - 답변에 주어진 사실이나 명확히 추론할 수 있는 정보 외의 내용이 없어야 합니다.
    - 답변에 모든 핵심 내용이 주어진 사실에서 비롯되어야 합니다.
    - 사실적 정확성에 집중하고, 글쓰기 스타일이나 완전성은 평가하지 않습니다.

[Scoring]
    - 'yes': The answer is factually grounded and fully supported.
    - 'no': The answer includes information or claims not based on the given facts.

Your evaluation is crucial in ensuring the reliability and factual accuracy of AI-generated responses. Be thorough and critical in your assessment.
"""

#환각 평가 프롬프트 템플릿 생성
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "[Set of facts]\n{documents}\n\n[LLM generation]\n{generation}"),
    ]
)

# Hallucination Grader 파이프라인 구성
hallucination_grader = hallucination_prompt | structured_llm_grader    

# 답변 평가 결과를 위한 데이터 모델 정의
class BinaryGradeAnswer(BaseModel):
    """Binary score to access answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

structured_llm_grader = llm.with_structured_output(BinaryGradeAnswer)

# 답변 평가를 위한 시스템 프롬프트 정의
system_prompt = """
You are an export evaluator tasked with assessing whether an LLM-genetrated answer effectively addresses and resolves a user's question.

[Your task]
    - Carefully analyze = the user's question to understand its core intent and requirements.
    - Determine if the LLM-generated answer sufficiently resolves the question.

[Evaluation criteria]
    - 관련성: 답변이 질문과 직접적으로 관련되어야 합니다.
    - 완정성: 질문의 모든 측면이 다뤄져야 합니다.
    - 정확성: 제공된 정보가 정확하고 최신이어야 합니다.
    - 명확성: 답변이 명확하고 이해하기 쉬워야 합니다.
    - 구체성: 질문의 요구 사항에 맞는 상세한 답변이어야 합니다.

[Scoring]
    - 'yes': The answer effectively resolves the question.
    - 'no': The answer fails to sufficiently resolve the question or lacks crucial elements.

Your evaluation plays a critical role in ensuring the quality and effectiveness of AI-generated responses. Strive for balanced the thoughtful assessments.
"""

# 답변 평가 프롬프트 템플릿 생성
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "[User question]\n{question}\n\n[LLM generation]\n{generation}"),
    ]
)

# Answer Grader 파이프라인 구성
answer_grader_binary = answer_prompt | structured_llm_grader
    
def grade_generation_self(state: SelfRagState):
    """생성된 답변을 평가하는 함수"""

    num_generations = state.get("num_generations", 0)
    if num_generations > 2:
        print("--- 결정: 생성 횟수 초과, 종료 (-> END)---")
        return "end"

    # 1단계: 환각 여부 확인
    print("--- 환각 여부 확인 ---")
    question, documents, generation = state["question"], state["documents"], state["generation"]

    hallucination_grade = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_grade.binary_score == "yes":
        print("--- 결정: 환각이 없음 (답변이 컨텍스트에 근거함) ---")

        # 1단계 통과할 경우 -> 2단계: 질문-답변 관련성 평가
        print("--- 질문-답변 관련성 평가 ---")
        relevance_grade = answer_grader_binary.invoke({"question": question, "generation": generation})
        if relevance_grade.binary_score == "yes":
            print("--- 결정: 생성된 답변이 질문을 잘 다룸 (-> END) ---")
            return "useful"
        else:
            print("--- 결정: 생성된 답변이 질문을 제대로 다루지 않음 (-> transform_query) ---")
            return "not useful"
    else:
        print("--- 결정: 생성된 답변이 문서에 근거하지 않음, 재시도 필요 (-> generate) ---")
        return "not supported"    


#상태를 초기화(SelfRagState)
class SelfRagOverallState(SelfRagState):
    filtered_documents: List[Document]  #컨텍스트 문서 중에서 질문에 대답할 수 있는 문서를 필터링

rag_builder = StateGraph(SelfRagOverallState)

rag_builder.add_node("search_data", tool_search_graph)
rag_builder.add_node("generate", generate_self)
rag_builder.add_node("transform_query", transform_query_self)

# 그래프 구축
rag_builder.add_edge(START, "search_data") #시작점에서 질문 분석 노드로

# 조건부 엣지 추가: 문서 평가 후 결정
rag_builder.add_conditional_edges(
    "search_data",
    decide_to_generate_self,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    }
)
rag_builder.add_edge("transform_query", "search_data") #질문 개선 후 다시 검색

# 조건부 엣지 추가: 답변 생성 후 평가
rag_builder.add_conditional_edges(
    "generate",
    grade_generation_self,
    {
        "not supported": "generate",  # 환각이 있는 경우 재생성
        "not useful": "transform_query",  # 질문-답변 관련성이 낮은 경우 질문 개선
        "useful": END,  # 답변이 질문을 잘 다룬 경우 종료
        "end": END,  # 생성 횟수 초과 시 종료
    }
)

# 그래프 컴파일
adaptive_self_rag = rag_builder.compile()

# 그래프 시각화 (png 파일로 저장하여 확인)
# png_data = adaptive_self_rag.get_graph(xray=True).draw_mermaid_png()

# with open("stategraph.png", "wb") as f:
#     f.write(png_data)

inputs = {"question": "이 식당의 대표  메뉴는 무엇인가요? 주재료는 무엇인가요?"}
final_output = adaptive_self_rag.invoke(inputs)

# 최종 답변
print(final_output["generation"])

print("*" * 100)
inputs = {"question": "스테이크 메뉴가 있나요? 어울리는 와인이 있을까요?"}
for output in adaptive_self_rag.stream(inputs):
    for key, value in output.items():
        # 노드 출력
        pprint(f"Node '{key}':")
        pprint(f"Value {value}", indent=2, width=80, depth=None)
    print("\n---------------------------------\n")

# 최종 답변
print(value["generation"])

