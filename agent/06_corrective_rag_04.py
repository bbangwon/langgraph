from typing import List, Tuple, Literal
from pydantic import BaseModel, Field
from pprint import pprint
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langgraph.graph import MessagesState, StateGraph, START, END

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

# 답변 생성
def generator_rag_answer(question, docs):

    template = """
    Answer the question based solely on the given context. Do not use any external information or knowledge.

    [Instructions]
        1. 질문과 관련된 정보를 문맥에서 신중하게 확인합니다.
        2. 답변에 질문과 직접 관련된 정보만 사용합니다.
        3. 불필요한 정보를 피하고, 답변을 간결하고 명확하게 작성합니다.
        4. 문맥에서 답을 찾을 수 없으면 "주어진 정보만으로는 답할 수 없습니다."라고 답변합니다.
        5. 적절한 경우 문맥에서 직접 인용하며, 따옴표를 사용합니다.

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


class CorrectiveRagState(MessagesState):
    """
    메시지 기반 그래프의 상태를 나타내는 클래스

    Attributes:
        messages: 대화 히스토리 (기본 제공, 설정 불필요)
        question: 사용자의 질문
        generation: AI 모델의 답변
        retrieved_documents: 검색된 문서 리스트 (문서, 점수)
        knowledge_strips: 지식 보강한 결과 리스트 (문서, 점수)
        num_generations: 생성 횟수(무한 루프 방지에 활용)
    """
    question: str
    generation: str
    retrieved_documents: List[Tuple[Document, str]]
    knowledge_strips: List[Tuple[Document, str]]
    num_generations: int

def retrieve_crag(state: CorrectiveRagState):
    """문서를 검색하는 함수"""
    print("--- 문서 검색 ---")
    question = state["question"]

    # 문서 검색 로직
    retrieved_documents = search_menu.invoke(question)
    retrieved_documents = [(doc, "ambiguous") for doc in retrieved_documents]

    return {"retrieved_documents": retrieved_documents}

def grade_documents_crag(state: CorrectiveRagState):
    """검색된 문서의 관련성을 평가하는 함수"""
    print("--- 문서 관련성 평가 ---")
    question = state["question"]
    retrieved_documents = state.get("retrieved_documents", [])
    knowledge_strips = state.get("knowledge_strips", [])

    scored_docs = []
    for doc, _ in retrieved_documents:
        score = retrieval_grader_multi.invoke({"question": question, "document": doc.page_content})
        grade = score.relevance_score.lower()
        if grade == "correct":
            print("---문서 관련성: 있음---")
            scored_docs.append((doc, "correct"))
            knowledge_strips.append((doc, "correct"))
        elif grade == "incorrect":
            print("---문서 관련성: 없음---")
            scored_docs.append((doc, "incorrect"))
        else:
            print("---문서 관련성: 모호함---")
            scored_docs.append((doc, "ambiguous"))

    return {"retrieved_documents": scored_docs, "knowledge_strips": knowledge_strips}

def refine_knowledge_crag(state: CorrectiveRagState):
    """지식을 정제하는 함수"""
    print("--- 지식 정제 ---")
    question = state["question"]
    knowledge_strips = state.get("knowledge_strips", [])

    refined_docs = []
    for doc, score in knowledge_strips:
        if score == "incorrect":
            continue

        refined_knowledge = knowledge_refiner.invoke({"question": question, "document": doc.page_content})
        knowledge = refined_knowledge.knowledge_strip
        grade = refined_knowledge.binary_score
        if grade == "yes":
            print("---정제된 지식: 추가---")
            refined_docs.append((Document(page_content=knowledge), "correct"))

    return {"knowledge_strips": refined_docs}

def web_search_crag(state: CorrectiveRagState):
    """웹 검색을 수행하는 함수"""
    print("--- 웹 검색 ---")
    question = state["question"]

    search_results = search_web.invoke(question)

    scored_docs = []
    for result in search_results:
        score = retrieval_grader_multi.invoke({"question": question, "document": result})
        grade = score.relevance_score.lower()
        if grade == "correct":
            print("---웹 검색 문서 관련성: 있음---")
            scored_docs.append((result, "correct"))
        else:
            print("---웹 검색 문서 관련성: 없음---")

    return {"knowledge_strips": scored_docs}

def generate_crag(state: CorrectiveRagState):
    """답변을 생성하는 함수"""
    print("--- 답변 생성 ---")
    question = state["question"]
    knowledge_strips = state.get("knowledge_strips", [])

    doc_texts = [doc for doc, _ in knowledge_strips]

    generation = generator_rag_answer(question, docs=doc_texts)

    return {
        "generation": generation,
        "messages" : [AIMessage(content=generation)]
    }

def transform_query_crag(state: CorrectiveRagState):
    """질문을 개선하는 함수"""
    print("--- 질문 개선---")
    question = state["question"]
    num_generations = state.get("num_generations", 0)
    num_generations += 1

    rewritten_question = rewrite_question(question)

    return {
        "question": rewritten_question,
        "num_generations": num_generations,
        "messages": [HumanMessage(content=rewritten_question)]
    }


def decide_to_generate_crag(state: CorrectiveRagState):
    """답변 생성 여부를 결정하는 함수"""
    print("--- 평가된 문서 분석 --- ")
    knowledge_strips = state.get("knowledge_strips", [])

    if not knowledge_strips:
        print("--- 결정: 모든 문서가 질문과 관련이 없음, 질문 개선 필요 (->transform_query)---")
        return "transform_query"
    else:
        print("--- 결정: 답변 생성 (-> generate)---")
        return "generate"

builder = StateGraph(CorrectiveRagState)

# 노드 정의
builder.add_node("retrieve", retrieve_crag) #문서 검색
builder.add_node("grade_documents", grade_documents_crag) #문서 관련성 평가
builder.add_node("refine_knowledge", refine_knowledge_crag) #지식 정제
builder.add_node("web_search", web_search_crag) #웹 검색
builder.add_node("generate", generate_crag) #답변 생성
builder.add_node("transform_query", transform_query_crag) #질문 개선

# 경로 정의
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "grade_documents")
builder.add_edge("grade_documents", "refine_knowledge")

# 조건부 엣시 추가: 문서 평가 후 결정
builder.add_conditional_edges(
    "refine_knowledge",
    decide_to_generate_crag,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    }
)

# 추가 경로
builder.add_edge("transform_query", "web_search")
builder.add_edge("web_search", "generate")
builder.add_edge("generate", END)

# 그래프 컴파일
corrective_rag = builder.compile()

# 그래프 시각화 (png 파일로 저장하여 확인)
# png_data = corrective_rag.get_graph(xray=True).draw_mermaid_png()

# with open("stategraph.png", "wb") as f:
#     f.write(png_data)

inputs = {"question": "스테이크에 어울리는 와인을 추천해주세요."}

for output in corrective_rag.stream(inputs):
    for key, value in output.items():
        #노드 출력
        pprint(f"Node: '{key}':")
        pprint(f"Value: {value}", indent=2, width=80, depth=None)
    print("\n----------------------------------------\n")

#최종 답변
print(value["generation"])
