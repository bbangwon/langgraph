from textwrap import dedent
from typing import TypedDict, List, Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from pprint import pprint
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_community.tools import TavilySearchResults
from langchain_core.tools import tool
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

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

# 상태 Schema 정의
class AdaptiveRagState(TypedDict):
    question: str
    documents: List[Document]
    generation: str

# 라우팅 결정을 위한 데이터 모델
class ToolSelector(BaseModel):
    """Routes the user quesion to the most appropriate tool."""
    tool: Literal["search_menu", "search_web", "search_wine"] = Field(
        description="Select one of the tools: search_menu, search_wine or search_web based on the user's question."
    )

# 구조회된 출력을 위한 LLM 설정
structured_llm = llm.with_structured_output(ToolSelector)

# 라우팅을 위한 프롬프트 템플릿
system = dedent("""Ypu are an AI assistant specializing in routing user questions to the appropriate tool.
Use the following guidlines:
- For questions about the restaurant's menu, use the search_menu tool.
- For wine recommendations or pairing information, use the search_wine tool.
- For any other information or the most up-to-date data, use the search_web tool.
Always choose the most appropriate tool based on the user's question.""")

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# 질문 라이터 정의
question_router = route_prompt | structured_llm

# 테스트 실행
# print(question_router.invoke({"question": "채식주의자를 위한 메뉴가 있나요?"}))
# print(question_router.invoke({"question": "스테이크 메뉴와 어울리는 와인을 추천해주세요."}))
# print(question_router.invoke({"question": "2022년 월드컵 우승 국가는 어디인가요?"}))

# 질문 라우팅 노드
def route_question_adaptive(state: AdaptiveRagState) -> Literal["search_menu", "search_wine", "search_web", "llm_fallback"]:
    question = state["question"]
    try:
        result = question_router.invoke({"question": question})
        datasource = result.tool

        if datasource == "search_menu":
            return "search_menu"
        elif datasource == "search_wine":
            return "search_wine"
        elif datasource == "search_web":
            return "search_web"
        else:
            return "llm_fallback"

    except Exception as e:
        print(f"Error in routing: {str(e)}")
        return "llm_fallback"

# 검색 노드
def search_menu_adaptive(state: AdaptiveRagState):
    """
    Node for searching information in the restaurant menu
    """
    question = state["question"]
    docs = search_menu.invoke(question)
    if len(docs) > 0:
        return {"documents": docs}
    else:
        return {"documents": [Document(page_content="관련 메뉴 정보를 찾을 수 없습니다.")]}

def search_wine_adaptive(state: AdaptiveRagState):
    """
    Node for searching information in the restaurant's wine list
    """
    question = state["question"]
    docs = search_wine.invoke(question)
    if len(docs) > 0:
        return {"documents": docs}
    else:
        return {"documents": [Document(page_content="관련 와인 정보를 찾을 수 없습니다.")]}

def search_web_adaptive(state: AdaptiveRagState):
    """
    Node for searching the web for information not available int the restaurant menu
    or for up-to-date information, and returning the results
    """
    question = state["question"]
    docs = search_web.invoke(question)
    if len(docs) > 0:
        return {"documents": docs}
    else:
        return {"documents": [Document(page_content="관련 웹 정보를 찾을 수 없습니다.")]}

# 생성 노드
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an assistant answering questions based on provided documents. Follow these guidelines:

1. Use only information from the given documents.
2. If the docuemt lacks relevant info, say "The provided documents don't contain information to answer this question.
3. Cite relevant parts of the document in your answers.
4. Don't speculate or add information not in the documents.
5. Keep answers concise and clear.
6. Omit irrelevant information."""
    ),
    ("human", "Answer the following question using these documents:\n\n[Documents]\n{documents}\n\n[Question]\n{question}"),
])

def generate_adaptive(state: AdaptiveRagState):
    """
    Genetate answer using the retrieved_documents
    """
    question = state.get("question", None)
    documents = state.get("documents", [])
    if not isinstance(documents, list):
        documents = [documents]

    #문서 내용을 문자열로 변환
    documents_text = "\n\n".join([f"---\n본문: {doc.page_content}\n메타데이터: {str(doc.metadata)}\n---" for doc in documents])

    # RAG generatoion
    rag_chain = rag_prompt | llm | StrOutputParser()
    generation = rag_chain.invoke({"documents": documents_text, "question": question})
    return {"generation": generation}

# LLM Fallback 프롬프트 정의
fallback_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an AI assistant helping woth various topics. Follow these guidelines:

1. Providee accurate and helpful information to the best of your ability.
2. Express uncertatinty when unsure; avoid speculation.
3. Keep answers concise yet informative.
4. Inform users they can ask for clarification if needed.
5. Respond ethically and constructively.
6. Mention reliable general sources when applicable."""),
    ("human", "{question}")
])

def llm_fallback_adaptive(state: AdaptiveRagState):
    """
    Generate answer using the LLM without context
    """
    question = state.get("question", "")

    # LLM chain
    llm_chain = fallback_prompt | llm | StrOutputParser()

    generation = llm_chain.invoke({"question": question})
    return {"generation": generation}

# 그래프 연결
builder = StateGraph(AdaptiveRagState)

# 노드 추가
builder.add_node("search_menu", search_menu_adaptive)
builder.add_node("search_wine", search_wine_adaptive)
builder.add_node("search_web", search_web_adaptive)
builder.add_node("generate", generate_adaptive)
builder.add_node("llm_fallback", llm_fallback_adaptive)

# 엣지 추가
builder.add_conditional_edges(
    START,
    route_question_adaptive
)

builder.add_edge("search_menu", "generate")
builder.add_edge("search_wine", "generate")
builder.add_edge("search_web", "generate")
builder.add_edge("generate", END)
builder.add_edge("llm_fallback", END)

memory = MemorySaver()

# 컴파일 - 'agent', 'generate' 노드 전에 중단점 추가
adaptive_rag_hitl = builder.compile(checkpointer=memory, interrupt_before=["generate"])

# 그래프 시각화 (png 파일로 저장하여 확인)
# png_data = adaptive_rag_hitl.get_graph().draw_mermaid_png()

# with open("stategraph.png", "wb") as f:
#     f.write(png_data)

# 도구 사용 전 중단점에서 실행을 멈춤
thread = {"configurable": {"thread_id": "breakpoint_update"}}
inputs = {"question": "매운 음식이 있나요?"}
for event in adaptive_rag_hitl.stream(inputs, config=thread):
    for k, v in event.items():
        # '__end__' 이벤트는 미출력
        if k != "__end__":
            print(f"{k}: {v}")  #이벤트의 키와 값을 함께 출력

# 상태확인
current_state = adaptive_rag_hitl.get_state(thread)

#다음에 실행될 노드를 확인
print(current_state.next)
print("-"*50)

# question, generation 필드 확인
print(current_state.values.get("question"))
print("-"*50)
print(current_state.values.get("generation"))

# 상태 업데이트 - 질문을 수정하여 업데이트
adaptive_rag_hitl.update_state(thread, {"question": "매콤한 해산물 요리가 있나요?"})

# 상태확인
new_state = adaptive_rag_hitl.get_state(thread)

print(new_state.values.get("question"))
print("-"*50)
print(new_state.values.get("generation"))

# 입력값을 None으로 지정하면 중단점부터 실행하고 최종 답변을 생성
for event in adaptive_rag_hitl.stream(None, config=thread):
    for k, v in event.items():
        # '__end__' 이벤트는 미출력
        if k != "__end__":
            print(f"{k}: {v}")  #이벤트의 키와 값을 함께 출력

# 최종 답변 확인
print(event["generate"]["generation"])
