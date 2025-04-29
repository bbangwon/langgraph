from pprint import pprint
from pydantic import BaseModel, Field
from typing import List, Sequence, TypedDict, Annotated, Literal
from operator import add
from textwrap import dedent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
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

# 그래프 시각화 (png 파일로 저장하여 확인)
# png_data = tool_search_graph.get_graph().draw_mermaid_png()

# with open("stategraph.png", "wb") as f:
#     f.write(png_data)

result = tool_search_graph.invoke({"question": "스테이크 메뉴가 있으면 추천해주세요."})
pprint(result)
print("-"*100)

result = tool_search_graph.invoke({"question": "스테이크 메뉴가 있으면 추천해주세요. 그리고 어울리는 와인도 소개해주세요."})
pprint(result)
print("-"*100)

result = tool_search_graph.invoke({"question": "파스타 메뉴가 있으면 추천해주세요. 그리고, 파스타의 유래에 대해 알려주세요."})
pprint(result)
