from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from textwrap import dedent
from typing import List
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver


embeddings_model = OllamaEmbeddings(model="bge-m3")

# Chroma 인덱스 로드
vector_db = Chroma(
     embedding_function=embeddings_model,
     collection_name="restaurant_menu",
     persist_directory="./chroma_db",  # Chroma DB가 저장된 경로
)

# Tool 정의
@tool
def search_menu(query: str) -> List[str]:
    """ 레스트랑 메뉴에서 정보를 검색합니다. """
    docs = vector_db.similarity_search(query, k=2)

    formatted_docs = "\n\n---\n\n".join(
        [
            f'<document source="{doc.metadata["source"]}"/>\n{doc.page_content}\n</Document>'
            for doc in docs
        ]
    )

    if len(docs) > 0:
        return formatted_docs
    
    return "관련 메뉴 정보를 찾을 수 없습니다."

@tool
def search_web(query: str) -> List[str]:
    """ 데이터베이스에 존재하지 않는 정보 또는 최신 정보를 인터넷에서 검색합니다."""

    tavily_search = TavilySearchResults(max_results=3)
    docs = tavily_search.invoke(query)

    formatted_docs = "\n\n---\n\n".join(
        [
            f'<document source="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in docs
        ]
    )

    if len(docs) > 0:
        return formatted_docs
    
    return "관련 웹 정보를 찾을 수 없습니다."

# LLM 모델
llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)

# 도구 목록
tools = [
    search_menu,
    search_web
]

# 시스템 프롬프트
system_prompt = dedent("""
You are a AI assistant designed to answer human questions.
You can use the provided tools to help generate your responses.
                       
Follow these steps to answer questions:
    1. Carefully read and understand the question.
    2. Use the provided tools to obtain necessary information.
    3. Immediately after using a tool, cite the source using the format below.
    4. Construct an accurate and helpful answer using the tool outputs and citations.
    5. Provide the final answer when you determine it's complete.
                       
When using tools, follow this format:
    Action: tool_name
    Action Input: input for the tool
                       
Immediately after receiving tool output, cite the source as follows:
    [Source: tool_name | document_title/item_name | url/file_path]
                       
For example:
    Action: search_menu
    Action Input: 스테이크
                       
    (After receiving tool output)
    [Source: search_menu | 스테이크 | ./data/data.txt]
    스테이크에 대한 정보는 다음과 같습니다...
    
    Action: search_web
    Action Input: History of AI
                       
    (After receiving tool output)
    [Source: search_web | AI History | https://en.wikipedia.org/wiki/History_of_artificial_intelligence]
    AI의 역사는 다음과 같이 요약됩니다...
                       
If tool use is not necessary, answer directly.
                       
Your final answer should be clear, concise, and directly related to the user's question.
Ensure that every place of factual information in your response is accompanied by a citation.
                       
Remember: ALWAYS include these citations for all factual information. tool outputs, and referenced documents in your response.
Do not provide any information without a corresponding citation.
""")

# 메모리 초기화
memory = MemorySaver()

# 그래프 생성
graph = create_react_agent(
    llm,
    tools=tools,
    state_modifier=system_prompt,
    checkpointer=memory
)

# 그래프 출력
# 그래프 시각화 (png 파일로 저장하여 확인)
# png_data = graph.get_graph().draw_mermaid_png()

# with open("stategraph.png", "wb") as f:
#     f.write(png_data)

config = {"configurable": {"thread_id": "2"}}
messages = [HumanMessage(content="채식주의자를 위한 메뉴가 있나요?")]
messages = graph.invoke({"messages": messages}, config=config)
for m in messages['messages']:
    m.pretty_print()

config = {"configurable": {"thread_id": "2"}}
messages = [HumanMessage(content="방금 답변에 대한 출처가 있나요?")]
messages = graph.invoke({"messages": messages}, config=config)
for m in messages['messages']:
    m.pretty_print()
