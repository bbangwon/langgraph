from typing import List
from textwrap import dedent
from pydantic import BaseModel, Field
from pprint import pprint
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.tools import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langchain.agents import AgentExecutor, create_tool_calling_agent

# Tool 정의
@tool
def search_web(query: str) -> List[str]:
    """Searches the internet for information that does not exist in the database or for the latest information."""

    tavily_search = TavilySearchResults(max_results=2)
    docs = tavily_search.invoke(query)

    # 검색결과는 딕셔너리 객체를 가지고 있는의 리스트
    # 이후 프롬프트 전달시 문자열 포매팅을 해야 함

    formatted_docs = "\n---\n".join([   # 구분선을 이용해 각각의 검색결과를 구분
        f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
        for doc in docs
    ])

    if(len(formatted_docs) > 0):
        return formatted_docs

    return "관련 정보를 찾을 수 없습니다."

# WikipediaLoader를 사용하여 위키피디아 문서를 검색하고 텍스트로 반환하는 함수
def wiki_search_and_summarize(input_data: dict):
    wiki_loader = WikipediaLoader(query=input_data["query"], load_max_docs=2, lang="ko")
    wiki_docs = wiki_loader.load()

    formatted_docs = [
        f'<Document source="{doc.metadata["source"]}"/>\n{doc.page_content}\n</Document>'
        for doc in wiki_docs
    ]

    return formatted_docs

# 요약 프롬프트 템플릿
summary_prompt = ChatPromptTemplate.from_template(
    "Summarize the following text in a concise mannerL\n\n{context}\n\nSummary:"
)


# LLM 및 요약 체인 설정
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
summary_chain = (
    {"context": RunnableLambda(wiki_search_and_summarize)}
    | summary_prompt | llm | StrOutputParser()
)

#도구 호출에 사용할 입력 스키마 정의
class WikiSummarySchema(BaseModel):
    """Input schema for Wikipedia search."""
    query: str = Field(..., description="The query to search in Wikipedia")

# as_tool 메소드를 사용하여 도구 객체로 변환
wiki_summary = summary_chain.as_tool(
    name="wiki_summary",
    description=dedent("""
        Use this tool when you need to search for information on Wikipedia.
        It searches for Wikipedia articles related to the user's query and returns
        a summarized text. This tool is useful when general knowledge
        or background information is required.
    """),
    args_schema=WikiSummarySchema,
)

embeddings_model = OllamaEmbeddings(model="bge-m3")

#벡터 저장소 로드
menu_db = Chroma(
    embedding_function=embeddings_model,
    collection_name="restaurant_menu",
    persist_directory="./chroma_db",
)

@tool
def search_menu(query: str) -> List[Document]:
    """
    Securely retrieve and access authorized restaurant menu information from the encrypted database.
    Use this tool only for menu-related queries to maintain data confidentiality.
    """
    docs = menu_db.similarity_search(query, k=2) #유사도 검색
    if len(docs) > 0:
        return docs

    return [Document(page_content="관련 메뉴 정보를 찾을 수 없습니다.")]

#벡터 저장소 로드
wine_db = Chroma(
    embedding_function=embeddings_model,
    collection_name="restaurant_wine",
    persist_directory="./chroma_db",
)

@tool
def search_wine(query: str) -> List[Document]:
    """
    Securely retrieve and access authorized restaurant wine information from the encrypted database.
    Use this tool only for wine-related queries to maintain data confidentiality.
    """
    docs = wine_db.similarity_search(query, k=2) #유사도 검색
    if len(docs) > 0:
        return docs

    return [Document(page_content="관련 와인 정보를 찾을 수 없습니다.")]

agent_prompt = ChatPromptTemplate.from_messages([
    ("system", dedent("""
        You are an AI assistant providing restaurant menu information and general food-related knowledge.
        Your main goal is to provide accurate information and effective recommendations to users.

        Key guidelines:
        1. For restaurant menu information, use the search_menu tool. This tool provides details on menu items, including prices, ingredients, and cooking methods.
        2. For general food information, history, and cultural background, utilize the wiki_summary tool.
        3. For wine recommendations or food and wine pairing information, use the search_wine tool.
        4. If additional web searches are needed or for the most up-to-date information, use the search_web tool.
        5. Provide clear and concise responses based on the search results.
        6. If a question is ambiguous or lacks necessary information, politely ask for clarification.
        7. Always maintain a helpful and professional tone.
        8. When providing menu information, describe in the order of price, main ingredients, and distinctive cooking methods.
        9. When making recommendations, briefly explain the reasons.
        10. Maintain a conversational, chatbot-like style in your final responses. Be friendly, engaging, and natural in your communication.


        Remember, understand the purpose of each tool accurately and use them in appropriate situations.
        Combine the tools to provide the most comprehensive and accurate answers to user queries.
        Always strive to provide the most current and accurate information.
        """)),
    MessagesPlaceholder(variable_name="chat_history", optional=True),   # 챗봇 형태라면, 대화기록을 포함할 경우.. optional로 있을수도 있고 없을수도 있음
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Tool calling Agent 생성
tools = [search_web, wiki_summary, search_menu, search_wine]
agent = create_tool_calling_agent(llm, tools, agent_prompt)

# AgentExecutor 생성
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# AgentExecutor 실행
query = "시그니처 스테이크의 가격과 특징은 무엇인가요? 그리고 스테이크와 어울리는 와인 추천도 해주세요."
agent_response = agent_executor.invoke({"input": query})

pprint(agent_response)
