from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_community.document_loaders import WikipediaLoader

from langchain_openai import ChatOpenAI
from langchain_community.tools import TavilySearchResults
from pprint import pprint
from pydantic import BaseModel, Field
from textwrap import dedent
from langchain_core.tools import tool
from typing import List

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

# 요약 테스트
# summarized_text = summary_chain.invoke({"query": "파스타의 유래"})
# pprint(summarized_text)


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

# #도구 속성
# print("자료형: ")
# print(type(wiki_summary))
# print("-"*100)

# print("name: ")
# print(wiki_summary.name)
# print("-"*100)

# print("description: ")
# pprint(wiki_summary.description)
# print("-"*100)

# print("args_schema: ")
# pprint(wiki_summary.args_schema.model_json_schema())
# print("-"*100)

# LLM에 도구를 바인딩
llm_with_tools = llm.bind_tools(tools=[search_web, wiki_summary])

# 도구 호출이 필요한 LLM 호출을 수행
query = "서울 강남의 유명한 파스타 맛집은 어디인가요? 그리고 파스타의 유래를 알려주세요."
ai_msg = llm_with_tools.invoke(query)

# # LLM의 전체 출력 결과 출력
# pprint(ai_msg)
# print("-" * 100)

# # 메시지 content 출력(텍스트 출력)
# pprint(ai_msg.content)
# print("-" * 100)

# # LLM이 호출한 도구 정보 출력
# pprint(ai_msg.tool_calls)
# print("-" * 100)

# 도구 실행
tool_message = wiki_summary.invoke(ai_msg.tool_calls[1])

print(tool_message)
print("-" * 100)
pprint(tool_message.content)
