from langchain_community.tools import TavilySearchResults
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from typing import List
from pprint import pprint
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, chain


#오늘 날짜 설정
today = datetime.today().strftime("%Y-%m-%d")

# 프롬프트 템플릿 설정
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", f"You are a helpful AI assistant."),
        ("system", f"Today's date is {today}."),
        ("human", "{user_input}"),
        ("placeholder", "{messages}"),
    ]
)

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


llm = ChatOpenAI(model = "gpt-4o-mini")

# LLM에 도구를 바인딩
llm_with_tool = llm.bind_tools([search_web])

llm_chain = prompt | llm_with_tool

@chain
def web_search_chain(user_input: str, config: RunnableConfig):
    input_ = {"user_input": user_input}
    ai_msg = llm_chain.invoke(input_)   #도구호출.. tool_calls
    print("AI 메시지:", ai_msg)
    print("-" * 100)
    tool_msgs = search_web.batch(ai_msg.tool_calls, config=config)
    print("도구 메시지:", tool_msgs)
    print("-" * 100)
    return llm_chain.invoke({**input_, "messages": [ai_msg, *tool_msgs]}, config=config)


response = web_search_chain.invoke("스테이크와 어울리는 와인을 추천해주세요.")

#응답 출력
pprint(response.content)