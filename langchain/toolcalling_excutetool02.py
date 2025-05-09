from datetime import datetime
from pprint import pprint
from langchain_openai import ChatOpenAI
from langchain_community.tools import TavilySearchResults
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

# ChatOpenAI 모델 설정
llm = ChatOpenAI(model="gpt-4o-mini")

# Tavily 검색 도구 초기화 (최대 2개의 결과 반환)
web_search = TavilySearchResults(max_results=2)

#LLM에 도구를 바인딩
llm_with_tools = llm.bind_tools([web_search])

# LLM체인 생성
llm_chain = prompt | llm_with_tools

# 도구 실행 체인 정의
@chain
def web_search_chain(user_input: str, config: RunnableConfig):
    input_ = {"user_input": user_input}
    ai_msg = llm_chain.invoke(input_)   #도구호출.. tool_calls
    print("AI 메시지:", ai_msg)
    print("-" * 100)
    tool_msgs = web_search.batch(ai_msg.tool_calls, config=config)
    print("도구 메시지:", tool_msgs)
    print("-" * 100)
    return llm_chain.invoke({**input_, "messages": [ai_msg, *tool_msgs]}, config=config)

# 체인 실행
response = web_search_chain.invoke("오늘 모엣샹동 샴페인의 가격은 얼마인가요?")

#응답 출력
pprint(response.content)