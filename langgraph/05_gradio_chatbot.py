from pprint import pprint
from typing import Annotated, Literal, Tuple, TypedDict, List
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
import gradio as gr

# 기본 State 초기화 방법을 사용
class GraphState(TypedDict):
     messages: Annotated[list[AnyMessage], add_messages]

# LangGraph MessageState라는 미리 만들어진 상태를 사용
class GraphState(MessagesState):
    # messages 키는 기본 제공 - 다른 키를 추가하고 싶을 경우 아래 주석과 같이 적용 가능
    documents: List[Document]
    grade: float
    num_generation: int

embeddings_model = OllamaEmbeddings(model="bge-m3")

# Chroma 인덱스 로드
vector_db = Chroma(
     embedding_function=embeddings_model,    
     collection_name="restaurant_menu",
     persist_directory="./chroma_db",  # Chroma DB가 저장된 경로    
)

# LLM 모델
llm = ChatOpenAI(model="gpt-4o-mini")

# RAG 체인 구성
def format_docs(docs):
     return "\n\n".join(doc.page_content for doc in docs)

system = """
You are a helpful assistant. Use the following context to answer the user's question.

[Context]
{context}
"""

prompt = ChatPromptTemplate.from_messages([
     ("system", system),
     ("human", "{question}")
])

# 검색기 정의
retriever = vector_db.as_retriever(
     search_kwargs={"k": 2}
)

# RAG 체인 구성
rag_chain = (
     {"context": retriever | format_docs, "question": RunnablePassthrough()}
     | prompt
     | llm
     | StrOutputParser()
)

# RAG 체인 실행
query = "채식주의자를 위한 메뉴를 추천해주세요."
response = rag_chain.invoke(query)

# RAG 수행 함수 정의
def retrieve_and_respond(state: GraphState):
     last_human_message = state["messages"][-1]

     # HumanMessage 객체의 content 속성에 접근
     query = last_human_message.content

     # 문서 검색
     retrieved_docs = retriever.invoke(query)

     # 응답 생성
     response = rag_chain.invoke(query)

     # 검색된 문서와 응답을 상태에 저장
     return {
          "messages": [AIMessage(content=response)],
          "documents": retrieved_docs
     }

class GradeResponse(BaseModel):
     "A scope for answers"
     score: float = Field(..., ge=0, le=1, description="A score from 0 to 1, where 1 is perfect")
     explanation: str = Field(..., description="An explanation for the given score")

# 답변 품질 평가 점수
def grade_answer(state: GraphState):
     messages = state['messages']
     question = messages[-2].content #사용자의 질문
     answer =  messages[-1].content #AI의 답변
     context = format_docs(state['documents'])

     grading_system = """You are an expert grader.
     Grade the following answer based on its relevance and  accuracy to the question, considering the given context.
     Provide a score from 0 to 1, where 1 is perfect, along with an explanation.""" #답변의 품질을 평가하기 위한 시스템 메시지

     grading_prompt = ChatPromptTemplate.from_messages([
          ("system", grading_system),
          ("human", "[Question]\n{question}\n\n[Context]\n{context}\n\n[Answer]\n{answer}\n\n[Grade]\n)")
     ])

     grading_chain = grading_prompt | llm.with_structured_output(schema=GradeResponse)
     grade_response = grading_chain.invoke({
          "question": question,
          "context": context,
          "answer": answer
     })

     # 답변 생성 횟수를 증가
     num_generation = state.get("num_generation", 0)
     num_generation += 1

     return {"grade": grade_response.score, "num_generation": num_generation}

def should_retry(state: GraphState) -> Literal["retrieve_and_respond", "generate"]:
     print("----GRADTING----")
     print("Grade Score: ", state["grade"])

     #답변 생성 횟수가 3회 이상이면 "genetrate"를 반환
     if state["num_generation"] > 2:
          return "generate"
     
     #답변 품질 평가점수가 0.7 미만이면 RAG 체인을 다시 실행
     if state["grade"] < 0.7:
          return "retrieve_and_respond"
     else:
          return "generate"
     
# 그래프 실행
builder = StateGraph(GraphState)
builder.add_node("retrieve_and_respond", retrieve_and_respond)
builder.add_node("grade_answer", grade_answer)

builder.add_edge(START, "retrieve_and_respond")
builder.add_edge("retrieve_and_respond", "grade_answer")
builder.add_conditional_edges(
     "grade_answer",
     should_retry,
     {
          "retrieve_and_respond": "retrieve_and_respond",
          "generate": END
     }
)

#그래프 컴파일
graph = builder.compile()

#그래프 시각화 (png 파일로 저장하여 확인)
# png_data = graph.get_graph().draw_mermaid_png()

# with open("stategraph.png", "wb") as f:
#     f.write(png_data)

# initial_state = {
#      "messages": [HumanMessage(content="채식주의자를 위한 메뉴를 추천해주세요.")]
# }

# # 그래프 실행
# final_state = graph.invoke(initial_state)

# # 최종 상태 출력
# print("최종 상태:", final_state)

# # 최종 답변만 출력
# pprint(final_state["messages"][-1].content)

# 예시 질문들
example_quesions = [
    "채식주의자를 위한 메뉴를 추천해주세요.",
    "오늘의 스페셜 메뉴는 무엇인가요?",
    "파스타에 어울리는 음료는 무엇인가요?"
]

# 대답 함수 정의
def answer_invoke(message: str, history: List[Tuple[str, str]]) -> str:
    try:
        # 채팅 기록을 AI에게 전달할 수 있는 형식으로 반환
        chat_history = []
        for human, ai in history:
            chat_history.append(HumanMessage(content=human))
            chat_history.append(AIMessage(content=ai))

        #기존 채팅 기록에 사용자의 메시지를 추가 (최근 2개 대화만 사용)
        initial_state = {
            "messages": chat_history[-2:] + [HumanMessage(content=message)]
        }

        # 메시지를 처리하고 최종 상태를 반환
        final_state = graph.invoke(initial_state)

        #최종 상태에서 필요한 부분 반환 (예: 추천 메뉴 등)
        return final_state["messages"][-1].content
        
    except Exception as e:
        # 오류 발생 시 사용자에게 알리고 로그 기록
        print(f"Error occurred: {str(e)}")
        return "죄송합니다. 응답을 생성하는 동안 오류가 발생했습니다. 다시 시도해 주세요."

# Gradio 인터페이스 설정
demo = gr.ChatInterface(
     fn=answer_invoke,
     title="레스토랑 메뉴 AI 어시스턴트",
     description="메뉴 정보, 추천, 음식 관련 질문에 답변해 드립니다.",
     examples=example_quesions,
     theme=gr.themes.Soft(),
)

# Gradio 앱 실행
demo.launch()
