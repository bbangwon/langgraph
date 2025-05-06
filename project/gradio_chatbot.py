from typing import List, Tuple
import uuid
import gradio as gr
from gradio_graph import search_builder
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
legal_rag_agent = search_builder.compile(checkpointer=memory, interrupt_before=["human_review"])

# 예시 질문들
example_questions = [
    "사업장에서 CCTV를 설치할 때 주의해야 할 법적 사항은 무엇인가요?",
    "전월세 계약 갱신 요구권의 행사 기간과 조건은 어떻게 되나요?",
    "개인정보 유출 시 기업이 취해야 할 법적 조치는 무엇인가요?",
]

# 챗봇 클래스 생성
class ChatBot:
    def __init__(self):
        self.thread_id = str(uuid.uuid4())
        self.user_decision = False

    def process_message(self, message: str) -> str:
        try:
            config = {"configurable": {"thread_id": self.thread_id}}

            if not self.user_decision:
                # Breakpoint 까지 먼지 실행
                inputs = {"question": message}
                legal_rag_agent.invoke(inputs, config=config)

                #Breadpoint에서 현재 상태를 출력하고, 사용자의 승인 여부를 입력받음
                current_state = legal_rag_agent.get_state(config)
                print(f"Current state: {current_state}")

                final_answer = current_state.values.get("final_answer", "No answer available")
                evaluation_report = current_state.values.get("evaluation_report", {"total_score": 0, 'brief_evaluation': 'No evaluation available'})

                response = f"""현재 답변:
            {final_answer}

            평점 결과:
            총점: {evaluation_report.get('total_score', 0)}/60
            {evaluation_report.get('brief_evaluation', 'No evaluation available')}

            이 답변을 승인하시겠습니까? (y/n): """

                # 사용자 승인 여부를 True로 변경
                self.user_decision = True
                return response
            else:
                # 사용자의 입력에 따라 다음 경로를 선택
                user_decision = message.lower()
                if user_decision == "y":
                    self.user_decision = False # 초기화
                    legal_rag_agent.update_state(config, {"user_decision": "approved"})
                    #나머지 작업을 이어서 진행
                    legal_rag_agent.invoke(None, config=config)
                    #작업이 종료되고 최종 상태의 메시지를 출력
                    current_state = legal_rag_agent.get_state(config)
                    print(f"Final state: {current_state}")
                    return current_state.values.get("final_answer", "No answer available")
                else:
                    self.user_decision = False
                    # 상태 업데이트 - 질문을 수정하여 업데이트
                    legal_rag_agent.update_state(config, {"user_decision": "rejected"})
                    #나머지 작업을 이어서 진행
                    legal_rag_agent.invoke(None, config=config)
                    # Breakpoint 까지 현재 상태를 출력하고, 사용자의 승인 여부를 입력받음
                    current_state = legal_rag_agent.get_state(config)
                    print(f"Revised state: {current_state}")

                    final_answer = current_state.values.get("final_answer", "No answer available")
                    evaluation_report = current_state.values.get("evaluation_report", {"total_score": 0, 'brief_evaluation': 'No evaluation available'})

                    response = f"""다시 생성한 답변:
                {final_answer}

                평가 결과:
                총점: {evaluation_report.get('total_score', 0)}/60
                {evaluation_report.get('brief_evaluation', 'No evaluation available')}

                이 답변을 승인하시겠습니까? (y/n): """

                    # 사용자 승인 여부를 True로 변경
                    self.user_decision = True
                    return response

        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return "죄송합니다. 응답을 생성하는 동안에 오류가 발생했습니다. 다시 시도해 주세요."

    def chat(self, message: str, history: List[Tuple[str, str]]) -> str:
        print(f"Thread ID: {self.thread_id}")
        response = self.process_message(message)
        return response


chatbot = ChatBot()

# ChatInterface 생성
demo = gr.ChatInterface(
    fn=chatbot.chat,
    title="생활법률 AI 어시스턴트",
    description="주택임대차보호법, 근로기준법, 개인정보보호법 관련 질문에 답변해드립니다.",
    examples=example_questions,
    theme=gr.themes.Soft()
)

# Gradio 앱 실행
demo.launch()
