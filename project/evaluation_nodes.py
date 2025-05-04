import json
from route_models import ResearchAgentState
from langchain_core.messages import HumanMessage
from evaluation_react_agent import answer_reviewer

def evaluate_answer_node(state:ResearchAgentState):
    question = state['question']
    final_answer = state['final_answer']

    messages = [HumanMessage(content=f"""질문\n{question}\n\n[답변]\n{final_answer}""")]
    response = answer_reviewer.invoke({"messages": messages})
    response_dict = json.loads(response['messages'][-1].content)

    return {"evaluation_report": response_dict, "question": question, "final_answer": final_answer}

# HITL 조건부 엣지 정의
def human_review(state: ResearchAgentState):
    print("\n현재 답변:")
    print(state['final_answer'])
    print("\n평가 결과:")
    print(f"총점: {state['evaluation_report']['total_score']}/60")
    print(state['evaluation_report']['brief_evaluation'])

    user_input = input("\n이 답변을 승인하시겠습니까? (y/n): ").lower()

    if user_input == 'y':
        return "approved"
    else:
        return "rejected"
