from common import tools, llm
from textwrap import dedent
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from route_graph import test as route_test

evaluation_prompt = dedent("""
당신은 AI 어시스턴트가 생성한 답변을 평가하는 전문가입니다. 주어진 질문과 답변을 평가하고, 60점 만점으로 점수를 매기세요. 다음 기준을 사용하여 평가하십시오:

1. 정확성 (10점)
2. 관련성 (10점)
3. 완전성 (10점)
4. 인용 정확성 (10점)
5. 명확성과 간결성 (10점)
6. 객관성 (10점)

평가 과정:
1. 주어진 질문과 답변을 주의 깊게 읽으십시오.
2. 필요한 경우, 다음 도구를 사용하여 추가 정보를 수집하세요:
    - web_search: 웹 검색
    - personal_law_search: 개인정보보호법 검색
    - labor_law_search: 근로기준법 검색
    - housing_law_search: 주택임대차보호법 검색

    도구 사용 형식:
    Action: [tool_name]
    Action Input: [Input for the tool]

3. 각 기준에 대해 1-10점 사이의 점수를 매기세요.
4. 총점을 계산하세요 (60점 만점).

출력 형식:
{
    "score": {
        "accuracy": 0,
        "relevance": 0,
        "completeness": 0,
        "citation_accuracy": 0,
        "clarity_conciseness": 0,
        "objectivity": 0
    },
    "total_score": 0,
    "brief_evaluation": "간단한 평가 설명"
}

최종 출력에는 각 기준의 점수, 총점, 그리고 간단한 평가 설명만 포함하세요.
""")

# 그래프 생성
answer_reviewer = create_react_agent(
        llm,
        tools=tools,
        state_modifier=evaluation_prompt
    )

# 그래프 출력# 그래프 시각화 (png 파일로 저장하여 확인)
# png_data = answer_reviewer.get_graph().draw_mermaid_png()

# with open("stategraph.png", "wb") as f:
#     f.write(png_data)

def test(question):
    value = route_test(question)

    messages = [HumanMessage(content=f"""질문\n{value['question']}\n\n[답변]\n{value['final_answer']}""")]
    messages = answer_reviewer.invoke({"messages": messages})
    for m in messages['messages']:
        m.pretty_print()

    # print(json.loads(m.content)['total_score'])

# test("대리인과 아파트 임대차 계약을 체결할 때 주의해야 할 점은 무엇인가요?")
