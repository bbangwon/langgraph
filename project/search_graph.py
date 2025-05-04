from pprint import pprint
from search_models import SearchRagState
from search_nodes import retrieve_documents, extract_and_evaluate_information, rewrite_query, generate_node_answer, should_continue
from langgraph.graph import StateGraph, START, END

# 그래프 생성
workflow = StateGraph(SearchRagState)

# 노드 추가
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("extract_and_evaluate", extract_and_evaluate_information)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("generate_answer", generate_node_answer)

# 엣지 추가
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "extract_and_evaluate")

# 조건부 엣지 추가
workflow.add_conditional_edges(
    "extract_and_evaluate",
    should_continue,
    {
        "계속": "rewrite_query",
        "종료": "generate_answer"
    }
)
workflow.add_edge("rewrite_query", "retrieve")
workflow.add_edge("generate_answer", END)

# 그래프 컴파일
personal_law_agent = workflow.compile()

# 그래프 시각화 (png 파일로 저장하여 확인)
# png_data = personal_law_agent.get_graph(xray=True).draw_mermaid_png()

# with open("stategraph.png", "wb") as f:
#     f.write(png_data)

def test(question):
    for output in personal_law_agent.stream(question):
        for key, value in output.items():
            #노드 출력
            pprint(f"Node: '{key}':")
            pprint(f"Value: {value}", indent=2, width=80, depth=None)
        print("\n---------------------------------------------\n")

    print(value['node_answer'])

test({"question": "대리인과 아파트 임대차 계약을 체결할 때 주의해야 할 점은 무엇인가요?"})