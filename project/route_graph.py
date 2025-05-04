from pprint import pprint
from route_nodes import analyze_question_tool_search, personal_rag_node, labor_rag_node, housing_rag_node, web_rag_node, answer_final, llm_fallback, route_datasources_tool_search
from route_models import ResearchAgentState
from langgraph.graph import StateGraph, START, END

nodes = {
    "analyze_question": analyze_question_tool_search,
    "search_personal": personal_rag_node,
    "search_labor": labor_rag_node,
    "search_housing": housing_rag_node,
    "search_web": web_rag_node,
    "generate_answer": answer_final,
    "llm_fallback": llm_fallback,
}

# 그래프 생성을 위한 StateGraph 객체를 정의
search_builder = StateGraph(ResearchAgentState)

# 노드 추가
for node_name, node_func in nodes.items():
    search_builder.add_node(node_name, node_func)

# 엣지 추가(병렬처리)
search_builder.add_edge(START, "analyze_question")
search_builder.add_conditional_edges(
    "analyze_question",
    route_datasources_tool_search,
    ["search_personal", "search_labor", "search_housing", "search_web", "llm_fallback"],
)

# 검색 노드들을 generate_answer에 연결
for node in ["search_personal", "search_labor", "search_housing", "search_web"]:
    search_builder.add_edge(node, "generate_answer")

search_builder.add_edge("generate_answer", END)
search_builder.add_edge("llm_fallback", END)

# 그래프 컴파일
rag_search_graph = search_builder.compile()

# 그래프 시각화 (png 파일로 저장하여 확인)
# png_data = rag_search_graph.get_graph().draw_mermaid_png()

# with open("stategraph.png", "wb") as f:
#     f.write(png_data)

def test(question):
    inputs = {"question": question}
    for output in rag_search_graph.stream(inputs):
        for key, value in output.items():
            #노드 출력
            pprint(f"Node: '{key}':")
            pprint(f"Value: {value}", indent=2, width=80, depth=None)
        print("\n---------------------------------------------\n")

    print(value['final_answer'])

# test("대리인과 아파트 임대차 계약을 체결할 때 주의해야 할 점은 무엇인가요?")
