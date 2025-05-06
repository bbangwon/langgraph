from langgraph.checkpoint.memory import MemorySaver
from route_models import ResearchAgentState
from route_nodes import analyze_question_tool_search, personal_rag_node, labor_rag_node, housing_rag_node, web_rag_node, answer_final, llm_fallback, route_datasources_tool_search
from evaluation_nodes import evaluate_answer_node
from langgraph.graph import StateGraph, START, END

# HITL 노드로 변경 (그라디오에서 입력을 처리)
def human_review(state: ResearchAgentState):
    pass

# 그래프 생성을 위한 StateGraph 객체를 정의
search_builder = StateGraph(ResearchAgentState)

#노드 추가
nodes = {
    "analyze_question": analyze_question_tool_search,
    "search_personal": personal_rag_node,
    "search_labor": labor_rag_node,
    "search_housing": housing_rag_node,
    "search_web": web_rag_node,
    "generate_answer": answer_final,
    "llm_fallback": llm_fallback,
    "evaluate_answer": evaluate_answer_node,
    "human_review": human_review,
}

# 노드 추가
for node_name, node_func in nodes.items():
    search_builder.add_node(node_name, node_func)

# 엣지 추가 (병렬처리)
search_builder.add_edge(START, "analyze_question")
search_builder.add_conditional_edges(
    "analyze_question",
    route_datasources_tool_search,
    ["search_personal", "search_labor", "search_housing", "search_web", "llm_fallback"],
)

# 검색 노드들을 generate_answer에 연결
for node in ["search_personal", "search_labor", "search_housing", "search_web"]:
    search_builder.add_edge(node, "generate_answer")

search_builder.add_edge("generate_answer", "evaluate_answer")
search_builder.add_edge("evaluate_answer", "human_review")

# HITL 결과에 따른 조건부 엣지 추가
search_builder.add_conditional_edges(
    "human_review",
    lambda x: "approved" if x.get("user_decision") == "approved" else "rejected",
    {
        "approved": END,
        "rejected": "analyze_question", #승인되지 않은 경우 질문 분석 단계로 돌아감
    }
)


search_builder.add_edge("llm_fallback", END)

# 메모리 추가
# 그래프 컴파일 (Breakpoint 설정)
# 그래프 시각화 (png 파일로 저장하여 확인)

# memory = MemorySaver()
# legal_rag_graph = search_builder.compile(checkpointer=memory, interrupt_before=["human_review"])

# png_data = legal_rag_graph.get_graph().draw_mermaid_png()

# with open("stategraph.png", "wb") as f:
#     f.write(png_data)
