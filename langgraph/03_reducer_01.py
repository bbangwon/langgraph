from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

# 상태 정의
class DocumentState(TypedDict):
    query: str
    documents: List[str]

# Node 1: query 업데이트
def node_1(state: DocumentState) -> DocumentState:
    print("---Node 1 (query update)---")
    query = state["query"]
    return {"query": query}

# Node 2: 검색된 문서 추가
def node_2(state: DocumentState) -> DocumentState:
    print("---Node 2 (add documents)---")
    return {"documents": ["doc1.pdf", "doc2.pdf", "doc3.pdf"]}

# Node 3: 추가적인 문서 검색 결과 추가
def node_3(state: DocumentState) -> DocumentState:
    print("---Node 3 (add more documents)---")
    return {"documents": ["doc2.pdf", "doc4.pdf", "doc5.pdf"]}

# 그래프 빌드
builder = StateGraph(DocumentState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# 엣지 추가
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", "node_3")
builder.add_edge("node_3", END)

# 그래프 컴파일
graph = builder.compile()

# 그래프 시각화 (png 파일로 저장하여 확인)
# png_data = graph.get_graph().draw_mermaid_png()

# with open("stategraph.png", "wb") as f:
#     f.write(png_data)

# 초기 상태(documents를 따로 설정하지 않으면 None으로 초기화됨)
initial_state = {"query":"채식주의자를 위한 비건 음식을 추천해주세요."}

# 그래프 실행
final_state = graph.invoke(initial_state)

#최종 상태 출력
print("최종 상태: ", final_state)