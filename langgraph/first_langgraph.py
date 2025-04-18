from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END

#그래프의 전체 상태(이것은 노드 간 공유되는 공용 상태)
class OverallState(BaseModel):
    text: str

# 노드 함수
def node(state: OverallState):
    return {"text": "반갑습니다"} # 상태를 변경해서 출력

# 그래프 구축 (엣지로 연결)
builder = StateGraph(OverallState)
builder.add_node(node) #첫 번째 노드
builder.add_edge(START, "node") # 그래프는 node로 시작
builder.add_edge("node", END) # node 실행 후 그래프를 종료
graph = builder.compile()

#유효한 입력으로 그래프를 테스트
result = graph.invoke({"text": "안녕하세요"})
print(result["text"])

