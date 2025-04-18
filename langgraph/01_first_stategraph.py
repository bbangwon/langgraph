import random
from typing import TypedDict
from langgraph.graph import StateGraph, START, END



# 상태 Schema 정의 - 사용자의 선호도, 추천된 메뉴, 그리고 메뉴 정보를 저장
class MenuState(TypedDict):
    user_preference: str
    recommended_menu: str
    menu_info: str

# 노드는 상태를 받아서 업데이트 할 상태를 반환하는 함수

def get_user_preference(state: MenuState) -> MenuState:
    print("---랜덤 사용자 선호도 생성---")
    preferences = ["육류", "해산물", "채식", "아무거나"]
    preference = random.choice(preferences)
    print(f"생성된 선호도: {preference}")
    return {"user_preference": preference}

def recommend_menu(state: MenuState) -> MenuState:
    print("---메뉴 추천---")
    preference = state["user_preference"]
    if preference == "육류":
        menu = "스테이크"
    elif preference == "해산물":
        menu = "랍스터 파스타"
    elif preference == "채식":
        menu = "그린 샐러드"
    else:
        menu = "오늘의 쉐프 특선"
    print(f"추천된 메뉴: {menu}")
    return {"recommended_menu": menu}

def provide_menu_info(state: MenuState) -> MenuState:
    print("---메뉴 정보 제공---")
    menu = state["recommended_menu"]
    if menu == "스테이크":
        info = "최상급 소고기로 만든 juicy한 스테이크입니다. 가격: 30,000원"
    elif menu == "랍스터 파스타":
        info = "신선한 랍스터와 al dente 파스타의 조화. 가격: 28,000원"
    elif menu == "그린 샐러드":
        info = "신선한 유기농 채소로 만든 건강한 샐러드. 가격: 15,000원"
    else:
        info = "쉐프가 그날그날 엄선한 특별 요리입니다. 가격: 35,000원"
    print(f"메뉴 정보: {info}")
    return {"menu_info": info}

# 그래프 구성
# 그래프 빌드 생성
builder = StateGraph(MenuState)

# 노드 추가(노드 이름, 노드 함수)
builder.add_node("get_preference", get_user_preference)
builder.add_node("recommend", recommend_menu)
builder.add_node("provide_info", provide_menu_info)

# 엣지 추가
builder.add_edge(START, "get_preference")
builder.add_edge("get_preference", "recommend")
builder.add_edge("recommend", "provide_info")
builder.add_edge("provide_info", END)

# 그래프 컴파일
graph = builder.compile()

# # 그래프 시각화 (png 파일로 저장하여 확인)
# png_data = graph.get_graph().draw_mermaid_png()

# with open("stategraph.png", "wb") as f:
#     f.write(png_data)

# 그래프 실행
def print_result(result: MenuState):
    print("---최종 결과---")
    print(f"사용자 선호도: {result['user_preference']}")
    print(f"추천된 메뉴: {result['recommended_menu']}")
    print(f"메뉴 정보: {result['menu_info']}")

#초기 입력
inputs = {"user_preference": ""}

# 여러 번 실행하여 테스트
for _ in range(2):
    result = graph.invoke(inputs)
    print_result(result)
    print("*" * 100)
    print()
