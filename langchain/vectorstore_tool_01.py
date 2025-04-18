from langchain_community.document_loaders import TextLoader

# 메뉴판 텍스트 데이터를 로드
loader = TextLoader("./data/restaurant_menu.txt", encoding="utf-8")
documents = loader.load()

print(len(documents))  # 문서 개수 출력