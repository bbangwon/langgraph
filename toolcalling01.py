import warnings
from langchain_community.tools import TavilySearchResults

warnings.filterwarnings("ignore")

# 검색할 쿼리 설정
query = "스테이크와 어울리는 와인을 추천해주세요."

# Tavily 검색 도구 초기화 (최대 2개의 결과 반환)
web_search = TavilySearchResults(max_results=2)

#웹 검색 실행
search_results = web_search.invoke(query)

# # 검색 결과 출력
for result in search_results:
    print(result)
    print("-" * 100)

