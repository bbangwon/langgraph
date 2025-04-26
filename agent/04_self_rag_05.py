from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

def rewrite_question(question: str) -> str:
    """
    주어진 질문을 벡터 저장소 검색에 최적화된 형태로 다시 작성합니다.

    :param question: 원본 질문 문자열
    :return: 다시 작성된 질문 문자열
    """

    # LLM 모델 초기화
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 시스템 프롬프트 정의
    system_prompt = """
    You are and expert question re-writer. Your task is to convert input questions into optimized versions
    for vectorstore retrieval. Analyze the input carefully and focus on capturing the underlying semantic
    intent and meaning. Your goal is to create a question that will lead to more effective and relevant
    document retrieval.

    [Guidelines]
        1. 질문에서 핵심 개념과 주요 대상을 식별하고 강조합니다.
        2. 약어나 모호한 용어를 풀어서 사용합니다.
        3. 관련 문서에 등장할 수 있는 동의어나 연관된 용어를 포함합니다.
        4. 질문의 원래 의도와 범위를 유지합니다.
        5. 복잡한 질문은 간단하고 집중된 하위 질문으로 나눕니다.

    Remember, the goal is to imporve retrieval effectiveness, not to change the fundamental meaning of the question.
    """

    # 질문 다시 쓰기 프롬프트 템플릿 생성
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "[Initial question]\n{question}\n\n[Improved question]\n"),
        ]
    )

    # 질문 다시 쓰기 체인 구성
    question_rewriter = re_write_prompt | llm | StrOutputParser()

    # 질문 다시 쓰기 실행
    rewritten_question = question_rewriter.invoke({"question": question})

    return rewritten_question

# 질문 다시 쓰기 테스트
question = "대표 메뉴는 무엇인가요?"
rewritten_question = rewrite_question(question)
print(f"원본 질문: {question}")
print(f"다시 작성된 질문: {rewritten_question}")

# 다시 쓴 질문을 사용하여 벡터 저장소에서 문서 검색

embedings_model = OllamaEmbeddings(model="bge-m3")

# 레스토랑 메뉴 검색
menu_db = Chroma(
    embedding_function=embedings_model,
    collection_name="restaurant_menu",
    persist_directory="./chroma_db",  # Chroma DB가 저장된 경로
)


query = rewritten_question
retrieved_docs = menu_db.similarity_search(query, k=2)
print(len(retrieved_docs))

for doc in retrieved_docs:
    print("문서:", doc.page_content)
    print("="*50)
