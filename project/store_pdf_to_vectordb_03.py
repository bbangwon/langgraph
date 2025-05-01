# 개인정보 보호법 (법률)(제19234호)(20240315).pdf를 벡터 DB에 저장
import os
import re
from pprint import pprint
from glob import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# 법률 문서를 로드하여 벡터 저장소에 저장
pdf_files = glob(os.path.join("data", "*.pdf"))

print(pdf_files)
print("*" * 100)

pdf_file = pdf_files[2] # "data/주택임대차보호법(법률)(제19356호)(20230719).pdf"

loader = PyPDFLoader(pdf_file)
pages = loader.load()

print(len(pages))
print("*" * 100)
print(pages[0].page_content)
print("*" * 100)
print(pages[0].metadata)
print("*" * 100)

def parse_law(law_text):
    #서문 분리
    # '^'로 시작하여 '제1장' 또는 '제1조' 직전까지의 모든 텍스트를 탐색
    preamble_pattern = r'^(.*?)(?=제1장|제1조)'
    preamble = re.search(preamble_pattern, law_text, re.DOTALL)
    if preamble:
        preamble = preamble.group(1).strip()

    # 장 분리
    # '제X장' 형식의 제목과 그 뒤에 오는 모든 조항을 하나의 그룹화
    chapter_pattern = r'(제\d+장\s+.+?)\n((?:제\d+조(?:의\d+)?(?:\(\w+\))?.*?)(?=제\d+장|부칙|$))'
    chapters = re.findall(chapter_pattern, law_text, re.DOTALL)

    # 부칙 분리
    # '부칙'으로 시작하는 모든 텍스트를 탐색
    appendix_pattern = r'(부칙.*)'
    appendix = re.search(appendix_pattern, law_text, re.DOTALL)
    if appendix:
        appendix = appendix.group(1)

    # 파싱 결과를 저장할 딕셔너리 초기화
    parsed_law = {'서문': preamble, '부칙': appendix}


    #조 분리 패턴
    article_pattern = r'(제\d+조(?:의\d+)?\s*\([^)]+\).*?)(?=제\d+조(?:의\d+)?\s*\([^)]+\)|$)'

    if chapters: # 장이 있는 경우
        parsed_law['장'] = {}
        for chapter_title, chapter_content in chapters:
            articles = re.findall(article_pattern, chapter_content, re.DOTALL)
            parsed_law['장'][chapter_title.strip()] = [article.strip() for article in articles]
    else: # 장이 없는 경우
        # 서문과 부칙을 제외한 본문에서 조문 추출
        main_text = re.sub(preamble_pattern, "", law_text, flags=re.DOTALL)
        main_text = re.sub(appendix_pattern, "", main_text, flags=re.DOTALL)
        articles = re.findall(article_pattern, main_text, re.DOTALL)
        parsed_law["조문"] = [article.strip() for article in articles]

    return parsed_law

# 각 페이지의 텍스트를 결합하여 재분리
file_text = "\n".join([p.page_content for p in pages])

text_for_delete = r"법제처\s+\d+\s+국가법령정보센터\n주택임대차보호법"
law_text = "\n".join([re.sub(text_for_delete, "", p.page_content).strip() for p in pages])
parsed_law = parse_law(law_text)

# 분할된 아이템 개수 확인
print(len(parsed_law["조문"]))
print("*" * 100)

# Document 객체에 메타데이터와 함께 정리
final_docs = []
for article in parsed_law["조문"]:

    # metadata 내용을 정리
    metadata = {
        "source": pdf_file,
        "name": "주택임대차보호법"
    }

    #metadata 내용을 본문에 추가
    content = f"[법률정보]\n다음 조항은 {metadata['name']}에서 발췌한 내용입니다.\n\n[법률조항]\n{article}"

    final_docs.append(Document(page_content=content, metadata=metadata))


print(len(final_docs))
print("*" * 100)

print(final_docs[0].page_content)
print("*" * 100)

print(final_docs[0].metadata)
print("*" * 100)

print(final_docs[-1].page_content)
print("*" * 100)

print(final_docs[-1].metadata)
print("*" * 100)


# 벡터저장소에 인덱싱
# 각 문서의 텍스트 길이를 확인

text_lengths = [len(d.page_content) for d in final_docs]
print(min(text_lengths), max(text_lengths))

print("*" * 100)

embeddings_model = OllamaEmbeddings(model="bge-m3")

#Chroma 인덱스 생성
housing_db = Chroma.from_documents(
    documents=final_docs,
    embedding=embeddings_model,
    collection_name="housing_law",
    persist_directory="./chroma_db"
)
