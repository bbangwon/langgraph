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

pdf_file = pdf_files[1] # "data/근로기준법(법률)(제18176호)(20211119).pdf"

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
    parsed_law = {'서문': preamble, '장': {}, '부칙': appendix}

    # 각 장 내에서 조 분리
    for chapter_title, chapter_content in chapters:
        # 조 분리 패턴
        # 1. '제X조'로 시작 ('제X조의Y' 형식도 가능)
        # 2. 조 번호 뒤에 반드시 '(항목명)' 형식의 제목이 와야 함
        # 3. 다음 조가 시작되기 전까지 또는 문서의 끝까지의 모든 내용을 포함
        article_pattern = r'(제\d+조(?:의\d+)?\s*\([^)]+\).*?)(?=제\d+조(?:의\d+)?\s*\([^)]+\)|$)'

        #정규표현식을 이용해 모든 조항을 탐색
        articles = re.findall(article_pattern, chapter_content, re.DOTALL)

        # 각 조항의 앞뒤 공백을 제거하고 결과 딕셔너리에 저장
        parsed_law['장'][chapter_title.strip()] = [article.strip() for article in articles]

    return parsed_law

# 각 페이지의 텍스트를 결합하여 재분리
file_text = "\n".join([p.page_content for p in pages])

text_for_delete = r"법제처\s+\d+\s+국가법령정보센터\n근로기준법"
law_text = "\n".join([re.sub(text_for_delete, "", p.page_content).strip() for p in pages])
parsed_law = parse_law(law_text)

# 분할된 아이템 개수 확인
print(len(parsed_law["장"]))
print("*" * 100)
pprint(parsed_law["장"])
print("*" * 100)

# Document 객체에 메타데이터와 함께 정리
final_docs = []
for law in parsed_law["장"].keys():
    for article in parsed_law["장"][law]:

        # metadata 내용을 정리
        metadata = {
            "source": pdf_file,
            "chapter": law,
            "name": "근로기준법"
        }

        #metadata 내용을 본문에 추가
        content = f"[법률정보]\n다음 조항은 {metadata['name']} {metadata['chapter']}에서 발췌한 내용입니다.\n\n[법률조항]\n{article}"

        final_docs.append(Document(page_content=content, metadata=metadata))


print(len(final_docs))
print("*" * 100)

print(final_docs[0].page_content)
print("*" * 100)

print(final_docs[0].metadata)
print("*" * 100)

print(final_docs[1].page_content)
print("*" * 100)

print(final_docs[1].metadata)
print("*" * 100)


# 벡터저장소에 인덱싱
# 각 문서의 텍스트 길이를 확인

text_lengths = [len(d.page_content) for d in final_docs]
print(min(text_lengths), max(text_lengths))

print("*" * 100)

embeddings_model = OllamaEmbeddings(model="bge-m3")

#Chroma 인덱스 생성
labor_db = Chroma.from_documents(
    documents=final_docs,
    embedding=embeddings_model,
    collection_name="labor_law",
    persist_directory="./chroma_db"
)
