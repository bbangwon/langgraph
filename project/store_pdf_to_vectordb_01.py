# 개인정보 보호법 (법률)(제19234호)(20240315).pdf를 벡터 DB에 저장
import os
from glob import glob
from langchain_community.document_loaders import PyPDFLoader

# 법률 문서를 로드하여 벡터 저장소에 저장
pdf_files = glob(os.path.join("data", "*.pdf"))

print(pdf_files)
print("*" * 100)

pdf_file = pdf_files[0] # "data/개인정보 보호법(법률)(제19234호)(20240315).pdf"

loader = PyPDFLoader(pdf_file)
pages = loader.load()

print(len(pages))
print("*" * 100)
print(pages[0].page_content)
print("*" * 100)
print(pages[0].metadata)
print("*" * 100)
