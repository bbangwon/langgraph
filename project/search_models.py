from typing import Optional
from common import CorrectiveRagState, ExtractedInformation

class SearchRagState(CorrectiveRagState):
    rewritten_query: str # 재작성된 질문
    extracted_info: Optional[ExtractedInformation] # 추출된 정보 조각
    node_answer: Optional[str] # 노드에서 생성된 답변
