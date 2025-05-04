from typing import Optional
from common import CorrectiveRagState, ExtractedInformation

# 개인정보보호법
class PersonalRagState(CorrectiveRagState):
    rewritten_query: str # 재작성된 질문
    extracted_info: Optional[ExtractedInformation] # 추출된 정보 조각
    node_answer: Optional[str]
