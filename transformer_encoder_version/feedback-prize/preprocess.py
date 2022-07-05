import pandas as pd
from transformers import BertConfig, BertModel


# 전처리된 상품명을 인덱스로 변환하여 token_ids를 만들었음  token_dict
# 전처리된 상품명을 하나의 텍스트벡터(text_vec)로 변환 BertModel
# 반환 튜플(시퀀스 아웃풋, 풀드(pooled) 아웃풋) 중 시퀀스 아웃풋만 사용

# # bertmodel 로 embedding vector



BertModel()