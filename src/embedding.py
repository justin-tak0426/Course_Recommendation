import json
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tqdm import tqdm
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from urllib.parse import quote_plus

# 임베딩 모델과 토크나이저를 로딩
# parameter: none
# return: 토크나이저와 모델
def load_embedding_model():
    model_name = "intfloat/multilingual-e5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

# 텍스트데이터와, 토크나이저, 모델을 파라미터로 넘기면 임베딩된 벡터 반환
# parameter: 텍스트데이터, 토크나이저, 모델
# return: 임베딩된 벡터 (numpy 배열)
def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # mean pooling
    attention_mask = inputs['attention_mask']
    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    return embeddings[0].numpy().tolist()