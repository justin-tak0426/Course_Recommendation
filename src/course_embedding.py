import json
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tqdm import tqdm
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from urllib.parse import quote_plus

def load_embedding_model():
    model_name = "intfloat/multilingual-e5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

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

def process_and_upload_to_mongodb():
    root = "/Users/justin/Desktop/특강/BigData_Camp_LLM_2025/project/course/"
    username = quote_plus("justintak0426")
    password = quote_plus("llmprojectteam3")
    uri = f"mongodb+srv://{username}:{password}@llm-project.5t4zx.mongodb.net/?retryWrites=true&w=majority&appName=llm-project"
    
    try:
        # JSON 파일 읽기
        print("Reading the JSON file...")
        with open(root+'final_all_course_info.json', 'r', encoding='utf-8') as f:
            courses_data = json.load(f)
        print(f"Successfully loaded {len(courses_data)} courses")
        
        # 임베딩 모델 로드
        print("Loading embedding model...")
        tokenizer, model = load_embedding_model()
        
        # MongoDB 연결
        print("Connecting to MongoDB...")
        client = MongoClient(uri, server_api=ServerApi('1'))
        db = client['course_info']
        collection = db['embed_course_info_json']
        
        print("Processing documents and creating embeddings...")
        # 단순화된 업로드 방식으로 변경
        for course in tqdm(courses_data):
            # content의 임베딩 생성
            if 'content' in course:
                # 임베딩 생성
                embedding = get_embedding(course['content'], tokenizer, model)
                course['embedding'] = embedding
            
            # 기존 문서가 있는지 확인할 조건 수정
            query = {
                'file_name': course['file_name'],
                'file_path': course['file_path']
            }
            
            try:
                # 개별 문서 업로드
                collection.update_one(
                    query,
                    {'$set': course},
                    upsert=True
                )
            except Exception as doc_error:
                print(f"문서 업로드 중 오류 발생: {doc_error}")
                print(f"문제가 발생한 문서: {course}")
                continue
                
        print("MongoDB 업로드 완료!")
        
        
        # 결과를 새로운 JSON 파일로도 저장
        output_file = root+'final_all_course_info_with_embeddings.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(courses_data, f, ensure_ascii=False, indent=4)
        
        print(f"Updated JSON file saved to: {output_file}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise e
    
    finally:
        if 'client' in locals():
            client.close()
            print("MongoDB connection closed")

if __name__ == "__main__":
    process_and_upload_to_mongodb()