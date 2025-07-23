#MongoDB 연결 설정
from langchain.document_loaders import PyPDFLoader
from pathlib import Path
import pandas as pd
import json
import os
from tqdm import tqdm

# MongoDB 관련 라이브러리 import
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from urllib.parse import quote_plus  # 상단에 import 추가

def connect_to_mongodb():
    try:
        username = quote_plus("justintak0426")
        password = quote_plus("llmprojectteam3")

        uri = f"mongodb+srv://{username}:{password}@llm-project.5t4zx.mongodb.net/?retryWrites=true&w=majority&appName=llm-project"
        # Create a new client and connect to the server
        client = MongoClient(uri, server_api=ServerApi('1'))
        # Send a ping to confirm a successful connection
       
        try:
            client.admin.command('ping')
            print("Pinged your deployment. You successfully connected to MongoDB!")
        except Exception as e:
            print(e)
        
        # 데이터베이스 선택
        db = client['major']
        
        # 컬렉션 선택
        collection = db['major_info_json']
        
        return client, collection
    
    except Exception as e:
        print(f"MongoDB 연결 실패: {str(e)}")
        return None, None

# JSON 파일을 MongoDB에 업로드하는 함수
def upload_json_to_mongodb(json_file_path, collection):
    try:
        # JSON 파일 읽기
        with open(json_file_path, 'r', encoding='utf-8') as f:
            courses_data = json.load(f)
        
        print(f"총 {len(courses_data)}개의 문서를 업로드합니다.")
        
        # 단순화된 업로드 방식으로 변경
        for course in tqdm(courses_data):
            # 기존 문서가 있는지 확인할 조건 수정
            query = {
                'idx': course['idx'],
                'source_file': course['source_file']
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
        
        # 업로드된 문서 수 확인
        doc_count = collection.count_documents({})
        print(f"컬렉션의 총 문서 수: {doc_count}")
        
    except Exception as e:
        print(f"업로드 중 오류 발생: {str(e)}")


# 기존 코드 맨 뒤에 추가
if __name__ == "__main__":
    # MongoDB 연결
    client, collection = connect_to_mongodb()
    
    if client is not None and collection is not None:
        try:
            # 앞서 저장한 JSON 파일 경로 사용
            json_file_path = "/Users/justin/Desktop/특강/BigData_Camp_LLM_2025/project/major/all_courses.json"  # output_path는 이전 코드에서 정의된 변수
            
            # MongoDB에 업로드
            upload_json_to_mongodb(json_file_path, collection)
            
        finally:
            # MongoDB 연결 종료
            client.close()
            print("MongoDB 연결이 종료되었습니다.")
    else:
        print("MongoDB 연결에 실패하여 업로드를 진행할 수 없습니다.")