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



def pdf2txt(course_dir):
   try:
       # 결과를 저장할 리스트
       courses_info = []
       
       course_dir = Path(course_dir)
       print("Scanning ", course_dir)
       
       # 디렉토리의 모든 pdf 파일 찾기
       pdf_files = list(course_dir.glob('*.pdf'))
       print("length of files: ", len(pdf_files))
       
       if not pdf_files:
           print(f"Warning: No pdf files found in {course_dir}")
           return []

       # 각 파일 처리
       for file_path in tqdm(pdf_files):
           try:
               # pdf 파일 읽기
               loader = PyPDFLoader(str(file_path))
               pages = loader.load_and_split()
               
               # 각 PDF 파일의 모든 페이지 내용을 하나의 문자열로 합치기
               full_text = ""
               for page in pages:
                   full_text += page.page_content + "\n"
               
               # 파일 정보와 내용을 딕셔너리로 저장
               course_info = {
                   'file_name': file_path.name,
                   'file_path': str(file_path),
                   'num_pages': len(pages),
                   'content': full_text
               }
               
               courses_info.append(course_info)
               print(f"Successfully processed: {file_path}")
               print("Pages: ", len(pages))

           except Exception as e:
               print(f"Error processing {file_path}: {str(e)}")
               continue
       
       return courses_info

   except Exception as e:
       print(f"Error processing directory: {str(e)}")
       return []


# MongoDB 연결 설정
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
        db = client['course_info']
        
        # 컬렉션 선택
        collection = db['course_info_json']
        
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
            # 기존 문서가 있는지 확인할 조건
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
                print(f"문제가 발생한 파일: {course['file_name']}")
                continue
                
        print("MongoDB 업로드 완료!")
        
        # 업로드된 문서 수 확인
        doc_count = collection.count_documents({})
        print(f"컬렉션의 총 문서 수: {doc_count}")
        
    except Exception as e:
        print(f"업로드 중 오류 발생: {str(e)}")

# 기존 코드 맨 뒤에 추가
if __name__ == "__main__":
    # 사용 예시
    project_dir = "/Users/justin/Desktop/특강/BigData_Camp_LLM_2025/project"
    course_dir = Path(project_dir + "/course")

    # 하위 디렉토리 찾기
    uni_course_dirs = list(course_dir.glob("*"))  # 모든 항목을 가져옴
    uni_course_dirs = [d for d in uni_course_dirs if d.is_dir()]  # 디렉토리만 필터링
    print("Found directories:", uni_course_dirs)

    # 각 디렉토리의 PDF 처리
    all_courses_info = []
    for dir_path in uni_course_dirs:
        course_infos = pdf2txt(dir_path)
        all_courses_info.extend(course_infos)

    # 결과 확인
    print(f"\nTotal PDFs processed: {len(all_courses_info)}")
    if all_courses_info:
        print("\nFirst entry sample:")
        print(json.dumps(all_courses_info[0], ensure_ascii=False, indent=4))

    # JSON 파일로 저장
    output_path = course_dir / "all_courses_info.json"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_courses_info, f, ensure_ascii=False, indent=4)

    print(f"\nJSON file saved at: {output_path}")


    # MongoDB 연결
    client, collection = connect_to_mongodb()
    
    if client is not None and collection is not None:
        try:
            # 앞서 저장한 JSON 파일 경로 사용
            json_file_path = str(output_path)  # output_path는 이전 코드에서 정의된 변수
            
            # MongoDB에 업로드
            upload_json_to_mongodb(json_file_path, collection)
            
        finally:
            # MongoDB 연결 종료
            client.close()
            print("MongoDB 연결이 종료되었습니다.")
    else:
        print("MongoDB 연결에 실패하여 업로드를 진행할 수 없습니다.")