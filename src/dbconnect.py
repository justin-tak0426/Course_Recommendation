from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from urllib.parse import quote_plus

def fetch_data_from_mongodb(db_name, collection_name):
    """
    MongoDB에서 강의 데이터를 불러와 JSON 리스트 형태로 반환하는 함수
    
    Returns:
        list: 강의 정보가 담긴 딕셔너리들의 리스트
    """
    # MongoDB 접속 정보
    username = quote_plus("justintak0426")
    password = quote_plus("llmprojectteam3")
    uri = f"mongodb+srv://{username}:{password}@llm-project.5t4zx.mongodb.net/?retryWrites=true&w=majority&appName=llm-project"
    
    try:
        # MongoDB 연결
        print("Connecting to MongoDB...")
        client = MongoClient(uri, server_api=ServerApi('1'))
        
        # 데이터베이스와 컬렉션 선택
        db = client[db_name]
        collection = db[collection_name]
        
        # 모든 문서 가져오기
        print("Fetching documents from MongoDB...")
        cursor = collection.find({})  # 빈 쿼리로 모든 문서 조회
        
        # 커서에서 모든 문서를 리스트로 변환
        # _id 필드 제외 (MongoDB 고유 ObjectId는 JSON 직렬화가 안 됨)
        documents = []
        for doc in cursor:
            doc.pop('_id', None)  # _id 필드 제거
            documents.append(doc)
            
        print(f"Successfully fetched {len(documents)} documents")
        return documents
    
    except Exception as e:
        print(f"Error occurred while fetching data: {str(e)}")
        raise e
    
    finally:
        if 'client' in locals():
            client.close()
            print("MongoDB connection closed")

if __name__ == "__main__":
    # 함수 사용 예시
    try:
        # 가져올 db의 이름 설정 (course_info 또는 major)
        db_name = 'course_info'
        # 가져올 collection의 이름 설정 (embed_course_info_json, major_info_json)
        collection_name = 'embed_course_info_json'

        course_data = fetch_data_from_mongodb(db_name, collection_name)
        print(f"Total number of courses: {len(course_data)}")
        # 첫 번째 문서의 키들을 출력하여 어떤 필드들이 있는지 확인
        if course_data:
            print("\nAvailable fields in the documents:")
            print(list(course_data[0].keys()))
        print(type(course_data))
        print(type(course_data[0]))
    except Exception as e:
        print(f"Error in main: {str(e)}")