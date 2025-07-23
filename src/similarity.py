from embedding import *
from dbconnect import fetch_data_from_mongodb
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(prompt, course_data, model, tokenizer, topk=5):
    # 프롬프트 임베딩
    embedded_prompt = get_embedding(prompt, tokenizer, model)
    
    # course_data에서 임베딩 추출
    course_embeddings = [course.get('embedding', []) for course in course_data]
    
    # 코사인 유사도 계산
    # numpy 배열로 변환하여 cosine_similarity 함수 사용
    embedded_prompt_array = np.array(embedded_prompt).reshape(1, -1)
    course_embeddings_array = np.array(course_embeddings)
    
    # 코사인 유사도 계산 (2D 배열 형태로 반환)
    similarities = cosine_similarity(embedded_prompt_array, course_embeddings_array)[0]
    
    # 유사도 기준으로 정렬된 인덱스 추출 (내림차순)
    sorted_indices = np.argsort(similarities)[::-1]
    
    # Top K 개의 강의 추출
    top_k_indices = sorted_indices[:topk]
    
    # 상위 K개 강의와 해당 유사도 반환
    top_k_courses = [
        {
            'course': course_data[idx],
            'similarity': similarities[idx]
        } 
        for idx in top_k_indices
    ]
    
    return top_k_courses



tokenizer, model = load_embedding_model()
prompt = "인공지능에 관련된 과목을 추천받고싶어"

# 가져올 db의 이름 설정 (course_info 또는 major)
db_name = 'course_info'
# 가져올 collection의 이름 설정 (embed_course_info_json, major_info_json)
collection_name = 'embed_course_info_json'

course_data = fetch_data_from_mongodb(db_name, collection_name)

# 반환되는 것은 course dictionary
similar_courses = calculate_similarity(
    prompt, 
    course_data, 
    model, 
    tokenizer, 
    topk=5  # optional, defaults to 5
)

print(similar_courses)