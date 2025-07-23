import json
import openai
import streamlit as st
import os
import base64
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from urllib.parse import quote_plus
from sklearn.metrics.pairwise import cosine_similarity

# OpenAI API 설정
openai.api_key = "2sg8kxsseRytW3HOGXaGe1ESnMlAz9qGW1vpZ6EpkmQbCP2FfHdJJQQJ99BAACfhMk5XJ3w3AAAAACOGmcz2"
openai.api_type = "azure"
openai.api_version = "2023-05-15"
openai.api_base = "https://youngwook-ai.openai.azure.com"

# MongoDB 연결 및 데이터 가져오기
def fetch_data_from_mongodb(db_name, collection_name):
    username = quote_plus("justintak0426")
    password = quote_plus("llmprojectteam3")
    uri = f"mongodb+srv://{username}:{password}@llm-project.5t4zx.mongodb.net/?retryWrites=true&w=majority&appName=llm-project"

    try:
        client = MongoClient(uri, server_api=ServerApi('1'))
        db = client[db_name]
        collection = db[collection_name]
        cursor = collection.find({})

        documents = []
        for doc in cursor:
            doc.pop('_id', None)  # MongoDB의 _id 제거
            documents.append(doc)

        return documents

    except Exception as e:
        print(f"MongoDB 에러: {str(e)}")
        return []

    finally:
        if 'client' in locals():
            client.close()

# 임베딩 모델과 토크나이저 로드
def load_embedding_model():
    model_name = "intfloat/multilingual-e5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

# 텍스트 임베딩 생성
def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    attention_mask = inputs['attention_mask']
    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    return embeddings[0].numpy().tolist()

# 유사도 계산 함수
def calculate_similarity(prompt, course_data, model, tokenizer, topk=5):
    embedded_prompt = get_embedding(prompt, tokenizer, model)
    course_embeddings = [course.get('embedding', []) for course in course_data]

    embedded_prompt_array = np.array(embedded_prompt).reshape(1, -1)
    course_embeddings_array = np.array(course_embeddings)

    similarities = cosine_similarity(embedded_prompt_array, course_embeddings_array)[0]

    sorted_indices = np.argsort(similarities)[::-1]
    top_k_indices = sorted_indices[:topk]

    top_k_courses = [
        {
            'course': course_data[idx],
            'similarity': similarities[idx]
        }
        for idx in top_k_indices
    ]

    return top_k_courses

# Streamlit 페이지 설정
st.set_page_config(
    page_title="강츄",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# CSS 스타일 추가
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .hero-section {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        border: none;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .stat-box {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)

# 현재 디렉토리 기준으로 파일 경로 설정
BASE_DIR = os.path.dirname(__file__)

university_logos = {
    "서울시립대학교": os.path.join(BASE_DIR, "서울시립대로고.png"),
    "전북대학교": os.path.join(BASE_DIR, "전북대로고.png"),
    "한동대학교": os.path.join(BASE_DIR, "한동대로고.png"),
}

# Base64로 이미지 변환 함수
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# 사이드바에서 스타일링
st.sidebar.markdown("""
    <div style="padding: 0.3rem; background: white; border-radius: 0.5rem; box-shadow: 0 2px 3px rgba(0, 0, 0, 0.1);">
        <h2 style="color: #4CAF50;">🌟 학교 선택</h2>
    </div>
    """, unsafe_allow_html=True)

# 사이드바에서 학교 선택
selected_university = st.sidebar.radio(
    "원하는 학교를 선택하세요:",
    options=list(university_logos.keys()),
    format_func=lambda x: x,
)

# Hero 섹션 추가 (학교 로고 포함)
logo_base64 = get_base64_image(university_logos[selected_university])
st.markdown(f"""
    <div class="hero-section">
        <div style="display: flex; align-items: center; justify-content: center;">
            <img src="data:image/png;base64,{logo_base64}" alt="{selected_university}" style="width: 50px; height: 50px; margin-right: 10px;">
            <h1 style="margin: 0;">강츄~💕</h1>
        </div>
        <p>당신에게 필요한 강의를 chu천드릴게요!</p>
    </div>
    """, unsafe_allow_html=True)

# 통계 섹션 추가
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
        <div class="stat-box">
            <h3>3개</h3>
            <p>참여 대학</p>
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
        <div class="stat-box">
            <h3>4개</h3>
            <p>학과</p>
        </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
        <div class="stat-box">
            <h3>100+</h3>
            <p>추천 과목</p>
        </div>
    """, unsafe_allow_html=True)

# 메인 컨텐츠 영역
st.markdown("## 🏫 학과 선택")
if selected_university == "서울시립대학교":
    department = st.selectbox(
        "서울시립대학교의 학과를 선택하세요:",
        ["전자전기컴퓨터공학부", "통계학과"],
    )
elif selected_university == "전북대학교":
    department = st.selectbox(
        "전북대학교의 학과를 선택하세요:",
        ["중어중문학과"],
    )
elif selected_university == "한동대학교":
    department = st.selectbox(
        "한동대학교의 학과를 선택하세요:",
        ["전산전자공학부"],
    )

# 선택된 학교와 학과 표시
st.markdown(
    f"""
    <div style="text-align: center; margin-top: 20px;">
        <h3>🏫 선택된 학교: <span style="color: #4CAF50;">{selected_university}</span></h3>
        <h4>🎓 선택된 학과: <span style="color: #2196F3;">{department}</span></h4>
    </div>
    """,
    unsafe_allow_html=True,
)

# 학교와 학과 입력란 수정
school = selected_university
department = department


def filter_courses_by_school_and_department(course_data, school, department):
    filtered_courses = []
    search_path = f"{school.lower()}_{department.lower()}"  # 대소문자 무시
    for course in course_data:
        file_path = course.get("file_path", "").lower()  # 키가 없으면 빈 문자열
        if search_path in file_path:
            filtered_courses.append(course)
    st.write(f"필터링된 강의 개수: {len(filtered_courses)}")  # 디버깅 로그
    return filtered_courses


# 정보 입력 섹션
st.markdown("""
    <div style="background: white; padding: 1.3rem; border-radius: 0.5rem; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); margin: 1rem 0;">
        <h3>📝 정보 입력</h3>
    </div>
    """, unsafe_allow_html=True)

cols = st.columns(2)
with cols[0]:
    subject = st.text_input("몇 학년인가요?", placeholder="예: 3학년", help="현재 학년을 입력하세요")
with cols[1]:
    content = st.text_area("희망 진로 및 배우고 싶은 내용이 어떻게 되나요?", placeholder="예: AI 연구원", help="구체적인 진로를 입력할수록 더 정확한 추천이 가능합니다")

# 언어 선택
st.markdown("### 🌐 답변 언어 선택")
language = st.radio(
    "답변 언어를 선택하세요(Choose the language):",
    options=["한국어", "English"],
    index=0,
    horizontal=True,
    key="language_selection",
    help="답변 언어를 선택하면 추천 강의표가 해당 언어로 제공됩니다.\n(If you choose the language option, you can get the answer with selected language.)"
)
print(language)

# 언어 선택 변수 설정
if language == "한국어":
    language = "Korean"
else:
    language = "English"

# 버튼 섹션
st.markdown("### 🚀 추천 강의표 생성")
button_click = st.button("✨ 맞춤 강의표 만들기")

prompt = content


if button_click:
    if not subject or not content:
        st.error("⚠️ 학년과 진로를 모두 입력해주세요!")
    else:
        with st.spinner('🔄 강의 추천 중...'):
            try:
                # MongoDB에서 데이터 가져오기
                db_name = 'course_info'
                collection_name = 'embed_course_info_json'

                course_data = fetch_data_from_mongodb(db_name, collection_name)

                # 학교와 학과로 필터링
                filtered_courses = filter_courses_by_school_and_department(course_data, school, department)

                if not filtered_courses:
                    st.error("입력한 학교와 학과에 해당하는 강의가 없습니다.")
                else:
                    # 모델과 토크나이저 로드
                    tokenizer, model = load_embedding_model()

                    # 유사도 계산 및 추천
                    similar_courses = calculate_similarity(prompt, filtered_courses, model, tokenizer, topk=5)

                    # 결과 출력
                    st.markdown("""
                        <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 0.5rem; border-left: 5px solid #4CAF50;">
                            <h3 style="color: #4CAF50;">✨ 추천 강의표</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    st.markdown("### 추천 과목과 이유")

                    for idx, course in enumerate(similar_courses, start=1):
                        course_info = course['course']
                        title = course_info.get('title', '제목 없음')  # 'title' 키가 없는 경우 기본값 사용
                        file_name = course_info.get('file_name', '파일명 없음')  # 파일명 추가
                        content_preview = course_info.get('content', '내용 없음')  # 내용 일부 출력 (100자 제한)

                        response = openai.ChatCompletion.create(
                                engine="dev-gpt-4o-mini",
                                messages=[
                                    {"role": "system", "content": f"Give me answer in {language}"},
                                    {"role": "user", "content": f"나의 희망진로와 배우고 싶은 내용: {prompt}"},
                                    {"role": "user", "content": f"너가 추천한 과목에 대한 정보: {content_preview}"},
                                    {"role": "user", "content": "두 정보를 보고 왜 이 과목을 추천한건지 설명해줘. 과목명을 맨 위에 제시해줘."}
                                ],
                                temperature=0.3,
                                max_tokens=500,
)

                        
                        # 응답 처리 및 출력
                        if response and "choices" in response:
                            assistant_reply = response["choices"][0]["message"]["content"]  # 필요한 내용만 선택
                            
                            st.markdown("### 📝 " + assistant_reply)  # 필요 데이터만 출력
                        else:
                            st.error("⚠️ GPT로부터 유효한 응답을 받지 못했습니다.")
                

            except Exception as e:
                st.error(f"❌ 오류가 발생했습니다: {e}")


# 푸터 추가
st.markdown("""
    <div style="text-align: center; margin-top: 2rem; padding: 1rem; color: #666;">
        <p>© 2025 Your Course Recommender. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)


