import json
import re
from typing import List, Dict, Any

def preprocess_uos_stat(content: str) -> str:
    """서울시립대학교 통계학과 텍스트 전처리"""
    try:
        # 1. "출력사용자:박지원\n수업계획서" 부터 "\n교과목명(영문명)" 전까지 제거 (시작 포함, 끝 미포함)
        content = re.sub(r'출력사용자:.*?\n(?=교과목명\(영문명\))', '', content, flags=re.DOTALL)
        
        # 2. "연락처" 부터 "개설학년" 전까지 제거 (시작 포함, 끝 미포함)
        content = re.sub(r'연락처.*?(?=개설학년)', '', content, flags=re.DOTALL)
        
        # 3. "집중수업구분" 부터 "전공능력 전공능력 대표성" 전까지 제거 (시작 포함, 끝 미포함)
        content = re.sub(r'집중수업구분.*?(?=전공능력 전공능력 대표성)', '', content, flags=re.DOTALL)
        
        # 4. "교재" 부터 "\n주 수업내용" 전까지 제거 (시작 포함, 끝 미포함)
        content = re.sub(r'교재.*?(?=\n주 수업내용)', '', content, flags=re.DOTALL)
        
        # 5. "수업방법" 부터 다음 "\n" 전까지 제거 (시작 포함, 끝 미포함)
        content = re.sub(r'수업방법.*?(?=\n)', '', content, flags=re.DOTALL)
        
        # 6. "\n2025/01/15" 부터 끝까지 제거 (시작 포함)
        content = re.sub(r'\n2025/01/15.*$', '', content, flags=re.DOTALL)
        
        # 연속된 공백과 줄바꿈 정리
        content = re.sub(r'\s+', ' ', content).strip()
        return content
    except Exception as e:
        print(f"Error preprocessing UOS Statistics content: {e}")
        return content

def preprocess_uos_eece(content: str) -> str:
    """서울시립대학교 전자전기컴퓨터공학부 텍스트 전처리"""
    # 통계학과와 동일한 전처리 로직 사용
    return preprocess_uos_stat(content)

def preprocess_jbnu_chinese(content: str) -> str:
    """전북대학교 중어중문학과 텍스트 전처리"""
    try:
        content = re.sub(r'^.*?(?=\n학점)', '', content, flags=re.DOTALL)
        content = re.sub(r'상담가능시간.*?(?=\n수업목표)', '', content, flags=re.DOTALL)
        content = re.sub(r'\n6대 핵심역량과의 관계.*?(?=DA\n수업방식)', '', content, flags=re.DOTALL)
        content = re.sub(r'√\n \* 장애학생 교수학습지원 사항.*?\n기타 참고사항 온라인 오프라인', '', content, flags=re.DOTALL)
        content = re.sub(r'\s+', ' ', content).strip()
        return content
    except Exception as e:
        print(f"Error preprocessing JBNU Chinese content: {e}")
        return content

def preprocess_handong_ce(content: str) -> str:
    """한동대학교 전산전자공학부 텍스트 전처리"""
    try:
        content = re.sub(r'개설년도.*?강의실 강의시간\n', '', content, flags=re.DOTALL)
        content = re.sub(r'● 평가\n.*?● 과제 및 프로젝트\(Assignments and Projects\)\n', '', content, flags=re.DOTALL)
        content = re.sub(r'\. 공지사항/부가정보\n.*$', '', content, flags=re.DOTALL)
        content = re.sub(r'\s+', ' ', content).strip()
        return content
    except Exception as e:
        print(f"Error preprocessing Handong University content: {e}")
        return content

def get_department_from_path(file_path: str) -> str:
    """파일 경로에서 학과 정보를 추출하는 함수"""
    import os
    import unicodedata
    
    # 경로를 정규화하고 디렉토리 부분만 추출
    normalized_path = unicodedata.normalize('NFC', os.path.normpath(file_path).replace("\\", "/"))
    path_parts = normalized_path.split("/")
    
    # course 디렉토리 다음에 오는 디렉토리를 학과로 간주
    try:
        course_index = path_parts.index("course")
        if course_index + 1 < len(path_parts):
            return unicodedata.normalize('NFC', path_parts[course_index + 1])
    except ValueError:
        pass
    
    return None

def preprocess_content(content: str, department: str) -> str:
    """학과별 전처리 함수를 호출하는 함수"""
    preprocessing_methods = {
        "서울시립대학교_통계학과": preprocess_uos_stat,
        "서울시립대학교_전자전기컴퓨터공학부": preprocess_uos_eece,
        "전북대학교_중어중문학과": preprocess_jbnu_chinese,
        "한동대학교_전산전자공학부": preprocess_handong_ce
    }
    
    print(f"Preprocessing for department: '{department}'")
    print("Available departments:")
    for key in preprocessing_methods.keys():
        print(f"- '{key}' (length: {len(key)})")
    print(f"Input department length: {len(department)}")
    print(f"Ascii codes for input department: {[ord(c) for c in department]}")
    
    if department in preprocessing_methods:
        print(f"Found preprocessing method for {department}")
        result = preprocessing_methods[department](content)
        if result == content:
            print("Warning: Preprocessing did not change the content!")
        return result
    
    # Try with stripped whitespace
    stripped_dept = department.strip()
    if stripped_dept in preprocessing_methods:
        print(f"Found preprocessing method after stripping whitespace")
        return preprocessing_methods[stripped_dept](content)
    
    print(f"No preprocessing method found for '{department}'")
    print("Character by character comparison with '서울시립대학교_전자전기컴퓨터공학부':")
    expected = "서울시립대학교_전자전기컴퓨터공학부"
    for i, (c1, c2) in enumerate(zip(department, expected)):
        if c1 != c2:
            print(f"Mismatch at position {i}: '{c1}' ({ord(c1)}) vs '{c2}' ({ord(c2)})")
    
    return content

def process_course_data(file_path: str) -> List[Dict[str, Any]]:
    """메인 처리 함수"""
    try:
        # JSON 파일 읽기
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 각 항목 처리
        processed_data = []
        for item in data:
            # 파일 경로에서 학과 정보 추출
            department = get_department_from_path(item['file_path'])
            print(f"Found department: {department}")
            
            if department:
                # 항목 복사 및 content 전처리
                processed_item = item.copy()
                if "content" in processed_item:
                    # content 길이 확인
                    print(f"Original content length: {len(processed_item['content'])}")
                    processed_content = preprocess_content(
                        processed_item["content"], 
                        department
                    )
                    processed_item["content"] = processed_content
                    print(f"Processed content length: {len(processed_content)}")
                    # 내용이 같은지 확인
                    if processed_item["content"] == item["content"]:
                        print("Warning: Content was not changed by preprocessing!")
                processed_data.append(processed_item)
        
        return processed_data
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return []

if __name__ == "__main__":
    root = "/Users/justin/Desktop/특강/BigData_Camp_LLM_2025/project/course/"
    file_path = root+"all_courses_info.json"
    
    # 전처리 메서드 목록 출력
    preprocessing_methods = {
        "서울시립대학교_통계학과": preprocess_uos_stat,
        "서울시립대학교_전자전기컴퓨터공학부": preprocess_uos_eece,
        "전북대학교_중어중문학과": preprocess_jbnu_chinese,
        "한동대학교_전산전자공학부": preprocess_handong_ce
    }
    print("\nAvailable preprocessing methods:")
    for key in preprocessing_methods.keys():
        print(f"- '{key}'")
    
    processed_data = process_course_data(file_path)
    
    # 결과 저장
    output_path =root+"processed_all_course_info.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)