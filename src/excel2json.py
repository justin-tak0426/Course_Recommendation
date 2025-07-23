import pandas as pd
import json
import os
from pathlib import Path
from tqdm import tqdm

def process_major_directory(major_dir):
    try:
        # 결과를 저장할 리스트
        all_courses = []
        
        # 디렉토리의 모든 xlsx 파일 찾기
        xlsx_files = list(Path(major_dir).glob('*.xlsx'))
        
        if not xlsx_files:
            print(f"Warning: No xlsx files found in {major_dir}")
            return []

        # 각 파일 처리
        for file_path in tqdm(xlsx_files):
            try:
                # Excel 파일 읽기 - 모든 컬럼을 문자열로 읽기
                df = pd.read_excel(file_path, dtype=str)
                
                # DataFrame을 JSON 형식의 리스트로 변환
                for _, row in tqdm(df.iterrows()):
                    course_dict = {
                        'idx': str(row['idx']).strip(),
                        'year': str(row['year']).strip(),
                        'sem': str(row['sem']).strip(),
                        'course': str(row['course']).strip(),
                        'credit': str(row['credit']).strip(),
                        'req': str(row['req']).strip(),
                        'source_file': str(file_path.name).strip()
                    }
                    all_courses.append(course_dict)
                
                print(f"Successfully processed: {file_path.name}")
                
            except Exception as e:
                print(f"Error processing {file_path.name}: {str(e)}")
                continue
        
        
        # JSON 파일로 저장
        output_path = os.path.join(major_dir, 'all_courses.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_courses, f, ensure_ascii=False, indent=4)
        
        print(f"\nJSON file saved at: {output_path}")
        return all_courses

    except Exception as e:
        print(f"Error processing directory: {str(e)}")
        return []

# 사용 예시
project_dir = "/Users/justin/Desktop/특강/BigData_Camp_LLM_2025/project"
major_dir = project_dir + "/major"

result = process_major_directory(major_dir)

if result:
    print(f"\nTotal courses processed: {len(result)}")
    print("\nFirst few entries:")
    print(json.dumps(result[:3], ensure_ascii=False, indent=4))