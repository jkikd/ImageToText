import easyocr
import cv2
import numpy as np
import difflib
import re
import itertools

# 이미지 파일 경로
IMAGE_PATH = 'sample.jpg'  # 추후 실제 이미지 파일명으로 변경

# EasyOCR Reader 생성 (한글, 영어 지원)
reader = easyocr.Reader(['ko', 'en'])

# 이미지 읽기
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {IMAGE_PATH}")

# 1. 그레이스케일 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. 이진화
thresh = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

# 3. 가로/세로선 검출
horizontal = thresh.copy()
vertical = thresh.copy()

# 가로선
cols = horizontal.shape[1]
horizontal_size = cols // 20
horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
horizontal = cv2.erode(horizontal, horizontal_structure)
horizontal = cv2.dilate(horizontal, horizontal_structure)

# 세로선
rows = vertical.shape[0]
vertical_size = rows // 20
vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
vertical = cv2.erode(vertical, vertical_structure)
vertical = cv2.dilate(vertical, vertical_structure)

# 4. 표 영역 찾기 (가로/세로선 합치기)
table_mask = cv2.add(horizontal, vertical)
contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 가장 큰 윤곽선(표 전체) 추출
if contours:
    table_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(table_contour)
    table_roi = image[y:y+h, x:x+w]
else:
    print('표 영역을 찾지 못했습니다. 전체 이미지에서 추출합니다.')
    table_roi = image

# OCR 수행 (표 영역만)
results = reader.readtext(table_roi)

# 자주 사용하는 한글 단어 리스트
COMMON_KOREAN_WORDS = [
    '빅웨이브', '초급', '중급', '상급', '나이트', '서핑', '베이', '자유', '파도', '금액', '일시','세션','펀딩', '명', '베이자유서핑', '나이트 서핑', '슈트/보드 포함', '슈트', '보드', '포함', '초급세션','중급세션','상급세션' , '나이트 서핑 운영시'
      ]

def clean_korean(text):
    # 한글, 영어, 숫자만 남기고 나머지 제거
    return re.sub(r'[^\uAC00-\uD7A3a-zA-Z0-9]', '', text)

def normalize_korean(text, common_words, cutoff=0.7):
    # 1. 공백 포함 원본 우선 매칭
    cleaned = re.sub(r'[^\uAC00-\uD7A3a-zA-Z0-9 ]', '', text)  # 공백은 남김
    if any('\uac00' <= ch <= '\ud7a3' for ch in cleaned):
        # 공백 포함 매칭
        matches = difflib.get_close_matches(cleaned, common_words, n=1, cutoff=cutoff)
        if matches:
            return matches[0]
        # 공백 제거 후 매칭
        cleaned_no_space = cleaned.replace(' ', '')
        matches = difflib.get_close_matches(cleaned_no_space, [w.replace(' ', '') for w in common_words], n=1, cutoff=cutoff)
        if matches:
            # 공백 위치를 원본에 맞춰 복원
            idx = [w.replace(' ', '') for w in common_words].index(matches[0])
            ref = common_words[idx]
            # 원본에 공백이 있으면 ref에도 같은 위치에 공백 삽입
            if ' ' in text:
                # 단순히 ref에서 원본 공백 위치에 공백 삽입
                ref_no_space = ref.replace(' ', '')
                result = list(ref_no_space)
                for i, c in enumerate(text):
                    if c == ' ':
                        result.insert(i, ' ')
                return ''.join(result)
            else:
                return ref
        return cleaned if cleaned else text
    # 한글이 없으면 원본 그대로 반환
    return text

def extract_row_info(row_cells):
    # 각 셀에서 정보 추출
    date_pattern = r'\d{1,2}/\d{1,2} ?\([A-Z]{3}\)'  # 예: 7/6 (SUN)
    m_pattern = r'M\d+'
    price_pattern = r'\d{1,3}(,\d{3})* ?KRW'
    kor_session_keywords = ['초급세션', '중급세션', '상급세션']
    eng_session_keywords = ['BEGINNER', 'INTERMEDIATE', 'ADVANCED']
    # 결과 변수
    date = ''
    kor_session = ''
    eng_session = ''
    m_numbers = []
    price = ''
    for cell in row_cells:
        # 날짜
        if re.search(date_pattern, cell):
            date = re.search(date_pattern, cell).group()
        # 한글 세션명
        for k in kor_session_keywords:
            if k in cell:
                kor_session = k
        # 영어 세션명
        for e in eng_session_keywords:
            if e in cell:
                eng_session = e
        # M번호
        m_found = re.findall(m_pattern, cell)
        if m_found:
            m_numbers.extend(m_found)
        # 금액
        if re.search(price_pattern, cell):
            price = re.search(price_pattern, cell).group()
    # M번호 중복 제거, 쉼표로 연결
    m_numbers = ', '.join(sorted(set(m_numbers), key=lambda x: int(x[1:]))) if m_numbers else ''
    return [date, kor_session, eng_session, m_numbers, price]

def split_kor_eng(text):
    kor = ''.join([c for c in text if '\uac00' <= c <= '\ud7a3'])
    eng = ''.join([c for c in text if ('a' <= c.lower() <= 'z')])
    num = ''.join([c for c in text if c.isdigit()])
    other = ''.join([c for c in text if not (('\uac00' <= c <= '\ud7a3') or ('a' <= c.lower() <= 'z') or c.isdigit())])
    if kor and eng:
        return kor, eng + num + other
    else:
        return text, None

def extract_header_indices(header_row):
    # 헤더에서 각 컬럼의 인덱스 추출 (유사도 허용)
    header_map = {'일시': -1, '파도': -1, '금액': -1}
    header_keywords = {'일시': ['일시', 'DATE', '날짜'], '파도': ['파도', 'SESSION', '세션'], '금액': ['금액', 'PRICE', 'KRW']}
    for idx, cell in enumerate(header_row):
        for key, keywords in header_keywords.items():
            for kw in keywords:
                if kw in cell:
                    header_map[key] = idx
    return header_map

def extract_table_row(row_cells, header_map):
    # 각 행에서 헤더 인덱스에 맞는 정보 추출
    # 일시: 날짜+요일
    date_pattern = r'\d{1,2}/\d{1,2} ?\([A-Z]{3}\)'
    # 파도: 세션명+파도크기
    session_keywords = ['빅웨이브', '초급세션', '중급세션', '상급세션']
    size_pattern = r'(T\d+|M\d+|B\d+)'  # T1, M2, B1 등
    # 금액
    price_pattern = r'\d{1,3}(,\d{3})* ?KRW'
    # 결과
    date = ''
    wave = []
    price = ''
    # 일시
    if header_map['일시'] != -1 and header_map['일시'] < len(row_cells):
        cell = row_cells[header_map['일시']]
        m = re.search(date_pattern, cell)
        if m:
            date = m.group()
    # 파도
    if header_map['파도'] != -1 and header_map['파도'] < len(row_cells):
        cell = row_cells[header_map['파도']]
        for k in session_keywords:
            if k in cell:
                wave.append(k)
        sizes = re.findall(size_pattern, cell)
        wave.extend(sizes)
    # 금액
    if header_map['금액'] != -1 and header_map['금액'] < len(row_cells):
        cell = row_cells[header_map['금액']]
        m = re.search(price_pattern, cell)
        if m:
            price = m.group()
    return [date, ', '.join(wave), price]

def extract_table_row_v2(row_cells):
    # 일시: 날짜+요일
    date_pattern = r'\d{1,2}/\d{1,2} ?\([A-Z]{3}\)'
    # 파도: 세션명(한글/영어) 및 파도 크기
    session_keywords = ['빅웨이브', '초급세션', '중급세션', '상급세션', '베이자유서핑', '나이트 서핑']
    eng_session_keywords = ['BIG WAVE', 'BEGINNER', 'INTERMEDIATE', 'ADVANCED', 'BAY SURFING', 'NIGHT SURFING']
    size_pattern = r'(T\d+|M\d+|B\d+)'
    price_pattern = r'\d{1,3}(,\d{3})* ?KRW'
    date = ''
    kor_wave = []
    eng_wave = []
    size_wave = []
    price = ''
    for cell in row_cells:
        # 일시
        if not date:
            m = re.search(date_pattern, cell)
            if m:
                date = m.group()
                continue
        # 금액
        if not price:
            m = re.search(price_pattern, cell)
            if m:
                price = m.group()
                continue
        # 파도 한글/영어/크기 분리
        for k in session_keywords:
            if k in cell and k not in kor_wave:
                kor_wave.append(k)
        for e in eng_session_keywords:
            if e in cell and e not in eng_wave:
                eng_wave.append(e)
        sizes = re.findall(size_pattern, cell)
        for s in sizes:
            if s not in size_wave:
                size_wave.append(s)
    # 한글/영어/크기 순서로 '/'로 연결
    wave = kor_wave + eng_wave + size_wave
    return [date, '/'.join(wave), price]

# 결과 출력 (한글 유사어 교정)
for bbox, text, conf in results:
    norm_text = normalize_korean(text, COMMON_KOREAN_WORDS)
    if norm_text != text:
        print(f'OCR: {text} -> 교정: {norm_text}, Confidence: {conf:.2f}')
    else:
        print(f'Text: {norm_text}, Confidence: {conf:.2f}')

# 결과를 행/열로 정렬하여 파일로 저장
output_lines = []
if results:
    # 각 결과: (bbox, text, conf)
    # bbox: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    # y좌표(상단) 기준으로 행 그룹화
    # 1. 각 텍스트의 중심 y좌표 계산
    items = []
    for bbox, text, conf in results:
        norm_text = normalize_korean(text, COMMON_KOREAN_WORDS)
        kor, eng = split_kor_eng(norm_text)
        x_coords = [pt[0] for pt in bbox]
        y_coords = [pt[1] for pt in bbox]
        cx = int(sum(x_coords) / 4)
        cy = int(sum(y_coords) / 4)
        items.append({'kor': kor, 'eng': eng, 'cx': cx, 'cy': cy, 'bbox': bbox})
    # 2. y좌표 기준으로 행 그룹화 (임계값: 20px)
    items.sort(key=lambda x: x['cy'])
    rows = []
    row = []
    last_cy = None
    threshold = 20  # 행 구분 y좌표 임계값
    for item in items:
        if last_cy is None or abs(item['cy'] - last_cy) < threshold:
            row.append(item)
            last_cy = item['cy'] if last_cy is None else (last_cy + item['cy']) // 2
        else:
            rows.append(row)
            row = [item]
            last_cy = item['cy']
    if row:
        rows.append(row)
    # 3. 각 행에서 x좌표 기준으로 정렬 후 '||'로 join
    header_map = None
    header_written = False
    for i, row in enumerate(rows):
        row.sort(key=lambda x: x['cx'])
        cell_texts = []
        for cell in row:
            if cell.get('eng'):
                cell_texts.append(cell['kor'] + ' ' + cell['eng'])
            else:
                cell_texts.append(cell['kor'])
        # 첫 행은 무조건 헤더로 간주
        if not header_written:
            output_lines.append('일시 DATE || 파도 SESSION || 금액 PRICE')
            header_written = True
            continue
        info = extract_table_row_v2(cell_texts)
        if all(info):
            output_lines.append(' || '.join(info))
else:
    output_lines.append('No text found.')

# 파일로 저장 (실행 시마다 초기화)
with open('output.txt', 'w', encoding='utf-8') as f:
    for line in output_lines:
        f.write(line + '\n')

# 콘솔에도 출력
for line in output_lines:
    print(line)
