# R9 GraphRAG - Gemini 유료 API (Vertex AI) 연동 해결

## 문제 분석

### 1. API 방식 불일치
- **현재 상태**: processor.py가 일반 Gemini API Key 방식 사용
- **필요한 설정**: Vertex AI 방식 (api_test.py에서 사용 중)
- **증상**: API 호출은 성공하지만 이후 프로그램 종료

### 2. 로그 분석 결과
```
[성공] PDF 로드 완료: 45페이지
[성공] 질문 처리 시작
[성공] Gemini API 응답 수신 (204자)
[문제] on_ask_finished 콜백 미호출 → UI 업데이트 실패 → 프로그램 크래시
```

### 3. Neo4j 연결 문제 (Optional)
```
WARNING: Neo4j 연결 실패 (무시됨): Unable to retrieve routing information
```
- URI 프로토콜 문제: `neo4j://` → `bolt://`로 변경 필요

---

## 해결 방법

### 1단계: .env 파일 업데이트

기존 .env 파일을 다음과 같이 수정

```env
# Gemini API 설정 (Vertex AI 사용)
GOOGLE_APPLICATION_CREDENTIALS=D:\RAG\vertex-key.json
GOOGLE_CLOUD_PROJECT={내 프로젝트 id}
GOOGLE_CLOUD_LOCATION=us-central1

# Neo4j 설정 (URI 프로토콜 변경)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD={내 neo4j 비밀번호}
```

**변경 사항:**
- `GEMINI_API_KEY` 제거 (Vertex AI는 API Key 사용 안 함)
- `GOOGLE_APPLICATION_CREDENTIALS` 추가 (이미 있음)
- `GOOGLE_CLOUD_PROJECT` 추가
- `GOOGLE_CLOUD_LOCATION` 추가
- `NEO4J_URI` 변경: `neo4j://` → `bolt://`

### 2단계: processor.py 교체

수정된 processor.py 파일을 프로젝트 폴더에 복사하세요.

**주요 변경 사항:**
```python
# 기존 코드 (일반 API Key 방식)
api_key = os.getenv("GEMINI_API_KEY")
self.client = genai.Client(api_key=api_key)

# 새 코드 (Vertex AI 방식)
credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

self.client = genai.Client(
    vertexai=True,
    project=project_id,
    location=location
)
```

### 3단계: vertex-key.json 확인

현재 파일 위치가 정확한지 확인:
```
D:\RAG\vertex-key.json
```

파일이 존재하고 .env의 경로와 일치하는지 확인

---

## 설정 확인

### 1. Vertex AI 설정 확인

### 2. Neo4j 연결 확인

Neo4j Desktop을 실행하고 데이터베이스가 시작되었는지 확인:
- Neo4j Desktop → 데이터베이스 → "Start" 버튼 클릭
- 상태가 "Running"인지 확인
- 비밀번호가 .env와 일치하는지 확인

---

## 실행 순서

1. Neo4j Desktop 실행 및 데이터베이스 시작
2. .env 파일 업데이트
3. processor.py 교체
4. 프로그램 실행
   ```bash
   python main.py
   ```

---

## 예상 로그 (성공 시)

```
[INFO] GOOGLE_APPLICATION_CREDENTIALS: D:\RAG\vertex-key.json
[INFO] Vertex AI 초기화 - Project: {내 프로젝트 id}, Location: us-central1
[INFO] Gemini API 초기화 성공 (Vertex AI, 모델: gemini-2.0-flash)
[INFO] Neo4j 연결 성공 (노드 수: 0)
[INFO] R9GraphProcessor 초기화 완료
```

---

## 문제 해결 팁

### 오류: "GOOGLE_APPLICATION_CREDENTIALS가 설정되지 않았습니다"
→ .env 파일에 `GOOGLE_APPLICATION_CREDENTIALS` 추가 확인

### 오류: "인증 파일을 찾을 수 없습니다"
→ vertex-key.json 파일 경로 확인
→ 절대 경로 사용 권장

### 오류: "Unable to retrieve routing information"
→ NEO4J_URI를 `bolt://localhost:7687`로 변경
→ Neo4j Desktop에서 데이터베이스 실행 확인

### 프로그램이 여전히 종료됨
→ logs/ 폴더의 최신 로그 파일 확인
→ 오류 메시지를 정확히 확인

---

## 유료 API 설정 확인 체크리스트

- [✓] Google Cloud Console → 결제 활성화
- [✓] 프로젝트: hanjun-R9-RAG ({내 프로젝트 id})
- [✓] IAM 서비스 계정: {내 IAM 서비스 계정}
- [✓] 서비스 계정 키 다운로드: vertex-key.json
- [✓] Vertex AI API 활성화 확인
  - Google Cloud Console → API 및 서비스 → Vertex AI API → "사용"

### Vertex AI API 활성화 방법
1. Google Cloud Console 접속
2. 프로젝트 선택: hanjun-R9-RAG
3. 메뉴 → "API 및 서비스" → "라이브러리"
4. "Vertex AI API" 검색
5. "사용 설정" 클릭

---

## 참고사항

### Vertex AI vs 일반 Gemini API

| 구분 | Vertex AI | 일반 Gemini API |
|------|-----------|-----------------|
| 인증 방식 | 서비스 계정 (JSON 키) | API Key |
| 결제 | Google Cloud 결제 계정 | 직접 결제 |
| 초기화 | `vertexai=True` | `api_key=key` |
| 장점 | 엔터프라이즈급, IAM 통합 | 간단한 설정 |

귀하의 설정은 **Vertex AI 방식**이므로 반드시 위 가이드대로 수정해야 합니다.

---

## 추가 지원

문제가 지속되면 다음 정보를 제공해주세요:
1. logs/ 폴더의 최신 로그 파일 2개
2. .env 파일 내용 (비밀번호 제외)
3. 오류 메시지 스크린샷
