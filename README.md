# LLM_with_GraphRAG
## 프로젝트 개요
- **Vertex AI Gemini 2.0 Flash** 모델을 활용하여 PDF 명세서의 내용을 분석하고, Graph 데이터와 연동하여 정확한 답변을 제공하는 **GraphRAG 시스템**

- PyQt6 기반의 GUI를 통해 간편하게 문서와 대화할 수 있음.

## 주요 기능들
- **Vertex AI Integration**: `google-genai` SDK를 사용한 최신 Gemini 2.0 Flash 모델 연동.
- **Smart Retrieval**: PDF 문서에서 질문과 가장 관련 있는 페이지를 지능적으로 추출.
- **Advanced UI**: PyQt6 기반의 사용자 친화적인 인터페이스 및 실시간 시스템 로깅.
- **Graph Context Support**: 외부 지식 그래프(Graph) 컨텍스트를 활용한 고도화된 답변 생성.

## 기술 스택
- **Language**: Python 3.10+
- **LLM**: Google Vertex AI (Gemini 2.0 Flash)
- **GUI**: PyQt6
- **PDF Processing**: pdfplumber
- **DB**: GrpahDB - Neo4j

## 선행 필요 내용
이 시스템을 실행하려면 Google Cloud 프로젝트와 서비스 계정 키가 필요
1. Vertex AI API 활성화
2. 서비스 계정 생성 및 `Vertex AI User` 권한 부여
3. JSON 키 파일 다운로드

## Installation & Setup
1. git clone

2. .env 파일 생성하고 아래 내용 입력
```
GOOGLE_APPLICATION_CREDENTIALS="{vertex-key.json 경로}"
GOOGLE_CLOUD_PROJECT="{구글 클라우드 프로젝트 id}"
GOOGLE_CLOUD_LOCATION="us-central1"
```

3. 라이브러리 설치
```
$ pip install -r requirements.txt
```

4. 실행
```
$ python main.py
```

## 구조
- main.py: PyQt6 GUI 어플리케이션 메인 로직

- processor.py: Vertex AI 연동 및 PDF/RAG 처리 엔진

- logs/: 실행 로그 저장 디렉토리