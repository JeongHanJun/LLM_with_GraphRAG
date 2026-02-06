"""
R9 GraphRAG Processor
- Gemini API 사용
- 페이지 기반 Retrieval
- 상세 로깅 시스템
"""

import os
import json
import re
import csv
import logging
import traceback
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# pdfplumber/pdfminer 로그 완전 억제 (반드시 import 전에 설정)
for lib in ["pdfminer", "pdfminer.pdfpage", "pdfminer.pdfparser", "pdfminer.pdfdocument",
            "pdfminer.pdfinterp", "pdfminer.converter", "pdfminer.cmapdb", "pdfminer.psparser",
            "pdfplumber", "PIL"]:
    logging.getLogger(lib).setLevel(logging.ERROR)
    logging.getLogger(lib).propagate = False

import pdfplumber


def setup_logger(log_dir: str = "logs") -> logging.Logger:
    """로거 설정"""
    os.makedirs(log_dir, exist_ok=True)
    
    # 로그 파일명 (날짜_시간)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"r9_graphrag_{timestamp}.log")
    
    # 외부 라이브러리 로그 완전 억제
    for lib in ["pdfminer", "pdfminer.pdfpage", "pdfminer.pdfparser", "pdfminer.pdfdocument",
                "pdfminer.pdfinterp", "pdfminer.converter", "pdfminer.cmapdb", "pdfminer.psparser",
                "pdfplumber", "neo4j", "urllib3", "httpx", "httpcore", "google", "grpc", "PIL"]:
        lib_logger = logging.getLogger(lib)
        lib_logger.setLevel(logging.ERROR)
        lib_logger.propagate = False
    
    # 로거 생성
    logger = logging.getLogger("R9GraphRAG")
    logger.setLevel(logging.DEBUG)
    
    # 파일 핸들러 (상세 로그)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    
    # 콘솔 핸들러 (간단 로그 - INFO 이상만)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('[%(levelname)s] %(message)s')
    console_handler.setFormatter(console_format)
    
    # 기존 핸들러 제거 후 추가
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"로그 파일 생성: {log_file}")
    
    return logger


# 전역 로거
logger = setup_logger()


@dataclass
class DebugInfo:
    """디버깅 정보"""
    step: str
    detail: str
    
    def __str__(self):
        return f"[{self.step}] {self.detail}"


class R9GraphProcessor:
    def __init__(self):
        logger.info("=" * 50)
        logger.info("R9GraphProcessor 초기화 시작")
        
        # Gemini API 초기화 (Vertex AI 방식)
        try:
            # 환경 변수 확인
            credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            logger.debug(f"GOOGLE_APPLICATION_CREDENTIALS: {credentials_path}")
            
            if not credentials_path:
                raise ValueError("GOOGLE_APPLICATION_CREDENTIALS가 .env 파일에 설정되지 않았습니다.")
            
            if not os.path.exists(credentials_path):
                raise FileNotFoundError(f"인증 파일을 찾을 수 없습니다: {credentials_path}")
            
            # google-genai 임포트
            logger.debug("google.genai 모듈 임포트 시도...")
            from google import genai
            from google.genai import types
            self.genai = genai
            self.types = types
            logger.debug("google.genai 모듈 임포트 성공")
            
            # Vertex AI 클라이언트 초기화
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "gen-lang-client-0112109181")
            location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
            
            logger.debug(f"Vertex AI 초기화 - Project: {project_id}, Location: {location}")
            
            self.client = genai.Client(
                vertexai=True,
                project=project_id,
                location=location
            )
            self.model_id = "gemini-2.0-flash"
            logger.info(f"Gemini API 초기화 성공 (Vertex AI, 모델: {self.model_id})")
            
        except ImportError as e:
            logger.error(f"google-genai 모듈 임포트 실패: {e}")
            logger.error("pip install google-genai 실행 필요")
            raise
        except Exception as e:
            logger.error(f"Gemini API 초기화 실패: {e}")
            logger.error(traceback.format_exc())
            raise
        
        # Neo4j 연결 (선택적)
        self.driver = None
        self.graph_available = False
        
        try:
            self.uri = os.getenv("NEO4J_URI")
            self.user = os.getenv("NEO4J_USER")
            self.password = os.getenv("NEO4J_PASSWORD")
            
            logger.debug(f"Neo4j URI: {self.uri}")
            logger.debug(f"Neo4j User: {self.user}")
            
            if self.uri and self.user and self.password:
                from neo4j import GraphDatabase
                self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
                
                # 연결 테스트
                with self.driver.session() as session:
                    result = session.run("MATCH (n) RETURN count(n) as cnt")
                    count = result.single()["cnt"]
                    self.graph_available = count > 0
                    logger.info(f"Neo4j 연결 성공 (노드 수: {count})")
            else:
                logger.warning("Neo4j 환경변수 미설정, Graph DB 비활성화")
                
        except Exception as e:
            logger.warning(f"Neo4j 연결 실패 (무시됨): {e}")
            self.driver = None
        
        # 캐시
        self._page_cache: dict = {}
        self._page_keywords: dict = {}
        
        # UI 콜백
        self.debug_callback = None
        
        logger.info("R9GraphProcessor 초기화 완료")
        logger.info("=" * 50)

    def _log(self, step: str, detail: str):
        """디버그 로그 (UI + 파일)"""
        logger.info(f"{step}: {detail}")
        if self.debug_callback:
            try:
                self.debug_callback(DebugInfo(step, detail))
            except Exception as e:
                logger.error(f"debug_callback 오류: {e}")

    def _call_gemini(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.3) -> str:
        """Gemini API 호출"""
        logger.debug(f"Gemini API 호출 - max_tokens: {max_tokens}, temp: {temperature}")
        logger.debug(f"프롬프트 길이: {len(prompt)}자")
        
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[prompt],
                config=self.types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )
            
            result = response.text.strip()
            logger.debug(f"API 응답 길이: {len(result)}자")
            logger.debug(f"API 응답 미리보기: {result[:200]}...")
            
            return result
            
        except Exception as e:
            logger.error(f"Gemini API 호출 실패: {e}")
            logger.error(traceback.format_exc())
            raise

    # ============================================================
    # PDF 처리
    # ============================================================
    
    def load_pdf(self, pdf_path: str) -> dict:
        """PDF 페이지별 로드 및 인덱싱"""
        logger.info(f"PDF 로드 시작: {pdf_path}")
        
        if pdf_path in self._page_cache:
            logger.info("캐시에서 로드")
            self._log("PDF 로드", f"캐시에서 로드 ({len(self._page_cache[pdf_path])}페이지)")
            return self._page_cache[pdf_path]
        
        self._log("PDF 로드", "페이지별 텍스트 추출 시작...")
        
        try:
            page_texts = {}
            page_keywords = {}
            
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"총 페이지 수: {total_pages}")
                
                for page_num, page in enumerate(pdf.pages, start=1):
                    logger.debug(f"페이지 {page_num}/{total_pages} 처리 중...")
                    
                    page_text = ""
                    
                    # 텍스트 추출
                    text = page.extract_text()
                    if text:
                        page_text += text
                    
                    # 표 추출
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            for row in table:
                                if row:
                                    cleaned = [str(c) if c else "" for c in row]
                                    page_text += " | ".join(cleaned) + "\n"
                    
                    page_texts[page_num] = page_text
                    
                    # 키워드 추출
                    keywords = set(re.findall(r'[가-힣a-zA-Z0-9]{2,}', page_text.lower()))
                    page_keywords[page_num] = keywords
                    
                    logger.debug(f"페이지 {page_num}: {len(page_text)}자, 키워드 {len(keywords)}개")
            
            self._page_cache[pdf_path] = page_texts
            self._page_keywords[pdf_path] = page_keywords
            
            logger.info(f"PDF 로드 완료: {total_pages}페이지")
            self._log("PDF 로드", f"완료: {total_pages}페이지, 인덱싱 완료")
            
            return page_texts
            
        except Exception as e:
            logger.error(f"PDF 로드 실패: {e}")
            logger.error(traceback.format_exc())
            raise

    def search_relevant_pages(self, pdf_path: str, query: str, max_pages: int = 5) -> list:
        """질문과 관련된 페이지 검색"""
        logger.debug(f"페이지 검색 시작 - 쿼리: {query}")
        
        if pdf_path not in self._page_keywords:
            self.load_pdf(pdf_path)
        
        keywords = set(re.findall(r'[가-힣a-zA-Z0-9]{2,}', query.lower()))
        logger.debug(f"검색 키워드: {keywords}")
        self._log("페이지 검색", f"키워드: {keywords}")
        
        page_scores = []
        
        for page_num, page_kws in self._page_keywords[pdf_path].items():
            matches = keywords & page_kws
            if matches:
                score = len(matches)
                page_scores.append((page_num, score, matches))
        
        page_scores.sort(key=lambda x: x[1], reverse=True)
        selected = page_scores[:max_pages]
        
        if selected:
            pages_info = ", ".join([f"p{p[0]}({p[1]}점)" for p in selected])
            logger.info(f"관련 페이지 발견: {pages_info}")
            self._log("페이지 검색", f"관련 페이지: {pages_info}")
        else:
            logger.info("관련 페이지 없음, 첫 5페이지 사용")
            self._log("페이지 검색", "관련 페이지 없음, 첫 5페이지 사용")
            return list(range(1, min(6, len(self._page_cache.get(pdf_path, {})) + 1)))
        
        return [p[0] for p in selected]

    def get_pages_text(self, pdf_path: str, page_nums: list) -> str:
        """지정된 페이지들의 텍스트 반환"""
        if pdf_path not in self._page_cache:
            self.load_pdf(pdf_path)
        
        texts = []
        for pn in sorted(page_nums):
            if pn in self._page_cache[pdf_path]:
                texts.append(f"=== 페이지 {pn} ===\n{self._page_cache[pdf_path][pn]}")
        
        combined = "\n\n".join(texts)
        logger.debug(f"컨텍스트 준비: {len(page_nums)}페이지, {len(combined)}자")
        self._log("컨텍스트 준비", f"{len(page_nums)}페이지, {len(combined)}자")
        
        return combined

    # ============================================================
    # Graph DB
    # ============================================================
    
    def build_graph(self, pdf_path: str, mode: str = "general") -> dict:
        """PDF에서 지식 그래프 생성"""
        logger.info(f"그래프 생성 시작 - 모드: {mode}")
        
        if pdf_path not in self._page_cache:
            self.load_pdf(pdf_path)
        
        page_texts = self._page_cache[pdf_path]
        instruction = "주요 구성요소만" if mode == "general" else "상세 스펙 포함"
        
        all_nodes = []
        all_edges = []
        seen_ids = set()
        
        total = len(page_texts)
        
        for i, (page_num, text) in enumerate(page_texts.items()):
            if not text.strip() or len(text) < 100:
                continue
            
            logger.debug(f"페이지 {page_num}/{total} 처리 중...")
            self._log("그래프 생성", f"페이지 {page_num}/{total} 처리 중...")
            
            truncated = text[:3000] if len(text) > 3000 else text
            
            prompt = f"""기술 문서에서 지식 그래프 JSON 추출.
분석: {instruction}

규칙:
- 노드: 장치/기능/스펙/부품 (id는 영문_스네이크)
- 엣지: CONTAINS, HAS_FEATURE, HAS_SENSOR, RUNS, USES, CONNECTS_TO
- page 필드에 {page_num} 기록

JSON만 출력:
{{"nodes":[{{"id":"..","name":"..","label":"Type","desc":"..","page":{page_num}}}],"edges":[{{"from":"..","to":"..","rel":".."}}]}}

내용:
{truncated}"""

            try:
                result = self._call_gemini(prompt, max_tokens=800, temperature=0.1)
                
                match = re.search(r'\{[\s\S]*\}', result)
                if match:
                    data = json.loads(match.group())
                    
                    for node in data.get('nodes', []):
                        if node.get('id') and node['id'] not in seen_ids:
                            seen_ids.add(node['id'])
                            node['page'] = page_num
                            all_nodes.append(node)
                    
                    all_edges.extend(data.get('edges', []))
                    logger.debug(f"페이지 {page_num}: 노드 {len(data.get('nodes', []))}개 추출")
                    
            except Exception as e:
                logger.warning(f"페이지 {page_num} 처리 오류: {e}")
                self._log("그래프 생성", f"페이지 {page_num} 오류: {str(e)[:50]}")
                continue
        
        valid_edges = [e for e in all_edges if e.get('from') in seen_ids and e.get('to') in seen_ids]
        
        graph_data = {"nodes": all_nodes, "edges": valid_edges}
        logger.info(f"그래프 생성 완료: 노드 {len(all_nodes)}개, 엣지 {len(valid_edges)}개")
        self._log("그래프 생성", f"완료: 노드 {len(all_nodes)}개, 엣지 {len(valid_edges)}개")
        
        return graph_data

    def save_to_neo4j(self, graph_data: dict):
        """Neo4j에 저장"""
        logger.info("Neo4j 저장 시작")
        
        if not self.driver:
            raise ValueError("Neo4j 연결이 설정되지 않았습니다.")
        
        self._log("Neo4j 저장", "기존 데이터 삭제 중...")
        
        try:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
                
                logger.info(f"노드 {len(graph_data.get('nodes', []))}개 생성 중...")
                self._log("Neo4j 저장", f"노드 {len(graph_data.get('nodes', []))}개 생성 중...")
                
                for node in graph_data.get('nodes', []):
                    label = re.sub(r'[^a-zA-Z0-9_]', '', node.get('label', 'Entity'))
                    if not label or not label[0].isalpha():
                        label = "Entity"
                    
                    session.run(
                        f"CREATE (n:{label} $props)",
                        props={
                            "id": str(node.get('id', '')),
                            "name": str(node.get('name', '')),
                            "desc": str(node.get('desc', '')),
                            "label_type": str(node.get('label', '')),
                            "page": int(node.get('page', 0)) if node.get('page') else 0
                        }
                    )
                
                logger.info(f"엣지 {len(graph_data.get('edges', []))}개 생성 중...")
                self._log("Neo4j 저장", f"엣지 {len(graph_data.get('edges', []))}개 생성 중...")
                
                for edge in graph_data.get('edges', []):
                    rel = re.sub(r'[^a-zA-Z0-9_]', '', edge.get('rel', 'RELATED')).upper()
                    if not rel:
                        rel = "RELATED"
                    
                    session.run(
                        f"MATCH (a), (b) WHERE a.id = $f AND b.id = $t CREATE (a)-[:{rel}]->(b)",
                        f=str(edge.get('from', '')),
                        t=str(edge.get('to', ''))
                    )
            
            self.graph_available = True
            logger.info("Neo4j 저장 완료")
            self._log("Neo4j 저장", "완료!")
            
        except Exception as e:
            logger.error(f"Neo4j 저장 실패: {e}")
            logger.error(traceback.format_exc())
            raise

    def save_to_csv(self, graph_data: dict, output_dir: str = "graph_db") -> dict:
        """CSV로 저장"""
        logger.info(f"CSV 저장 시작: {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        nodes_file = os.path.join(output_dir, f"nodes_{ts}.csv")
        with open(nodes_file, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=['id', 'name', 'label', 'desc', 'page'])
            w.writeheader()
            for n in graph_data.get('nodes', []):
                w.writerow({k: n.get(k, '') for k in ['id', 'name', 'label', 'desc', 'page']})
        
        edges_file = os.path.join(output_dir, f"edges_{ts}.csv")
        with open(edges_file, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=['from', 'to', 'rel'])
            w.writeheader()
            for e in graph_data.get('edges', []):
                w.writerow({'from': e.get('from', ''), 'to': e.get('to', ''), 'rel': e.get('rel', '')})
        
        json_file = os.path.join(output_dir, f"graph_{ts}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"CSV 저장 완료: {output_dir}/")
        self._log("CSV 저장", f"저장 완료: {output_dir}/")
        
        return {'nodes_csv': nodes_file, 'edges_csv': edges_file, 'json_file': json_file}

    def query_neo4j_pages(self, query: str, limit: int = 5) -> tuple:
        """Neo4j에서 관련 페이지 검색"""
        logger.debug(f"Neo4j 검색: {query}")
        
        if not self.driver or not self.graph_available:
            logger.debug("Neo4j 사용 불가")
            return [], ""
        
        keywords = re.findall(r'[가-힣a-zA-Z0-9]{2,}', query)
        pages = set()
        context_parts = []
        
        self._log("Neo4j 검색", f"키워드: {keywords[:5]}")
        
        try:
            with self.driver.session() as session:
                for kw in keywords[:5]:
                    if len(kw) < 2:
                        continue
                    
                    result = session.run(
                        """
                        MATCH (n)
                        WHERE toLower(n.name) CONTAINS toLower($kw) OR toLower(n.desc) CONTAINS toLower($kw)
                        OPTIONAL MATCH (n)-[r]-(m)
                        RETURN n.name as name, n.page as page, n.desc as desc,
                               collect(DISTINCT {rel: type(r), target: m.name})[..3] as rels
                        LIMIT $limit
                        """,
                        kw=kw, limit=limit
                    )
                    
                    for rec in result:
                        if rec['page']:
                            pages.add(int(rec['page']))
                        
                        ctx = f"- {rec['name']}: {rec['desc'][:50] if rec['desc'] else ''}"
                        if rec['rels']:
                            rels_str = ", ".join([f"{r['rel']}→{r['target']}" for r in rec['rels'] if r['target']])
                            if rels_str:
                                ctx += f" [{rels_str}]"
                        
                        if ctx not in context_parts:
                            context_parts.append(ctx)
            
            pages_list = sorted(list(pages))[:limit]
            context = "\n".join(context_parts[:10])
            
            logger.info(f"Neo4j 검색 결과: 페이지 {pages_list}, 컨텍스트 {len(context_parts)}개")
            self._log("Neo4j 검색", f"관련 페이지: {pages_list}")
            
            return pages_list, context
            
        except Exception as e:
            logger.error(f"Neo4j 검색 오류: {e}")
            return [], ""

    # ============================================================
    # 챗봇
    # ============================================================
    
    def ask(self, question: str, pdf_path: str) -> str:
        """질문에 답변"""
        logger.info("=" * 50)
        logger.info(f"질문 처리 시작: {question[:50]}...")
        
        self._log("질문 처리", f"질문: {question[:50]}...")
        
        try:
            # 1. PDF 로드 확인
            if pdf_path not in self._page_cache:
                logger.info("PDF 캐시 없음, 로드 시작")
                self.load_pdf(pdf_path)
            
            # 2. 관련 페이지 찾기
            graph_context = ""
            
            if self.graph_available:
                logger.info("Graph DB 활용 모드")
                self._log("검색 방식", "Graph DB + 페이지 검색")
                neo4j_pages, graph_context = self.query_neo4j_pages(question, limit=5)
                
                if neo4j_pages:
                    relevant_pages = neo4j_pages
                    logger.info(f"Neo4j에서 페이지 발견: {relevant_pages}")
                else:
                    logger.info("Neo4j에서 페이지 못 찾음, 키워드 검색 수행")
                    relevant_pages = self.search_relevant_pages(pdf_path, question, max_pages=4)
            else:
                logger.info("키워드 검색 모드 (Graph DB 없음)")
                self._log("검색 방식", "페이지 키워드 검색 (Graph DB 없음)")
                relevant_pages = self.search_relevant_pages(pdf_path, question, max_pages=5)
            
            logger.info(f"최종 관련 페이지: {relevant_pages}")
            
            # 3. 관련 페이지 텍스트 추출
            page_text = self.get_pages_text(pdf_path, relevant_pages)
            
            # 4. 프롬프트 구성
            logger.info("프롬프트 구성 중...")
            self._log("API 호출", "Gemini API 요청 중...")
            
            prompt = f"""사양서 기반 Q&A. 아래 정보로 질문에 답변하세요.

{"[그래프 정보]" + chr(10) + graph_context + chr(10) if graph_context else ""}
[사양서 내용 (페이지 {', '.join(map(str, relevant_pages))})]
{page_text[:6000]}

[규칙]
1. 사양서 내용 기반으로만 답변
2. 출처 페이지 명시
3. 정보 없으면 "해당 정보가 문서에 없습니다" 답변
4. 한국어로 간결하게

[질문] {question}"""

            logger.debug(f"프롬프트 길이: {len(prompt)}자")
            
            # 5. API 호출
            answer = self._call_gemini(prompt, max_tokens=800, temperature=0.3)
            
            logger.info(f"답변 생성 완료: {len(answer)}자")
            self._log("답변 완료", f"응답 길이: {len(answer)}자")
            
            return answer
            
        except Exception as e:
            logger.error(f"질문 처리 실패: {e}")
            logger.error(traceback.format_exc())
            raise

    def close(self):
        """리소스 정리"""
        logger.info("리소스 정리 시작")
        
        if self.driver:
            try:
                self.driver.close()
                logger.info("Neo4j 연결 종료")
            except:
                pass
        
        self._page_cache.clear()
        self._page_keywords.clear()
        
        logger.info("리소스 정리 완료")
