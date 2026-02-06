"""
R9 GraphRAG Admin Tool
- 상세 로깅 시스템
- 전역 예외 처리 (크래시 방지)
"""

import sys
import os
import logging
import traceback
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

from dotenv import load_dotenv

load_dotenv()

# 로그 디렉토리 생성
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 외부 라이브러리 로그 완전 억제 (import 전에 설정)
for lib in ["pdfminer", "pdfminer.pdfpage", "pdfminer.pdfparser", "pdfminer.pdfdocument",
            "pdfminer.pdfinterp", "pdfminer.converter", "pdfminer.cmapdb", "pdfminer.psparser",
            "pdfplumber", "neo4j", "urllib3", "httpx", "httpcore", "google", "grpc", "PIL"]:
    lib_logger = logging.getLogger(lib)
    lib_logger.setLevel(logging.ERROR)
    lib_logger.propagate = False

# 로거 설정
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"r9_main_{timestamp}.log")

# 루트 로거 설정 변경 (DEBUG 로그는 파일에만, 콘솔은 INFO만)
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(name)-15s | %(funcName)-20s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # 콘솔은 INFO 이상만
console_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[file_handler, console_handler]
)

logger = logging.getLogger("R9Main")
logger.info(f"로그 파일: {log_file}")


def exception_hook(exc_type, exc_value, exc_tb):
    """전역 예외 처리"""
    logger.critical("=" * 60)
    logger.critical("처리되지 않은 예외 발생!")
    logger.critical("=" * 60)
    
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
    for line in tb_lines:
        logger.critical(line.rstrip())
    
    logger.critical("=" * 60)


# 전역 예외 핸들러 등록
sys.excepthook = exception_hook


# PySide6 임포트
try:
    logger.info("PySide6 임포트 시작...")
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QPushButton, QVBoxLayout,
        QHBoxLayout, QWidget, QFileDialog, QTextEdit,
        QLabel, QRadioButton, QMessageBox, QGroupBox, QSplitter
    )
    from PySide6.QtCore import Qt, QThread, Signal
    from PySide6.QtGui import QTextCursor
    logger.info("PySide6 임포트 성공")
except ImportError as e:
    logger.critical(f"PySide6 임포트 실패: {e}")
    raise


# processor 임포트
try:
    logger.info("processor 모듈 임포트 시작...")
    from processor import R9GraphProcessor
    logger.info("processor 모듈 임포트 성공")
except ImportError as e:
    logger.critical(f"processor 임포트 실패: {e}")
    logger.critical(traceback.format_exc())
    raise


# pyvis 임포트
try:
    logger.info("pyvis 임포트 시작...")
    from pyvis.network import Network
    logger.info("pyvis 임포트 성공")
except ImportError as e:
    logger.warning(f"pyvis 임포트 실패 (그래프 시각화 불가): {e}")
    Network = None


class ChatInput(QTextEdit):
    """Enter로 전송, Shift+Enter로 줄바꿈"""
    enter_pressed = Signal()
    
    def keyPressEvent(self, event):
        try:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                if event.modifiers() == Qt.ShiftModifier:
                    super().keyPressEvent(event)
                else:
                    self.enter_pressed.emit()
            else:
                super().keyPressEvent(event)
        except Exception as e:
            logger.error(f"ChatInput keyPressEvent 오류: {e}")


class BuildWorker(QThread):
    """Graph DB 구축 백그라운드 스레드"""
    finished = Signal(dict, dict)
    error = Signal(str)
    progress = Signal(str)
    
    def __init__(self, processor, pdf_path, mode):
        super().__init__()
        self.processor = processor
        self.pdf_path = pdf_path
        self.mode = mode
        logger.info(f"BuildWorker 생성 - 모드: {mode}")
    
    def run(self):
        logger.info("BuildWorker 실행 시작")
        try:
            self.progress.emit("그래프 생성 중...")
            data = self.processor.build_graph(self.pdf_path, self.mode)
            
            self.progress.emit("Neo4j 저장 중...")
            self.processor.save_to_neo4j(data)
            
            self.progress.emit("CSV 저장 중...")
            files = self.processor.save_to_csv(data)
            
            logger.info("BuildWorker 완료")
            self.finished.emit(data, files)
            
        except Exception as e:
            logger.error(f"BuildWorker 오류: {e}")
            logger.error(traceback.format_exc())
            self.error.emit(str(e))


class AskWorker(QThread):
    """챗봇 응답 백그라운드 스레드"""
    finished = Signal(str)
    error = Signal(str)
    
    def __init__(self, processor, question, pdf_path):
        super().__init__()
        self.processor = processor
        self.question = question
        self.pdf_path = pdf_path
        logger.info(f"AskWorker 생성 - 질문: {question[:30]}...")
    
    def run(self):
        logger.info("AskWorker 실행 시작")
        try:
            answer = self.processor.ask(self.question, self.pdf_path)
            logger.info(f"AskWorker 완료 - 답변 길이: {len(answer)}자")
            self.finished.emit(answer)
            
        except Exception as e:
            logger.error(f"AskWorker 오류: {e}")
            logger.error(traceback.format_exc())
            self.error.emit(f"오류 발생: {str(e)}\n\n상세 로그는 logs/ 폴더를 확인하세요.")


class R9App(QMainWindow):
    def __init__(self):
        logger.info("=" * 60)
        logger.info("R9App 초기화 시작")
        
        super().__init__()
        self.setWindowTitle("R9 GraphRAG Admin Tool (Gemini)")
        self.resize(1200, 850)
        
        self.processor = None
        self.pdf_path = ""
        self.current_data = None
        self.worker = None
        self.ask_worker = None
        
        # Processor 초기화
        try:
            logger.info("R9GraphProcessor 초기화 중...")
            self.processor = R9GraphProcessor()
            self.processor.debug_callback = self.on_debug
            logger.info("R9GraphProcessor 초기화 성공")
        except Exception as e:
            logger.critical(f"Processor 초기화 실패: {e}")
            logger.critical(traceback.format_exc())
            QMessageBox.critical(None, "초기화 오류", 
                f"Processor 초기화 실패:\n{str(e)}\n\n로그 파일: {log_file}")
            raise
        
        self.init_ui()
        self.update_status()
        
        logger.info("R9App 초기화 완료")
        logger.info("=" * 60)

    def init_ui(self):
        logger.debug("UI 초기화 시작")
        
        main_layout = QHBoxLayout()
        
        # === 좌측 패널 ===
        left_widget = QWidget()
        left = QVBoxLayout(left_widget)
        left.setSpacing(10)
        
        # PDF 선택
        pdf_group = QGroupBox("1. PDF 파일")
        pdf_layout = QVBoxLayout()
        
        self.btn_select = QPushButton("PDF 파일 선택")
        self.btn_select.clicked.connect(self.select_file)
        pdf_layout.addWidget(self.btn_select)
        
        self.lbl_file = QLabel("파일을 선택하세요.")
        self.lbl_file.setWordWrap(True)
        self.lbl_file.setStyleSheet("color: #666; padding: 5px;")
        pdf_layout.addWidget(self.lbl_file)
        
        pdf_group.setLayout(pdf_layout)
        left.addWidget(pdf_group)
        
        # Graph DB (선택적)
        graph_group = QGroupBox("2. Graph DB (선택)")
        graph_layout = QVBoxLayout()
        
        self.radio_gen = QRadioButton("General (주요 구성요소)")
        self.radio_det = QRadioButton("Detailed (상세 스펙)")
        self.radio_gen.setChecked(True)
        graph_layout.addWidget(self.radio_gen)
        graph_layout.addWidget(self.radio_det)
        
        self.btn_build = QPushButton("Graph DB 구축")
        self.btn_build.clicked.connect(self.build_graph)
        self.btn_build.setEnabled(False)
        graph_layout.addWidget(self.btn_build)
        
        self.btn_view = QPushButton("지식 그래프 보기")
        self.btn_view.clicked.connect(self.view_graph)
        self.btn_view.setEnabled(False)
        graph_layout.addWidget(self.btn_view)
        
        graph_group.setLayout(graph_layout)
        left.addWidget(graph_group)
        
        # 상태
        status_group = QGroupBox("상태")
        status_layout = QVBoxLayout()
        
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("font-size: 11px;")
        status_layout.addWidget(self.lbl_status)
        
        # 로그 파일 위치 표시
        log_label = QLabel(f"로그: {log_file}")
        log_label.setWordWrap(True)
        log_label.setStyleSheet("font-size: 10px; color: #999;")
        status_layout.addWidget(log_label)
        
        status_group.setLayout(status_layout)
        left.addWidget(status_group)
        
        left.addStretch()
        
        # === 우측 패널 ===
        right_widget = QWidget()
        right = QVBoxLayout(right_widget)
        
        header = QLabel("사양서 Q&A (GraphRAG)")
        header.setStyleSheet("font-size: 14px; font-weight: bold; padding: 5px;")
        right.addWidget(header)
        
        # 대화 영역
        self.history = QTextEdit()
        self.history.setReadOnly(True)
        self.history.setStyleSheet("""
            QTextEdit {
                background-color: #fafafa;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        right.addWidget(self.history, stretch=7)
        
        # 디버그 영역
        debug_label = QLabel("처리 과정 (실시간)")
        debug_label.setStyleSheet("font-size: 11px; color: #666; margin-top: 5px;")
        right.addWidget(debug_label)
        
        self.debug_log = QTextEdit()
        self.debug_log.setReadOnly(True)
        self.debug_log.setMaximumHeight(100)
        self.debug_log.setStyleSheet("""
            QTextEdit {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 3px;
                font-family: Consolas, monospace;
                font-size: 11px;
                padding: 5px;
            }
        """)
        right.addWidget(self.debug_log, stretch=1)
        
        # 입력 영역
        self.input = ChatInput()
        self.input.setFixedHeight(70)
        self.input.setPlaceholderText("질문 입력 (Enter: 전송, Shift+Enter: 줄바꿈)")
        self.input.enter_pressed.connect(self.ask)
        self.input.setEnabled(False)
        self.input.setStyleSheet("""
            QTextEdit {
                border: 2px solid #4CAF50;
                border-radius: 5px;
                padding: 8px;
                font-size: 13px;
            }
            QTextEdit:disabled {
                border-color: #ccc;
                background-color: #f5f5f5;
            }
        """)
        right.addWidget(self.input)
        
        self.btn_send = QPushButton("질문하기 (Enter)")
        self.btn_send.clicked.connect(self.ask)
        self.btn_send.setEnabled(False)
        self.btn_send.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:disabled { background-color: #ccc; }
        """)
        right.addWidget(self.btn_send)
        
        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([280, 900])
        
        main_layout.addWidget(splitter)
        
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        
        logger.debug("UI 초기화 완료")

    def update_status(self):
        """상태 표시 업데이트"""
        try:
            lines = []
            
            if self.pdf_path and self.processor:
                page_count = len(self.processor._page_cache.get(self.pdf_path, {}))
                lines.append(f"PDF: {page_count}페이지 로드됨")
            else:
                lines.append("PDF: 미선택")
            
            if self.processor and self.processor.graph_available:
                lines.append("Graph DB: 사용 가능")
            else:
                lines.append("Graph DB: 미구축")
            
            if self.current_data:
                n = len(self.current_data.get('nodes', []))
                e = len(self.current_data.get('edges', []))
                lines.append(f"노드: {n}개, 엣지: {e}개")
            
            self.lbl_status.setText("\n".join(lines))
        except Exception as e:
            logger.error(f"update_status 오류: {e}")

    def on_debug(self, info):
        """디버그 콜백"""
        try:
            self.debug_log.append(str(info))
            cursor = self.debug_log.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.debug_log.setTextCursor(cursor)
            QApplication.processEvents()
        except Exception as e:
            logger.error(f"on_debug 오류: {e}")

    def select_file(self):
        logger.info("파일 선택 다이얼로그 열기")
        
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "PDF 선택", "", "PDF Files (*.pdf)"
            )
            
            if file_path:
                logger.info(f"파일 선택됨: {file_path}")
                
                self.pdf_path = os.path.normpath(file_path)
                self.lbl_file.setText(f"선택: {Path(file_path).name}")
                self.lbl_file.setStyleSheet("color: #4CAF50; padding: 5px; font-weight: bold;")
                
                # PDF 로드
                self.debug_log.clear()
                self.history.append(f"[시스템] PDF 로드 중: {Path(file_path).name}")
                QApplication.processEvents()
                
                self.processor.load_pdf(self.pdf_path)
                
                # 버튼 활성화
                self.btn_build.setEnabled(True)
                self.input.setEnabled(True)
                self.btn_send.setEnabled(True)
                
                self.current_data = None
                self.btn_view.setEnabled(False)
                
                self.update_status()
                
                self.history.append("[시스템] PDF 로드 완료!")
                self.history.append("[시스템] 질문을 입력하세요. (Graph DB 구축은 선택 사항)")
                self.history.append("-" * 40)
                
                logger.info("PDF 로드 완료")
                
        except Exception as e:
            logger.error(f"select_file 오류: {e}")
            logger.error(traceback.format_exc())
            QMessageBox.critical(self, "오류", f"파일 로드 실패:\n{str(e)}")

    def build_graph(self):
        if not self.pdf_path:
            return
        
        logger.info("Graph DB 구축 시작")
        
        try:
            mode = "general" if self.radio_gen.isChecked() else "detailed"
            
            self.btn_build.setEnabled(False)
            self.btn_select.setEnabled(False)
            self.debug_log.clear()
            self.history.append(f"\n[시스템] {mode} 모드로 Graph DB 구축 시작...")
            
            self.worker = BuildWorker(self.processor, self.pdf_path, mode)
            self.worker.progress.connect(lambda m: self.history.append(f"[시스템] {m}"))
            self.worker.finished.connect(self.on_build_finished)
            self.worker.error.connect(self.on_build_error)
            self.worker.start()
            
        except Exception as e:
            logger.error(f"build_graph 오류: {e}")
            logger.error(traceback.format_exc())
            self.btn_build.setEnabled(True)
            self.btn_select.setEnabled(True)
            QMessageBox.critical(self, "오류", f"Graph 구축 실패:\n{str(e)}")

    def on_build_finished(self, data, files):
        logger.info("Graph 구축 완료 콜백")
        
        try:
            self.current_data = data
            self.btn_build.setEnabled(True)
            self.btn_select.setEnabled(True)
            self.btn_view.setEnabled(True)
            
            n = len(data.get('nodes', []))
            e = len(data.get('edges', []))
            
            self.history.append(f"[시스템] Graph DB 구축 완료! (노드 {n}개, 엣지 {e}개)")
            self.history.append(f"[시스템] CSV 저장: graph_db/")
            
            self.update_status()
            
            QMessageBox.information(self, "완료", f"Graph DB 구축 완료!\n노드: {n}개, 엣지: {e}개")
            
        except Exception as e:
            logger.error(f"on_build_finished 오류: {e}")

    def on_build_error(self, error):
        logger.error(f"Graph 구축 오류: {error}")
        
        self.btn_build.setEnabled(True)
        self.btn_select.setEnabled(True)
        self.history.append(f"[오류] {error}")
        QMessageBox.critical(self, "오류", f"구축 실패:\n{error}")

    def view_graph(self):
        if not self.current_data:
            QMessageBox.warning(self, "알림", "먼저 Graph DB를 구축하세요.")
            return
        
        if Network is None:
            QMessageBox.warning(self, "알림", "pyvis가 설치되지 않았습니다.\npip install pyvis")
            return
        
        logger.info("그래프 시각화 시작")
        
        try:
            net = Network(
                height="700px", width="100%",
                bgcolor="#ffffff", font_color="black", directed=True
            )
            
            net.set_options("""
            {
                "nodes": {"font": {"size": 14}},
                "edges": {
                    "arrows": {"to": {"enabled": true}},
                    "font": {"size": 9},
                    "smooth": {"type": "curvedCW", "roundness": 0.2}
                },
                "physics": {
                    "forceAtlas2Based": {"gravitationalConstant": -50},
                    "solver": "forceAtlas2Based"
                },
                "interaction": {"hover": true}
            }
            """)
            
            colors = {
                'Device': '#4CAF50', 'Feature': '#2196F3', 'Spec': '#FF9800',
                'Component': '#9C27B0', 'Sensor': '#E91E63', 'Function': '#3F51B5',
                'Setting': '#009688', 'default': '#607D8B'
            }
            
            for n in self.current_data.get('nodes', []):
                nid = str(n.get('id', ''))
                label = n.get('name', nid)
                ntype = n.get('label', 'default')
                desc = n.get('desc', '')
                page = n.get('page', '')
                
                title = f"이름: {label}\n타입: {ntype}\n설명: {desc}\n페이지: {page}"
                
                net.add_node(nid, label=label, title=title,
                           color=colors.get(ntype, colors['default']), size=20)
            
            for e in self.current_data.get('edges', []):
                rel = e.get('rel', '')
                net.add_edge(str(e.get('from', '')), str(e.get('to', '')),
                           label=rel, title=f"관계: {rel}")
            
            output = os.path.join(os.getcwd(), "graph_view.html")
            net.save_graph(output)
            
            # 범례 추가
            self._add_legend(output)
            
            os.startfile(output)
            self.history.append(f"[시스템] 그래프 시각화: {output}")
            
            logger.info("그래프 시각화 완료")
            
        except Exception as e:
            logger.error(f"view_graph 오류: {e}")
            logger.error(traceback.format_exc())
            QMessageBox.critical(self, "오류", f"시각화 실패:\n{str(e)}")

    def _add_legend(self, html_path):
        """HTML에 범례 추가"""
        try:
            legend = """
            <div style="position:fixed;bottom:0;left:0;right:0;background:#f8f9fa;
                        border-top:2px solid #dee2e6;padding:15px;font-family:'Malgun Gothic',sans-serif;font-size:13px;">
                <div style="display:flex;gap:40px;flex-wrap:wrap;">
                    <div>
                        <strong>노드 = 구성요소</strong><br>
                        <span style="background:#4CAF50;color:white;padding:2px 6px;border-radius:3px;">Device</span>
                        <span style="background:#2196F3;color:white;padding:2px 6px;border-radius:3px;">Feature</span>
                        <span style="background:#FF9800;color:white;padding:2px 6px;border-radius:3px;">Spec</span>
                        <span style="background:#9C27B0;color:white;padding:2px 6px;border-radius:3px;">Component</span>
                    </div>
                    <div>
                        <strong>엣지 = 관계</strong><br>
                        CONTAINS:포함 | HAS_FEATURE:기능보유 | RUNS:실행 | USES:사용
                    </div>
                </div>
            </div>
            <style>#mynetwork{margin-bottom:100px!important;}</style>
            """
            
            with open(html_path, 'r', encoding='utf-8') as f:
                content = f.read()
            content = content.replace('</body>', f'{legend}</body>')
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            logger.warning(f"범례 추가 실패: {e}")

    def ask(self):
        question = self.input.toPlainText().strip()
        if not question or not self.pdf_path:
            return
        
        logger.info(f"질문 전송: {question[:30]}...")
        
        try:
            self.history.append(f"\n[나] {question}")
            self.input.clear()
            self.debug_log.clear()
            
            # UI 비활성화
            self.btn_send.setEnabled(False)
            self.input.setEnabled(False)
            
            self.history.append("[시스템] 답변 생성 중...")
            QApplication.processEvents()
            
            # 백그라운드 처리
            self.ask_worker = AskWorker(self.processor, question, self.pdf_path)
            self.ask_worker.finished.connect(self.on_ask_finished)
            self.ask_worker.error.connect(self.on_ask_error)
            self.ask_worker.start()
            
        except Exception as e:
            logger.error(f"ask 오류: {e}")
            logger.error(traceback.format_exc())
            self.btn_send.setEnabled(True)
            self.input.setEnabled(True)
            self.history.append(f"[오류] {str(e)}")

    def on_ask_finished(self, answer):
        logger.info("답변 수신 완료")
        
        try:
            self.history.append(f"\n[AI] {answer}")
            self.history.append("-" * 40)
            
            self.btn_send.setEnabled(True)
            self.input.setEnabled(True)
            self.input.setFocus()
            
            cursor = self.history.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.history.setTextCursor(cursor)
            
        except Exception as e:
            logger.error(f"on_ask_finished 오류: {e}")

    def on_ask_error(self, error):
        logger.error(f"답변 오류: {error}")
        
        self.history.append(f"\n[오류] {error}")
        self.btn_send.setEnabled(True)
        self.input.setEnabled(True)
        self.input.setFocus()

    def closeEvent(self, event):
        logger.info("앱 종료 시작")
        
        try:
            if self.processor:
                self.processor.close()
        except Exception as e:
            logger.error(f"closeEvent 오류: {e}")
        
        logger.info("앱 종료 완료")
        event.accept()


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("프로그램 시작")
    logger.info("=" * 60)
    
    try:
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        
        app.setStyleSheet("""
            QMainWindow { background-color: #ffffff; }
            QGroupBox { font-weight: bold; border: 1px solid #ddd; border-radius: 5px; margin-top: 10px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton { padding: 8px 15px; }
        """)
        
        window = R9App()
        window.show()
        
        logger.info("메인 윈도우 표시 완료")
        
        sys.exit(app.exec())
        
    except Exception as e:
        logger.critical(f"메인 실행 오류: {e}")
        logger.critical(traceback.format_exc())
        
        # 오류 메시지 박스 (가능한 경우)
        try:
            QMessageBox.critical(None, "치명적 오류", 
                f"프로그램 실행 실패:\n{str(e)}\n\n로그 파일: {log_file}")
        except:
            pass
        
        raise
