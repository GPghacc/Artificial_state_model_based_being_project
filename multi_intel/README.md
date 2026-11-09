# multi_intel

지능을 기능 단위로 나누고, 각 기능을 별도 처리기가 채우며, 내용을 모르는 심판이 조정하는 구조의 프로토타입. 설계 근거는 `DESIGN.md`, 심리학 재료는 `다중지능_설계재료_정리.md`.

## 실행

Python 3.10 이상, 외부 의존성 없음.

```
python run_demo.py                          # 모의 백엔드로 여섯 개 자극을 한 사이클씩
python -m experiments.calibration --n 300   # 확신도 보정 실험, 습관 층 켬/끔 비교
python -m unittest discover tests           # 테스트
```

실제 모델을 꽂을 때:

```
python run_demo.py ollama                   # 로컬 Ollama (기본 모델 llama3.1, OLLAMA_HOST로 주소 변경)
python run_demo.py anthropic                # ANTHROPIC_API_KEY 필요
python -m experiments.calibration --backend ollama --n 100
```

## 구조

```
mi/
  types.py        공용 타입 (Stimulus, Signature, Cues, Bid, Result, Outcome, Episode)
  backend.py      모델 백엔드 인터페이스 + Mock / Ollama / Anthropic 어댑터
  classifier.py   싼 표면 분류기 — 양식만 보고 후보 추리기
  modules/        도구들. cues()는 싸게, run()은 이긴 놈만
    retrieval.py  검색 — 단서: 유사도, 격차, 히트 수
    code.py       계산 — 검증 가능 (파싱·실행)
    reasoning.py  추론 — 단서: 자기 일관성 (k 샘플 합치)
  confidence.py   단서 → P(성공). 모듈별 온라인 로지스틱, 성패로만 학습
  habit.py        호출 이력 → 사전 확률. 탐색 매개변수 포함
  referee.py      심판 — 충돌·신규·목표 위반 시에만 개입
  workspace.py    일화 로그
  system.py       한 사이클 배선
experiments/
  tasks.py        정답 있는 과제 생성 (산술 / 검색 / 정답 없음)
  calibration.py  보정 실험
tests/
run_demo.py
```

## 모듈 추가

`mi/modules/base.py`의 `Module`을 상속해 `name`, `cue_names`, `cues()`, `run()`을 채운다. 단서 값은 0~1로 맞춘다. 검증할 수 있으면 `Result.verified`를 채운다. 정답과 비교하는 방식이 문자열 일치가 아니면 `judge()`를 덮어쓴다. `MultiIntelSystem`에 인스턴스를 넘기면 끝이다. 분류기가 그 모듈을 언제 후보에서 빼야 하는지는 `classifier.candidates_for`에 적는다.
