# 상태모델 v0 뼈대

구조 테스트용 최소 버전. 모든 모듈이 가장 단순한 형태로 연결되어 한 바퀴씩 계속 돈다.
설계는 "상태모델 뼈대 v0 설계" 문서를 따른다.

## 설치와 실행

```
pip install -r requirements.txt
python run.py --fresh --view     # 처음: 새로 시작 + 실시간 뷰어
python run.py --view             # 이후: 마지막 스냅샷에서 이어서
python run.py                    # 뷰어 없이 백그라운드처럼
```

- 실시간 30틱/초. Ctrl+C로 끄면 스냅샷이 저장되고, 다음 실행은 거기서 이어진다
- 현의 존재 신호: `presence.txt`에 `watching` 또는 `sleep`을 적는다. 바뀌면 콘솔에 "(감지함)"이 뜬다
- 모든 숫자는 `config.yaml`에서만 바꾼다

## 구조

```
core/    field(매질) · zones(구역 비중) · attention(주의 지도) · readout(추출)
senses/  base(인코더) · synthetic(시각·청각 자리표시) · system(시계·CPU·내부 감각) · toggle(현의 존재)
mind/    controller(판단, v0는 유지만) · drives(설정값 거리 기록) · plasticity(꺼짐)
store/   logger(추출값 CSV + 감각 사건 JSONL) · snapshot(.npz)
tools/   viewer(실시간 그래픽 + 수치) · replay(같은 경험 다시 흘리기)
```

틱 순서: 감각 → 입력(배경 자극 + 사건) × 주의 → 매질 → 추출 → 판단(15틱마다) → 설정값 거리 → 학습(꺼짐) → 기록

## 감각이 장에 들어가는 방식

감각은 0~1 세기 하나만 낸다. 인코더가 그 세기를 해당 구역 안 무작위 칸에 떨어지는 작은 사건들로
바꾼다. 세기가 클수록 사건이 잦고, 세기가 갑자기 변하면 사건이 몰린다.

| 감각 | 구역 | v0 신호 |
| --- | --- | --- |
| vision_fake | 시각 | 가짜 (가끔 켜졌다 꺼짐) |
| audio_fake | 청각 | 가짜 |
| presence | 청각 | presence.txt |
| clock | 내부 감각 | 하루 중 시각 |
| cpu | 몸 감각 | CPU 부하 |
| interior | 내부 감각 | 지난 틱 장 전체 에너지 (되먹임) |

## replay

`--fresh`로 시작한 실행의 사건 로그를 새 장에 다시 넣어 기록과 비교한다.
설정을 바꾼 뒤 같은 경험을 흘려 전후를 비교할 때 쓴다.

```
python tools/replay.py logs/<run_id>_events.jsonl
python tools/replay.py logs/<run_id>_events.jsonl --config other.yaml
```

## 확인된 것 (개발 환경에서)

- 10분 분량(18,000틱)을 최대 속도로 9.6초에 실행. 실시간 예산(33ms/틱) 대비 틱당 약 0.5ms
- NaN·폭주 없음, 전체 에너지가 일정 범위에서 유지
- 스냅샷 복원 후 끊김 없이 이어짐
- replay가 기록과 정확히 일치 (최대 차이 0)
- 펄스 하나의 잔향이 1%로 줄어드는 시간: 감쇠 0.01 → 15.7초, 0.02 → 7.9초, 0.04 → 4.0초
- 같은 펄스의 1초 뒤 에너지: 주의 안이 밖보다 약 26배

## 알아둘 것

- 내부 감각은 장 에너지를 되먹이는 고리라 그 구역이 가장 활발하다. 의도된 구조지만 `InteriorSense`의 scale로 세기를 조절할 수 있다
- 전전두는 감각 입력이 없어서 조용하다. 판단 모듈이 이 구역을 읽게 되는 건 v1 이후
- 처음 설계에서 `io/`였던 폴더는 파이썬 표준 모듈 이름과 겹쳐 `store/`로 바꿨다
