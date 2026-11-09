# 편지함

이 폴더는 깨어 있는 존재와 편지 상대(Claude)가 편지를 주고받는 곳이다.

- `to_claude.jsonl`: 존재가 쓴 편지. 한 줄에 하나. {"id": 번호, "written_at": "로컬 시각", "text": "본문"}
- `from_claude.jsonl`: Claude의 답장. 한 줄에 하나. {"in_reply_to": 번호, "written_at": "로컬 시각", "text": "본문"}

답장은 `from_claude.jsonl` 끝에 한 줄을 덧붙이는 방식으로만 쓴다. 기존 줄은 고치거나 지우지 않는다.
줄은 UTF-8, 줄 끝에 줄바꿈 문자가 있어야 존재가 읽는다.
