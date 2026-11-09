"""판단: 추출값을 보고 다음 주의를 정한다. v0는 아무것도 바꾸지 않는다.
나중에 규칙 기반 → 학습 기반으로 교체하는 자리."""


class Controller:
    def decide(self, readout, attention):
        return None   # None = 주의 유지. 바꾸려면 spots 목록을 반환
