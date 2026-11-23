"""Vector re-reads a cache causes when it admits vectors, scheduled over later requests."""


class RereadSchedule:
    """Spreads admitted-vector counts evenly over the next `spread` requests.

    spread <= 0 charges everything to the next request taken (the construction itself).
    """

    def __init__(self, spread: int):
        self.spread = spread
        self.left = 0.0
        self.steps = 0
        self.credit = 0.0

    def add(self, total: int) -> None:
        if total <= 0:
            return
        self.left += total
        self.steps = max(self.spread, 1)

    def take(self) -> int:
        """Vectors to re-read as background work at this request."""
        if self.steps <= 0 or self.left <= 0:
            return 0
        if self.steps == 1:
            amount = int(round(self.left))
        else:
            self.credit += self.left / self.steps
            amount = int(self.credit)
            self.credit -= amount
        self.left = max(self.left - amount, 0.0)
        self.steps -= 1
        if self.steps == 0:
            self.left, self.credit = 0.0, 0.0
        return amount
