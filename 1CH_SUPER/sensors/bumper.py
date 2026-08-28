from config import BUMPER_GPIO


class Bumper:
    def __init__(self):
        self._btn = None
        self.hit = False
        try:
            from gpiozero import Button
            self._btn = Button(BUMPER_GPIO, pull_up=True, bounce_time=0.01)
            self._btn.when_pressed = self._on_hit
        except Exception as e:
            pass

    def _on_hit(self):
        self.hit = True

    @property
    def pressed(self):
        return self.hit or (self._btn is not None and self._btn.is_pressed)

    def reset(self):
        self.hit = False

    def stop(self):
        if self._btn:
            self._btn.close()
