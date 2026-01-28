import time

import cv2


class FPSCounter:
    """!Utility to estimate FPS over a sliding window."""

    def __init__(self, window=30):
        """!Initialize FPS counter.

        @param window Number of frames in the FPS window.
        """
        self.window = window
        self._times = []

    def tick(self):
        """!Record a frame and return current FPS estimate.

        @return FPS value (float).
        """
        now = time.time()
        self._times.append(now)
        if len(self._times) > self.window:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])


def draw_hud(frame, fps, effect_name, state, help_text=True):
    """!Draw HUD overlay (FPS, effect name, help).

    @param frame BGR frame to draw on (in-place).
    @param fps Current FPS estimate.
    @param effect_name Active effect string.
    @param state EffectState containing toggles.
    @param help_text Whether to draw the help line.
    @return Frame with HUD.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    color = (235, 235, 235)
    shadow = (20, 20, 20)

    def put(text, x, y):
        cv2.putText(frame, text, (x + 1, y + 1), font, scale, shadow, 2, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), font, scale, color, 1, cv2.LINE_AA)

    put(f"FPS: {fps:.1f}", 12, 22)
    put(f"Effect: {effect_name}", 12, 44)

    if help_text:
        put(
            "Keys: 1-7 effects | m mask | f freeze bg | r record | s snapshot | d dump | q quit | use Controls window",
            12,
            frame.shape[0] - 12,
        )

    if state.show_mask:
        put("Mask preview: ON", frame.shape[1] - 170, 22)

    return frame
