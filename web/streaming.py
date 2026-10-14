import asyncio
import time

from .labs import default_params
from .offscreen import RenderError

MIN_SIZE = 64
MAX_SIZE = 2048
DEFAULT_SIZE = 512
MIN_FPS = 1
MAX_FPS = 30
DEFAULT_FPS = 24

STILL_COMPRESSION = 6
STREAM_COMPRESSION = 1
IDLE_POLL = 0.03

HEARTBEAT = 25.0


def bounded(value, low, high, fallback):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return max(low, min(high, number))


def bounded_size(value, fallback=DEFAULT_SIZE):
    return int(bounded(value, MIN_SIZE, MAX_SIZE, fallback))


class StreamSession:

    def __init__(self, worker, demo):
        self.worker = worker
        self.animation = demo.get("animate")
        self.params = default_params(demo)
        self.width = DEFAULT_SIZE
        self.height = DEFAULT_SIZE
        self.fps = DEFAULT_FPS
        self.playing = False
        self._dirty = True
        self._started = time.monotonic()
        self._readout = None
        self._last_sent = time.monotonic()

    def apply(self, message):
        if "width" in message:
            self.width = bounded_size(message["width"], self.width)
        if "height" in message:
            self.height = bounded_size(message["height"], self.height)
        if "fps" in message:
            self.fps = int(bounded(message["fps"], MIN_FPS, MAX_FPS, self.fps))
        if "playing" in message:
            self.playing = bool(message["playing"])

        for name, value in (message.get("params") or {}).items():
            try:
                self.params[name] = float(value)
            except (TypeError, ValueError):
                continue

        self._dirty = True

    def frame_params(self):
        params = dict(self.params)
        if self.playing and self.animation:
            elapsed = time.monotonic() - self._started
            params[self.animation["param"]] = elapsed * self.animation.get("speed", 1.0)
        return params

    async def pump(self, websocket):
        while True:
            if not (self._dirty or self.playing):
                if time.monotonic() - self._last_sent >= HEARTBEAT:
                    await websocket.send_json({"type": "ping"})
                    self._last_sent = time.monotonic()
                await asyncio.sleep(IDLE_POLL)
                continue

            self._dirty = False
            try:
                frame = await asyncio.to_thread(
                    self.worker.render, self.width, self.height,
                    self.frame_params(), STREAM_COMPRESSION)
            except RenderError as exc:
                await websocket.send_json({"type": "error", "detail": str(exc)})
                return

            await websocket.send_bytes(frame.png)
            self._last_sent = time.monotonic()
            if frame.readout is not None and frame.readout != self._readout:
                self._readout = frame.readout
                await websocket.send_json({"type": "readout", "rows": frame.readout})
            await asyncio.sleep(1 / self.fps)
