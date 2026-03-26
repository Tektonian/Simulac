import atexit
import json
import time
from abc import abstractmethod
from datetime import datetime, timezone
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from tt.base.envvar.envvar import IEnvvarService
from tt.base.instantiate.instantiate import ServiceIdentifier, service_identifier
from tt.sdk.telemetry_service.common.telemetry import TelemetryLevelEnum


@service_identifier("ITelemetryService")
class ITelemetryService(ServiceIdentifier["ITelemetryService"]):
    @abstractmethod
    def public_log(self, event_name: str, data: dict): ...

    @abstractmethod
    def public_log2(self, event_name: str, data: dict):
        """Use this when user agreement needed"""

    @abstractmethod
    def public_error(self, event_name: str, data: dict): ...


class TelemetryService(ITelemetryService):
    BUFFER_FLUSH_TIMEOUT = 10  # 10s
    BUFFER_SIZE_LIMIT = 1000  # max queued event count
    REQUEST_TIMEOUT = 3
    EXIT_JOIN_TIMEOUT = 5
    _QUEUE_CLOSE = object()

    def __init__(self, EnvvarService: IEnvvarService):
        self.EnvvarService = EnvvarService

        self.telemetry_thread: None | Thread = None
        self._telemetry_thread_lock = Lock()
        self._queue: Queue[dict[str, object] | object] = Queue(
            maxsize=self.BUFFER_SIZE_LIMIT
        )
        self._shutdown_event = Event()
        self._is_closed = False
        self._dropped_events = 0

        atexit.register(self._close)

    def public_log(self, event_name: str, data: dict):
        self._enqueue(
            self._build_event(
                event_name,
                data,
                TelemetryLevelEnum.USAGE,
                requires_user_agreement=False,
            )
        )

    def public_log2(self, event_name: str, data: dict):
        self._enqueue(
            self._build_event(
                event_name,
                data,
                TelemetryLevelEnum.USAGE,
                requires_user_agreement=True,
            )
        )

    def public_error(self, event_name: str, data: dict):
        self._enqueue(
            self._build_event(
                event_name,
                data,
                TelemetryLevelEnum.ERROR,
                requires_user_agreement=False,
            )
        )

    def _close(self):
        with self._telemetry_thread_lock:
            if self._is_closed:
                return

            self._is_closed = True
            self._shutdown_event.set()
            thread = self.telemetry_thread

        self._wake_worker()

        if thread is not None and thread.is_alive():
            thread.join(timeout=self.EXIT_JOIN_TIMEOUT)

    def _enqueue(self, payload: dict[str, object]) -> None:
        if self.EnvvarService.telemetry_disabled:
            return

        if not self._ensure_worker():
            return

        try:
            self._queue.put_nowait(payload)
            return
        except Full:
            pass

        # Prefer the latest telemetry over the oldest when the buffer is saturated.
        try:
            self._queue.get_nowait()
            self._note_dropped_event()
        except Empty:
            pass

        try:
            self._queue.put_nowait(payload)
        except Full:
            self._note_dropped_event()

    def _ensure_worker(self) -> bool:
        with self._telemetry_thread_lock:
            if self._is_closed or self.EnvvarService.telemetry_disabled:
                return False

            if self.telemetry_thread is not None and self.telemetry_thread.is_alive():
                return True

            self.telemetry_thread = Thread(
                target=self._telemetry_worker,
                name="simulac-telemetry",
                daemon=True,
            )
            self.telemetry_thread.start()
            return True

    def _wake_worker(self) -> None:
        try:
            self._queue.put_nowait(self._QUEUE_CLOSE)
        except Full:
            pass

    def _note_dropped_event(self) -> None:
        with self._telemetry_thread_lock:
            self._dropped_events += 1

    def _consume_dropped_events(self) -> int:
        with self._telemetry_thread_lock:
            dropped_events = self._dropped_events
            self._dropped_events = 0
            return dropped_events

    def _telemetry_worker(self) -> None:
        batch: list[dict[str, object]] = []
        flush_deadline: float | None = None

        while True:
            if self._shutdown_event.is_set():
                try:
                    queued = self._queue.get_nowait()
                except Empty:
                    if batch:
                        self._send_batch(batch)
                    return

                if queued is self._QUEUE_CLOSE:
                    continue

                batch.append(queued)
                if len(batch) >= self.BUFFER_SIZE_LIMIT:
                    self._send_batch(batch)
                    batch = []
                continue

            try:
                if batch and flush_deadline is not None:
                    wait_time = flush_deadline - time.monotonic()
                    if wait_time <= 0:
                        raise Empty
                    queued = self._queue.get(timeout=wait_time)
                else:
                    queued = self._queue.get(timeout=self.BUFFER_FLUSH_TIMEOUT)
            except Empty:
                if batch:
                    self._send_batch(batch)
                    batch = []
                    flush_deadline = None
                continue

            if queued is self._QUEUE_CLOSE:
                self._shutdown_event.set()
                continue

            batch.append(queued)
            if len(batch) == 1:
                flush_deadline = time.monotonic() + self.BUFFER_FLUSH_TIMEOUT

            if len(batch) >= self.BUFFER_SIZE_LIMIT:
                self._send_batch(batch)
                batch = []
                flush_deadline = None

    def _send_batch(self, batch: list[dict[str, object]]) -> None:
        events = list(batch)
        batch.clear()

        dropped_events = self._consume_dropped_events()
        if dropped_events > 0:
            events.insert(
                0,
                self._build_event(
                    "telemetry.events_dropped",
                    {"count": dropped_events},
                    TelemetryLevelEnum.ERROR,
                    requires_user_agreement=False,
                ),
            )

        for event in events:
            self._send_event(event)

    def _send_event(self, event: dict[str, object]) -> None:
        token = self.EnvvarService.token
        url = urljoin(
            self.EnvvarService.base_url.rstrip("/") + "/", "telemetry/simulac"
        )

        payload = {
            "sdk": "simulac-python",
            "sent_at": datetime.now(timezone.utc).isoformat(),
            "data": event,
            "api_key": token,
        }

        attempt_count = 2 if self._shutdown_event.is_set() else 1

        for count in range(attempt_count):
            request = Request(
                url=url,
                data=json.dumps(payload, default=str).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            try:
                with urlopen(request, timeout=self.REQUEST_TIMEOUT) as response:
                    response.read()
                return
            except (HTTPError, URLError, OSError, TimeoutError):
                if count + 1 >= attempt_count:
                    return
                time.sleep(0.25)

    def _build_event(
        self,
        event_name: str,
        data: dict,
        level: TelemetryLevelEnum,
        requires_user_agreement: bool,
    ) -> dict[str, object]:
        return {
            "event_name": event_name,
            "level": level.name.lower(),
            "requires_user_agreement": requires_user_agreement,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "data": dict(data),
        }
