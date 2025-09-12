import time, requests, threading, queue

LIVE_ALL = "http://127.0.0.1:2999/liveclientdata/allgamedata"
LIVE_EVENTS = "http://127.0.0.1:2999/liveclientdata/eventdata"

def _get(url: str, timeout: float=1.5):
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

def fetch_all(timeout: float=1.5):
    return _get(LIVE_ALL, timeout)

def fetch_events(timeout: float=1.5):
    j = _get(LIVE_EVENTS, timeout)
    if isinstance(j, dict):
        return j.get("Events") or []
    return []

def poll_live_client(stop_event: threading.Event, out_q: queue.Queue, interval_s: float=1.0):
    while not stop_event.is_set():
        j = fetch_all()
        events = fetch_events() or []
        out_q.put({"all": j, "events": events, "t": time.time()})
        time.sleep(interval_s)
