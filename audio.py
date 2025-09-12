import pyttsx3

_engine = None

def _get():
    global _engine
    if _engine is None:
        try:
            _engine = pyttsx3.init()
        except Exception:
            _engine = None
    return _engine

def speak(text: str):
    eng = _get()
    if eng is None:
        print("[TTS]", text)
        return
    try:
        eng.say(text)
        eng.runAndWait()
    except Exception:
        print("[TTS-Fallback]", text)
