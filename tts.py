import os
import time
import threading
import queue
from typing import Optional, Tuple, Generator
import numpy as np
import sounddevice as sd

# Kokoro TTS imports
KOKORO_AVAILABLE = False
try:
    from kokoro import KPipeline
    KOKORO_AVAILABLE = True
except ImportError:
    pass


class TextToSpeech:
    """
    Kokoro TTS with barge-in support for natural conversation.
    """

    def __init__(
        self,
        voice: str = "af_heart",
        speed: float = 1.0,
        sample_rate: int = 24000
    ):
        self.voice = voice
        self.speed = speed
        self.sample_rate = sample_rate
        self.pipeline = None

        # Latency tracking
        self.last_synthesis_latency_ms = 0
        self.last_total_latency_ms = 0

        # Playback control
        self._stop_event = threading.Event()
        self._playback_thread = None
        self._is_speaking = False
        self._was_interrupted = False

        if KOKORO_AVAILABLE:
            try:
                self.pipeline = KPipeline(lang_code='a')
            except Exception as e:
                print(f"Kokoro init failed: {e}")
                self.pipeline = None

    def synthesize(self, text: str) -> Tuple[np.ndarray, float]:
        """Synthesize text to audio."""
        if not text or not text.strip():
            return np.array([], dtype=np.float32), 0

        if self.pipeline is None:
            return np.array([], dtype=np.float32), 0

        start_time = time.perf_counter()

        try:
            audio_chunks = []

            for _, _, audio in self.pipeline(
                text,
                voice=self.voice,
                speed=self.speed
            ):
                if audio is not None:
                    audio_chunks.append(audio)

            synthesis_time = (time.perf_counter() - start_time) * 1000
            self.last_synthesis_latency_ms = synthesis_time

            if audio_chunks:
                return np.concatenate(audio_chunks), synthesis_time

            return np.array([], dtype=np.float32), synthesis_time

        except Exception as e:
            print(f"TTS error: {e}")
            return np.array([], dtype=np.float32), 0

    def speak(self, text: str, blocking: bool = True) -> dict:
        """Synthesize and play text."""
        if not text or not text.strip():
            return {"error": "Empty text"}

        total_start = time.perf_counter()

        self.stop()
        self._stop_event.clear()
        self._was_interrupted = False

        audio, synth_time = self.synthesize(text)

        metrics = {
            "synthesis_ms": synth_time,
            "audio_duration_ms": 0,
            "total_ms": 0,
            "interrupted": False
        }

        if len(audio) > 0:
            metrics["audio_duration_ms"] = len(audio) / self.sample_rate * 1000

            if blocking:
                self._play_blocking(audio)
            else:
                self._play_async(audio)

            metrics["interrupted"] = self._was_interrupted

        metrics["total_ms"] = (time.perf_counter() - total_start) * 1000
        self.last_total_latency_ms = metrics["total_ms"]

        return metrics

    def speak_interruptible(self, text: str, interrupt_detector) -> dict:
        """
        Speak text with barge-in support.
        Stops immediately when interrupt_detector returns True.
        
        Args:
            text: Text to speak
            interrupt_detector: Callable that returns True if user is speaking
        """
        if not text or not text.strip():
            return {"error": "Empty text"}

        total_start = time.perf_counter()

        self.stop()
        self._stop_event.clear()
        self._was_interrupted = False

        audio, synth_time = self.synthesize(text)

        metrics = {
            "synthesis_ms": synth_time,
            "audio_duration_ms": 0,
            "total_ms": 0,
            "interrupted": False
        }

        if len(audio) > 0:
            metrics["audio_duration_ms"] = len(audio) / self.sample_rate * 1000
            
            # Play in chunks with interrupt checking
            self._is_speaking = True
            chunk_duration = 0.15  # 150ms chunks for responsive interruption
            chunk_samples = int(self.sample_rate * chunk_duration)
            
            try:
                for i in range(0, len(audio), chunk_samples):
                    if self._stop_event.is_set():
                        self._was_interrupted = True
                        break
                    
                    # Check for interruption before playing each chunk
                    if interrupt_detector():
                        self._was_interrupted = True
                        sd.stop()
                        break
                    
                    chunk = audio[i:i + chunk_samples]
                    sd.play(chunk, samplerate=self.sample_rate)
                    sd.wait()
                    
            except Exception as e:
                print(f"Playback error: {e}")
            finally:
                self._is_speaking = False
                
            metrics["interrupted"] = self._was_interrupted

        metrics["total_ms"] = (time.perf_counter() - total_start) * 1000
        self.last_total_latency_ms = metrics["total_ms"]

        return metrics

    def _play_blocking(self, audio: np.ndarray):
        """Play audio and wait for completion."""
        self._is_speaking = True
        try:
            sd.play(audio, samplerate=self.sample_rate)
            sd.wait()
        except Exception as e:
            print(f"Playback error: {e}")
        finally:
            self._is_speaking = False

    def _play_async(self, audio: np.ndarray):
        """Play audio in background."""
        def play():
            self._is_speaking = True
            try:
                sd.play(audio, samplerate=self.sample_rate)
                sd.wait()
            except Exception as e:
                print(f"Playback error: {e}")
            finally:
                self._is_speaking = False

        self._playback_thread = threading.Thread(target=play)
        self._playback_thread.start()

    def stop(self):
        """Stop current playback immediately."""
        self._stop_event.set()
        self._was_interrupted = True
        sd.stop()
        self._is_speaking = False
        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=0.5)

    @property
    def is_speaking(self):
        """Check if TTS is currently playing audio."""
        return self._is_speaking

    def set_voice(self, voice: str):
        """Change voice."""
        self.voice = voice

    def set_speed(self, speed: float):
        """Change speed (0.5-2.0)."""
        self.speed = max(0.5, min(2.0, speed))


def get_tts_engine() -> TextToSpeech:
    """Get Kokoro TTS engine only."""
    if not KOKORO_AVAILABLE:
        raise RuntimeError("Kokoro TTS is required! Install with: pip install kokoro")
    return TextToSpeech(voice="af_heart", speed=1.1)


if __name__ == "__main__":
    print("Testing TTS...")

    tts = get_tts_engine()

    test_phrases = [
        "Hello! Welcome to Askri Bank Limited.",
        "How can I help you today?"
    ]

    for phrase in test_phrases:
        print(f"Test: {phrase}")
        metrics = tts.speak(phrase)
        print(f"[{metrics.get('synthesis_ms', 0):.0f}ms]")