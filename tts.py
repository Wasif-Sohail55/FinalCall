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

# Fallback TTS
PYTTSX3_AVAILABLE = False
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    pass


class TextToSpeech:
    """
    Kokoro TTS with streaming support for low latency.
    """

    def __init__(
        self,
        voice:  str = "af_heart",
        speed: float = 1.0,
        sample_rate: int = 24000
    ):
        """
        Initialize Kokoro TTS.

        Args:
            voice:  Voice model.  Options:
                - 'af_heart' (American Female - warm)
                - 'af_bella' (American Female - professional)
                - 'af_sarah' (American Female - friendly)
                - 'am_adam' (American Male)
                - 'am_michael' (American Male - deep)
                - 'bf_emma' (British Female)
                - 'bm_george' (British Male)
            speed: Speech rate (0.5 - 2.0)
            sample_rate: Audio sample rate
        """
        self.voice = voice
        self.speed = speed
        self.sample_rate = sample_rate
        self.pipeline = None

        # Latency tracking
        self.last_synthesis_latency_ms = 0
        self.last_total_latency_ms = 0

        # Playback control
        self._stop_event = threading. Event()
        self._playback_thread = None

        if KOKORO_AVAILABLE:
            try:
                # Initialize with American English
                self.pipeline = KPipeline(lang_code='a')
                print(f"✓ Kokoro TTS initialized (voice: {voice})")
            except Exception as e:
                print(f"⚠ Kokoro init failed: {e}")
                self.pipeline = None
        else:
            print("⚠ Kokoro not available")

    def synthesize(self, text: str) -> Tuple[np.ndarray, float]:
        """
        Synthesize text to audio.

        Args:
            text: Text to synthesize

        Returns:
            Tuple of (audio_array, synthesis_time_ms)
        """
        if not text or not text.strip():
            return np.array([], dtype=np. float32), 0

        if self.pipeline is None:
            print(f"[TTS]:  {text}")
            return np.array([], dtype=np.float32), 0

        start_time = time. perf_counter()

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
            print(f"TTS synthesis error: {e}")
            return np. array([], dtype=np.float32), 0

    def synthesize_streaming(self, text:  str) -> Generator[np.ndarray, None, None]:
        """
        Stream audio chunks for lower latency.

        Yields:
            Audio chunks as they're generated
        """
        if not text or self. pipeline is None:
            return

        try:
            for _, _, audio in self. pipeline(
                text,
                voice=self.voice,
                speed=self. speed
            ):
                if audio is not None:
                    yield audio
        except Exception as e:
            print(f"TTS streaming error:  {e}")

    def speak(self, text:  str, blocking: bool = True) -> dict:
        """
        Synthesize and play text.

        Args:
            text: Text to speak
            blocking: Wait for playback to complete

        Returns:
            Timing metrics
        """
        if not text or not text. strip():
            return {"error": "Empty text"}

        total_start = time.perf_counter()

        # Stop any ongoing playback
        self. stop()
        self._stop_event.clear()

        print(f"🔊 Speaking:  {text[: 60]}..." if len(text) > 60 else f"🔊 Speaking: {text}")

        # Synthesize
        audio, synth_time = self.synthesize(text)

        metrics = {
            "synthesis_ms": synth_time,
            "audio_duration_ms":  0,
            "total_ms": 0
        }

        if len(audio) > 0:
            metrics["audio_duration_ms"] = len(audio) / self.sample_rate * 1000

            # Play audio
            if blocking:
                self._play_blocking(audio)
            else:
                self._play_async(audio)
        else:
            # Fallback:  print text
            print(f"   [Voice]: {text}")

        metrics["total_ms"] = (time.perf_counter() - total_start) * 1000
        self.last_total_latency_ms = metrics["total_ms"]

        return metrics

    def speak_streaming(self, text: str) -> dict:
        """
        Speak with streaming for lowest latency.
        Starts playing audio while still synthesizing.

        Args:
            text: Text to speak

        Returns:
            Timing metrics
        """
        if not text or self.pipeline is None:
            return self.speak(text)

        total_start = time. perf_counter()
        first_chunk_time = None

        self.stop()
        self._stop_event.clear()

        print(f"🔊 Speaking (streaming): {text[:50]}...")

        audio_queue = queue.Queue()

        # Producer thread - synthesize audio
        def producer():
            nonlocal first_chunk_time
            for chunk in self.synthesize_streaming(text):
                if self._stop_event. is_set():
                    break
                if first_chunk_time is None:
                    first_chunk_time = (time.perf_counter() - total_start) * 1000
                audio_queue.put(chunk)
            audio_queue.put(None)  # Signal end

        # Consumer thread - play audio
        def consumer():
            while not self._stop_event.is_set():
                try:
                    chunk = audio_queue. get(timeout=0.1)
                    if chunk is None:
                        break
                    sd.play(chunk, samplerate=self. sample_rate)
                    sd.wait()
                except queue.Empty:
                    continue

        prod_thread = threading. Thread(target=producer)
        cons_thread = threading.Thread(target=consumer)

        prod_thread.start()
        cons_thread.start()

        prod_thread.join()
        cons_thread.join()

        total_time = (time.perf_counter() - total_start) * 1000

        return {
            "first_chunk_ms":  first_chunk_time or 0,
            "total_ms":  total_time
        }

    def _play_blocking(self, audio: np. ndarray):
        """Play audio and wait for completion."""
        try:
            sd. play(audio, samplerate=self. sample_rate)
            sd.wait()
        except Exception as e:
            print(f"Playback error: {e}")

    def _play_async(self, audio: np.ndarray):
        """Play audio in background."""
        def play():
            try:
                sd.play(audio, samplerate=self.sample_rate)
                sd.wait()
            except Exception as e:
                print(f"Playback error: {e}")

        self._playback_thread = threading.Thread(target=play)
        self._playback_thread.start()

    def stop(self):
        """Stop current playback."""
        self._stop_event. set()
        sd.stop()
        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread. join(timeout=0.5)

    def set_voice(self, voice:  str):
        """Change voice."""
        self.voice = voice

    def set_speed(self, speed: float):
        """Change speed (0.5-2.0)."""
        self.speed = max(0.5, min(2.0, speed))


class FallbackTTS:
    """
    Fallback TTS using pyttsx3 for when Kokoro isn't available.
    """

    def __init__(self, rate: int = 175):
        self.engine = None
        self.last_synthesis_latency_ms = 0
        self.last_total_latency_ms = 0

        if PYTTSX3_AVAILABLE:
            try:
                self. engine = pyttsx3.init()
                self.engine.setProperty('rate', rate)

                # Try to find a good voice
                voices = self.engine.getProperty('voices')
                for voice in voices:
                    if 'female' in voice.name.lower():
                        self. engine.setProperty('voice', voice.id)
                        break

                print("✓ Fallback TTS (pyttsx3) initialized")
            except Exception as e:
                print(f"⚠ pyttsx3 init failed: {e}")
                self.engine = None
        else:
            print("⚠ No TTS engine available")

    def speak(self, text:  str, blocking: bool = True) -> dict:
        """Speak text."""
        start_time = time. perf_counter()

        if not text:
            return {"error": "Empty text"}

        print(f"🔊 Speaking: {text[:60]}...")

        if self.engine:
            try:
                self.engine.say(text)
                if blocking:
                    self.engine.runAndWait()
            except Exception as e:
                print(f"TTS Error: {e}")
                print(f"[Voice]: {text}")
        else:
            print(f"[Voice]:  {text}")

        total_time = (time.perf_counter() - start_time) * 1000
        self.last_total_latency_ms = total_time

        return {"total_ms": total_time}

    def stop(self):
        """Stop speaking."""
        if self.engine:
            try:
                self. engine.stop()
            except:
                pass

    def set_speed(self, speed: float):
        """Set speech rate."""
        if self.engine:
            rate = int(175 * speed)
            self.engine. setProperty('rate', rate)


def get_tts_engine() -> TextToSpeech:
    """Get the best available TTS engine."""
    if KOKORO_AVAILABLE:
        return TextToSpeech(voice="af_heart", speed=1.0)
    elif PYTTSX3_AVAILABLE:
        return FallbackTTS()
    else:
        return FallbackTTS()  # Will just print text


# Test module
if __name__ == "__main__":
    print("=" * 60)
    print("Testing TTS Module")
    print("=" * 60)

    tts = get_tts_engine()

    test_phrases = [
        "Hello!  Welcome to XYZ Bank.",
        "I'd be happy to help you with your account today.",
        "Your current balance can be viewed in our mobile app."
    ]

    for phrase in test_phrases:
        print(f"\nTest: {phrase}")
        metrics = tts.speak(phrase)
        print(f"Metrics: {metrics}")