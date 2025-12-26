"""
Real-Time Speech-to-Text with optimized latency
"""

import os
import io
import wave
import time
from collections import deque
import numpy as np
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False


class EnergyVAD:
    def __init__(self, threshold, sample_rate):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.noise_floor = 200
        self.noise_samples = deque(maxlen=30)

    def is_speech(self, audio_chunk):
        audio = audio_chunk.flatten().astype(np.float32)
        if len(audio) == 0:
            return False, 0.0

        rms = np.sqrt(np.mean(audio ** 2))
        peak = np.max(np.abs(audio))

        if rms < self.threshold * 0.5:
            self.noise_samples.append(rms)
            if len(self.noise_samples) >= 10:
                self.noise_floor = np.percentile(list(self.noise_samples), 75) * 1.5

        dynamic_threshold = max(self.threshold, self.noise_floor * 3)
        is_speech = rms > dynamic_threshold or peak > dynamic_threshold * 2
        confidence = min(rms / max(dynamic_threshold, 1), 1.0)

        return is_speech, confidence

    def reset(self):
        self.noise_samples.clear()
        self.noise_floor = 200


class SpeechToText:
    def __init__(self, device_index=None, vad_threshold=400,
                 silence_duration=0.8, chunk_duration=1.5):

        if not PYAUDIO_AVAILABLE:
            raise RuntimeError("PyAudio required!")

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found!")

        self.client = Groq(api_key=api_key)
        self.model = "whisper-large-v3-turbo"
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1024
        self.silence_duration = silence_duration
        self.chunk_duration = chunk_duration

        self.pa = pyaudio.PyAudio()

        if device_index is None:
            info = self.pa.get_default_input_device_info()
            self.device_index = int(info['index'])
            self.device_name = info['name']
        else:
            self.device_index = device_index
            info = self.pa.get_device_info_by_index(device_index)
            self.device_name = info['name']

        self.vad = EnergyVAD(threshold=vad_threshold, sample_rate=self.sample_rate)
        self.stream = None
        self.is_running = False

    def __del__(self):
        self.close()

    def close(self):
        self.is_running = False
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
            self.stream = None
        if hasattr(self, 'pa'):
            try:
                self.pa.terminate()
            except:
                pass

    def open_stream(self):
        if self.stream is None:
            self.stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size
            )

    def close_stream(self):
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
            self.stream = None

    def create_wav_bytes(self, audio):
        """Create WAV file in memory."""
        if len(audio) == 0:
            return None

        if audio.dtype != np.int16:
            audio = audio.astype(np.int16)

        buffer = io.BytesIO()

        try:
            wf = wave.open(buffer, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio.tobytes())
            wf.close()
        except Exception as e:
            return None

        buffer.seek(0)
        return buffer

    def transcribe(self, audio):
        """Send audio to Groq Whisper for transcription."""
        if len(audio) < self.sample_rate * 0.3:
            return ""

        peak = np.max(np.abs(audio))
        if peak > 100:
            audio = (audio.astype(np.float32) / peak * 30000).astype(np.int16)

        wav_bytes = self.create_wav_bytes(audio)
        if wav_bytes is None:
            return ""

        try:
            result = self.client.audio.transcriptions.create(
                file=("recording.wav", wav_bytes, "audio/wav"),
                model=self.model,
                language="en",
                response_format="text"
            )
            return result.strip()
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""

    def record_utterance(self, max_duration=30.0):
        """Record a single utterance until silence."""

        self.open_stream()
        self.vad.reset()

        audio_buffer = []
        pre_buffer = deque(maxlen=10)

        samples_per_chunk = self.chunk_size
        chunk_time = samples_per_chunk / self.sample_rate
        max_chunks = int(max_duration / chunk_time)
        silence_chunks = int(self.silence_duration / chunk_time)

        speech_active = False
        silent_count = 0
        speech_count = 0

        print("Listening...")

        for i in range(max_chunks):
            try:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                chunk = np.frombuffer(data, dtype=np.int16)
            except Exception as e:
                continue

            is_speech, conf = self.vad.is_speech(chunk)

            if not speech_active:
                pre_buffer.append(chunk.copy())

                if is_speech:
                    speech_active = True
                    audio_buffer = list(pre_buffer)
                    speech_count = 1
            else:
                audio_buffer.append(chunk.copy())

                if is_speech:
                    speech_count += 1
                    silent_count = 0
                else:
                    silent_count += 1
                    if silent_count >= silence_chunks:
                        break

        self.close_stream()

        if not audio_buffer:
            return None, {}

        audio = np.concatenate(audio_buffer)
        duration = len(audio) / self.sample_rate

        metrics = {
            "audio_duration": duration,
            "speech_chunks": speech_count
        }

        return audio, metrics

    def listen_once(self):
        """Listen for one utterance and transcribe."""

        audio, metrics = self.record_utterance(max_duration=30.0)

        if audio is None:
            return "", {}

        start = time.perf_counter()
        text = self.transcribe(audio)
        stt_time = (time.perf_counter() - start) * 1000

        metrics["stt_ms"] = stt_time

        return text, metrics

    def listen_realtime(self, on_text=None, stop_phrase="stop listening"):
        """Real-time transcription with periodic updates."""

        self.is_running = True
        self.open_stream()
        self.vad.reset()

        audio_buffer = []
        pre_buffer = deque(maxlen=10)

        chunk_time = self.chunk_size / self.sample_rate
        silence_chunks = int(self.silence_duration / chunk_time)
        update_chunks = int(self.chunk_duration / chunk_time)

        speech_active = False
        silent_count = 0
        chunks_since_update = 0

        try:
            while self.is_running:
                try:
                    data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    chunk = np.frombuffer(data, dtype=np.int16)
                except:
                    continue

                is_speech, _ = self.vad.is_speech(chunk)

                if is_speech:
                    if not speech_active:
                        speech_active = True
                        audio_buffer = list(pre_buffer)
                        chunks_since_update = 0

                    silent_count = 0
                    audio_buffer.append(chunk.copy())
                    chunks_since_update += 1

                    # Periodic transcription
                    if chunks_since_update >= update_chunks and len(audio_buffer) > 0:
                        temp_audio = np.concatenate(audio_buffer)
                        partial = self.transcribe(temp_audio)
                        chunks_since_update = 0

                elif speech_active:
                    audio_buffer.append(chunk.copy())
                    silent_count += 1

                    if silent_count >= silence_chunks:
                        if audio_buffer:
                            full_audio = np.concatenate(audio_buffer)
                            final_text = self.transcribe(full_audio)

                            if final_text:
                                if on_text:
                                    on_text(final_text)

                                if stop_phrase.lower() in final_text.lower():
                                    self.is_running = False
                                    break

                        audio_buffer = []
                        speech_active = False
                        silent_count = 0
                        chunks_since_update = 0
                else:
                    pre_buffer.append(chunk.copy())

        except KeyboardInterrupt:
            pass

        self.close_stream()

    def listen_continuous(self, on_text=None, stop_phrase="goodbye"):
        """Continuous listening mode."""

        self.is_running = True

        while self.is_running:
            text, metrics = self.listen_once()

            if text:
                if on_text:
                    on_text(text)

                if stop_phrase.lower() in text.lower():
                    self.is_running = False
                    break


def list_devices():
    p = pyaudio.PyAudio()
    print("Input Devices:")

    default = p.get_default_input_device_info()['index']

    for i in range(p.get_device_count()):
        d = p.get_device_info_by_index(i)
        if d['maxInputChannels'] > 0:
            mark = " <-- DEFAULT" if i == default else ""
            print(f"  [{i}] {d['name']}{mark}")

    p.terminate()


def main():
    print("Speech-to-Text Test")

    if not PYAUDIO_AVAILABLE:
        print("Install PyAudio: pip install pyaudio")
        return

    list_devices()

    try:
        stt = SpeechToText()
    except Exception as e:
        print(f"Init error: {e}")
        return

    text, metrics = stt.listen_once()

    if text:
        print(f"Result: {text}")
        print(f"[{metrics.get('stt_ms', 0):.0f}ms]")
    else:
        print("No transcription")


if __name__ == "__main__":
    main()