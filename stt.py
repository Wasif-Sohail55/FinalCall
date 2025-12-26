"""
Real-Time Streaming Speech-to-Text
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
    print("PyAudio not installed!  Run: pip install pyaudio")


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
            self.noise_samples. append(rms)
            if len(self.noise_samples) >= 10:
                self. noise_floor = np.percentile(list(self.noise_samples), 75) * 1.5

        dynamic_threshold = max(self.threshold, self.noise_floor * 3)
        is_speech = rms > dynamic_threshold or peak > dynamic_threshold * 2
        confidence = min(rms / max(dynamic_threshold, 1), 1.0)

        return is_speech, confidence

    def reset(self):
        self.noise_samples. clear()
        self.noise_floor = 200


class SpeechToText:
    def __init__(self, device_index=None, vad_threshold=500,
                 silence_duration=1.2, chunk_duration=2.0):

        if not PYAUDIO_AVAILABLE:
            raise RuntimeError("PyAudio required!")

        api_key = os. getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found!")

        self.client = Groq(api_key=api_key)
        self.model = "whisper-large-v3-turbo"
        self.sample_rate = 16000
        self. channels = 1
        self.chunk_size = 1024
        self.silence_duration = silence_duration
        self. chunk_duration = chunk_duration

        self.pa = pyaudio. PyAudio()

        if device_index is None:
            info = self.pa. get_default_input_device_info()
            self.device_index = int(info['index'])
            self.device_name = info['name']
        else:
            self.device_index = device_index
            info = self.pa. get_device_info_by_index(device_index)
            self.device_name = info['name']

        print("Microphone:  " + self.device_name)

        self.vad = EnergyVAD(threshold=vad_threshold, sample_rate=self.sample_rate)
        self.stream = None
        self.is_running = False

    def __del__(self):
        self.close()

    def close(self):
        self.is_running = False
        if self.stream:
            try:
                self. stream.stop_stream()
                self. stream.close()
            except:
                pass
            self.stream = None
        if hasattr(self, 'pa'):
            try:
                self. pa.terminate()
            except:
                pass

    def open_stream(self):
        if self.stream is None:
            self. stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size
            )

    def close_stream(self):
        if self. stream:
            try:
                self. stream.stop_stream()
                self. stream.close()
            except:
                pass
            self. stream = None

    def create_wav_bytes(self, audio):
        """Create WAV file in memory."""
        if len(audio) == 0:
            return None

        if audio.dtype != np. int16:
            audio = audio.astype(np.int16)

        buffer = io.BytesIO()

        try:
            wf = wave.open(buffer, 'wb')
            wf.setnchannels(self. channels)
            wf.setsampwidth(2)
            wf.setframerate(self. sample_rate)
            wf.writeframes(audio.tobytes())
            wf.close()
        except Exception as e:
            print("WAV creation error:  " + str(e))
            return None

        buffer.seek(0)
        return buffer

    def transcribe(self, audio):
        """Send audio to Groq Whisper for transcription."""
        if len(audio) < self.sample_rate * 0.3:
            return ""

        # Normalize
        peak = np.max(np.abs(audio))
        if peak > 100:
            audio = (audio. astype(np. float32) / peak * 30000).astype(np.int16)

        wav_bytes = self. create_wav_bytes(audio)
        if wav_bytes is None:
            return ""

        try:
            result = self.client. audio.transcriptions.create(
                file=("recording.wav", wav_bytes, "audio/wav"),
                model=self.model,
                language="en",
                response_format="text"
            )
            return result.strip()
        except Exception as e:
            print("Transcription error: " + str(e))
            return ""

    def record_utterance(self, max_duration=60.0):
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

        print("   Waiting for speech...")

        for i in range(max_chunks):
            try:
                data = self.stream. read(self.chunk_size, exception_on_overflow=False)
                chunk = np.frombuffer(data, dtype=np.int16)
            except Exception as e:
                print("Read error: " + str(e))
                continue

            is_speech, conf = self.vad.is_speech(chunk)

            # Visual feedback
            level = np.max(np.abs(chunk)) / 32768.0
            bars = int(level * 30)
            meter = "#" * bars + "-" * (30 - bars)
            status = "SPEAK" if is_speech else "     "

            if not speech_active:
                pre_buffer.append(chunk. copy())
                duration = 0.0
                print("\r   [" + status + "] [" + meter + "]", end="", flush=True)

                if is_speech:
                    speech_active = True
                    audio_buffer = list(pre_buffer)
                    speech_count = 1
                    print("\r   Speech started!                               ")
            else:
                audio_buffer.append(chunk.copy())
                duration = len(audio_buffer) * chunk_time
                remain = max(0, (silence_chunks - silent_count) * chunk_time)

                info = str(round(duration, 1)) + "s"
                if not is_speech:
                    info = info + " | silence " + str(round(remain, 1)) + "s"

                print("\r   [" + status + "] [" + meter + "] " + info + "    ", end="", flush=True)

                if is_speech:
                    speech_count += 1
                    silent_count = 0
                else:
                    silent_count += 1
                    if silent_count >= silence_chunks:
                        print("\r   End of speech                                    ")
                        break

        self.close_stream()

        if not audio_buffer:
            print("   No speech detected")
            return None, {}

        audio = np.concatenate(audio_buffer)
        duration = len(audio) / self.sample_rate

        print("   Recorded " + str(round(duration, 1)) + "s")

        metrics = {
            "audio_duration": duration,
            "speech_chunks": speech_count
        }

        return audio, metrics

    def listen_once(self):
        """Listen for one utterance and transcribe."""
        print("Listening... (speak now)")

        audio, metrics = self. record_utterance(max_duration=60.0)

        if audio is None:
            return "", {}

        print("   Transcribing...")
        start = time.perf_counter()
        text = self. transcribe(audio)
        stt_time = (time.perf_counter() - start) * 1000

        metrics["stt_ms"] = stt_time

        return text, metrics

    def listen_realtime(self, on_text=None, stop_phrase="stop listening"):
        """Real-time transcription with periodic updates."""

        print("\n" + "=" * 50)
        print("REAL-TIME MODE")
        print("Say '" + stop_phrase + "' to stop")
        print("=" * 50 + "\n")

        self.is_running = True
        self.open_stream()
        self.vad.reset()

        audio_buffer = []
        pre_buffer = deque(maxlen=10)

        chunk_time = self.chunk_size / self.sample_rate
        silence_chunks = int(self.silence_duration / chunk_time)
        update_chunks = int(self. chunk_duration / chunk_time)

        speech_active = False
        silent_count = 0
        chunks_since_update = 0
        current_text = ""

        print("Listening.. .\n")

        try:
            while self.is_running:
                try:
                    data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    chunk = np.frombuffer(data, dtype=np.int16)
                except:
                    continue

                is_speech, _ = self.vad.is_speech(chunk)

                level = np.max(np.abs(chunk)) / 32768.0
                bars = int(level * 25)
                meter = "|" * bars

                if is_speech:
                    marker = ">>>"
                else:
                    marker = "   "

                print("\r" + marker + " [" + meter. ljust(25) + "]", end="", flush=True)

                if is_speech:
                    if not speech_active:
                        speech_active = True
                        audio_buffer = list(pre_buffer)
                        chunks_since_update = 0
                        print("\r[Speaking...]" + " " * 30)

                    silent_count = 0
                    audio_buffer.append(chunk.copy())
                    chunks_since_update += 1

                    # Periodic transcription
                    if chunks_since_update >= update_chunks and len(audio_buffer) > 0:
                        temp_audio = np. concatenate(audio_buffer)
                        partial = self.transcribe(temp_audio)
                        if partial:
                            current_text = partial
                            print("\r   > " + current_text[: 60] + " " * 10)
                        chunks_since_update = 0

                elif speech_active:
                    audio_buffer. append(chunk.copy())
                    silent_count += 1

                    if silent_count >= silence_chunks:
                        # Final transcription
                        if audio_buffer:
                            full_audio = np. concatenate(audio_buffer)
                            final_text = self. transcribe(full_audio)

                            if final_text:
                                print("\r>> " + final_text + " " * 20 + "\n")

                                if on_text:
                                    on_text(final_text)

                                if stop_phrase. lower() in final_text.lower():
                                    print("Stop phrase detected.")
                                    self.is_running = False
                                    break

                        # Reset
                        audio_buffer = []
                        speech_active = False
                        silent_count = 0
                        chunks_since_update = 0
                        current_text = ""
                        print("\nListening.. .\n")
                else:
                    pre_buffer.append(chunk.copy())

        except KeyboardInterrupt:
            print("\n\nStopped by user.")

        self.close_stream()

    def listen_continuous(self, on_text=None, stop_phrase="goodbye"):
        """Continuous listening mode."""

        print("\n" + "=" * 50)
        print("CONTINUOUS MODE")
        print("Say '" + stop_phrase + "' to stop")
        print("=" * 50 + "\n")

        self.is_running = True
        turn = 0

        while self.is_running:
            turn += 1
            print("[Turn " + str(turn) + "]")

            text, metrics = self. listen_once()

            if text:
                print("\n>> " + text)
                stt_ms = metrics.get("stt_ms", 0)
                print("[" + str(int(stt_ms)) + "ms]\n")

                if on_text:
                    on_text(text)

                if stop_phrase. lower() in text.lower():
                    print("Stop phrase detected.  Ending.")
                    self. is_running = False
                    break
            else:
                print("(no speech)\n")


def list_devices():
    p = pyaudio. PyAudio()
    print("\nInput Devices:")
    print("-" * 40)

    default = p.get_default_input_device_info()['index']

    for i in range(p.get_device_count()):
        d = p.get_device_info_by_index(i)
        if d['maxInputChannels'] > 0:
            mark = " <-- DEFAULT" if i == default else ""
            print("  [" + str(i) + "] " + d['name'] + mark)

    print("-" * 40)
    p.terminate()


def main():
    print("=" * 50)
    print("REAL-TIME SPEECH TO TEXT")
    print("=" * 50)

    if not PYAUDIO_AVAILABLE:
        print("Install PyAudio: pip install pyaudio")
        return

    list_devices()

    print("\nDevice number (Enter for default): ", end="")
    try:
        choice = input().strip()
        dev = int(choice) if choice else None
    except:
        dev = None

    try:
        stt = SpeechToText(
            device_index=dev,
            vad_threshold=500,
            silence_duration=1.2,
            chunk_duration=2.0
        )
    except Exception as e:
        print("Init error: " + str(e))
        return

    print("\nMode:")
    print("  1 = Single utterance")
    print("  2 = Real-time streaming")
    print("  3 = Continuous")
    print("\nChoice: ", end="")

    try:
        mode = input().strip()
    except:
        mode = "1"

    if mode == "2":
        stt.listen_realtime(stop_phrase="stop listening")

    elif mode == "3":
        stt.listen_continuous(stop_phrase="goodbye")

    else:
        print("\n" + "=" * 50)
        print("Speak now (no time limit):")
        print("=" * 50 + "\n")

        text, metrics = stt.listen_once()

        print("\n" + "=" * 50)
        if text:
            print("Result:  " + text)
        else:
            print("No transcription")
        print("=" * 50)

        stt_ms = metrics.get("stt_ms", 0)
        duration = metrics.get("audio_duration", 0)
        print("STT: " + str(int(stt_ms)) + "ms")
        print("Audio:  " + str(round(duration, 1)) + "s")


if __name__ == "__main__":
    main()