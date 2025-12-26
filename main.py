import os
import sys
import signal
import time
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Import modules
from stt import SpeechToText
from llm import BankingLLM
from tts import TextToSpeech, FallbackTTS, get_tts_engine, KOKORO_AVAILABLE


class LatencyTracker:
    """Track and display latency metrics."""

    def __init__(self):
        self.history = []
        self.current_turn = {}

    def start_turn(self):
        """Start tracking a new conversation turn."""
        self.current_turn = {
            "start_time": time.perf_counter(),
            "stt_ms": 0,
            "llm_ms": 0,
            "tts_synthesis_ms": 0,
            "tts_total_ms": 0,
            "total_processing_ms": 0,
            "total_turn_ms": 0
        }

    def record(self, component: str, latency_ms: float):
        """Record latency for a component."""
        self.current_turn[component] = latency_ms

    def end_turn(self):
        """End the current turn and calculate totals."""
        self.current_turn["total_turn_ms"] = (
            time.perf_counter() - self.current_turn["start_time"]
        ) * 1000

        self.current_turn["total_processing_ms"] = (
            self.current_turn["stt_ms"] +
            self.current_turn["llm_ms"] +
            self.current_turn["tts_synthesis_ms"]
        )

        self.history.append(self.current_turn.copy())
        return self.current_turn

    def display_metrics(self, metrics: dict):
        """Display formatted latency metrics."""
        stt = metrics.get('stt_ms', 0)
        llm = metrics.get('llm_ms', 0)
        tts = metrics.get('tts_synthesis_ms', 0)
        total = metrics.get('total_processing_ms', 0)
        print(f"\n[Latency] STT:{stt:.0f}ms LLM:{llm:.0f}ms TTS:{tts:.0f}ms Total:{total:.0f}ms")

    def display_session_summary(self):
        """Display session summary statistics."""
        if not self.history:
            return

        avg_stt = sum(t["stt_ms"] for t in self.history) / len(self.history)
        avg_llm = sum(t["llm_ms"] for t in self.history) / len(self.history)
        avg_tts = sum(t["tts_synthesis_ms"] for t in self.history) / len(self.history)
        avg_total = sum(t["total_processing_ms"] for t in self.history) / len(self.history)

        print(f"\nSession: {len(self.history)} turns | Avg: STT:{avg_stt:.0f}ms LLM:{avg_llm:.0f}ms TTS:{avg_tts:.0f}ms Total:{avg_total:.0f}ms")


class VoiceAgent:
    """
    Main voice agent with VAD, noise cancellation, and latency tracking.
    """

    def __init__(
        self,
        llm_model: str = "llama-3.1-8b-instant",
        tts_voice: str = "af_heart",
        show_latency: bool = True
    ):
        print("=" * 60)
        print("       ASKRI BANK LIMITED - VOICE AGENT")
        print("=" * 60)
        print("\nInitializing...\n")

        # Initialize STT with optimized settings for low latency
        self.stt = SpeechToText(
            vad_threshold=400,
            silence_duration=0.8,
            chunk_duration=1.5
        )
        print("✓ STT ready")

        # Initialize LLM
        self.llm = BankingLLM(model=llm_model)
        print("✓ LLM ready")

        # Initialize TTS
        self.tts = get_tts_engine()
        if isinstance(self.tts, TextToSpeech):
            self.tts.voice = tts_voice
        print("✓ TTS ready")

        # Latency tracking
        self.latency_tracker = LatencyTracker()
        self.show_latency = show_latency

        # State
        self.running = False
        self.session_start = None

        print("\n" + "=" * 60)
        print("All components initialized successfully!")
        print("=" * 60 + "\n")

    def start(self):
        """Start the voice agent."""
        self.running = True
        self.session_start = datetime.now()

        # Graceful shutdown handler
        signal.signal(signal.SIGINT, self._signal_handler)

        self._print_instructions()

        # Initial greeting
        greeting, _ = self.llm.get_greeting()
        print(f"\n🤖 Agent: {greeting}\n")
        self.tts.speak(greeting)

        # Main loop
        self._conversation_loop()

    def _print_instructions(self):
        """Print usage instructions."""
        print("╔" + "═" * 48 + "╗")
        print("║" + "  Voice Session Active".center(48) + "║")
        print("╠" + "═" * 48 + "╣")
        print("║  Say 'goodbye' or 'exit' to end              ║")
        print("╚" + "═" * 48 + "╝\n")

    def _conversation_loop(self):
        """Main conversation loop."""
        while self.running:
            try:
                self.latency_tracker.start_turn()

                # === STT Phase ===
                user_input, stt_metrics = self.stt.listen_once()

                self.latency_tracker.record("stt_ms", stt_metrics.get("stt_ms", 0))

                if not user_input:
                    continue

                print(f"\n👤 Customer: {user_input}")

                # Check for exit
                if self._check_exit(user_input):
                    self._end_session()
                    break

                # === LLM Phase ===
                response, llm_metrics = self.llm.get_response(user_input)
                self.latency_tracker.record("llm_ms", llm_metrics.get("latency_ms", 0))

                print(f"🤖 Agent: {response}")

                # === TTS Phase ===
                tts_metrics = self.tts.speak(response)
                self.latency_tracker.record("tts_synthesis_ms", tts_metrics.get("synthesis_ms", 0))
                self.latency_tracker.record("tts_total_ms", tts_metrics.get("total_ms", 0))

                # End turn and show metrics
                turn_metrics = self.latency_tracker.end_turn()

                if self.show_latency:
                    self.latency_tracker.display_metrics(turn_metrics)

            except KeyboardInterrupt:
                self._end_session()
                break
            except Exception as e:
                print(f"Error: {e}")
                self.tts.speak("Sorry, please repeat that.")

    def _check_exit(self, text: str) -> bool:
        """Check if user wants to exit."""
        exit_phrases = [
            'goodbye', 'bye', 'exit', 'quit', 'end call',
            'hang up', 'stop', "that's all", 'thanks bye',
            'thank you bye', 'end session', 'close'
        ]
        text_lower = text.lower().strip()
        return any(phrase in text_lower for phrase in exit_phrases)

    def _end_session(self):
        """End the session gracefully."""
        self.running = False

        farewell = "Thank you for contacting Askri Bank Limited. Have a wonderful day!"
        print(f"\n🤖 Agent: {farewell}")
        self.tts.speak(farewell)

        self.latency_tracker.display_session_summary()
        print("\nSession Ended")

    def _signal_handler(self, signum, frame):
        """Handle interrupt signal."""
        print("\nInterrupt received...")
        self._end_session()
        sys.exit(0)


def check_dependencies() -> bool:
    """Check required dependencies."""
    missing = []

    # Required packages
    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    try:
        import sounddevice
    except ImportError:
        missing.append("sounddevice")

    try:
        from groq import Groq
    except ImportError:
        missing.append("groq")

    try:
        from dotenv import load_dotenv
    except ImportError:
        missing.append("python-dotenv")

    try:
        import pyaudio
    except ImportError:
        missing.append("pyaudio")

    # Check API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not found! Set it in .env file.")
        return False

    if missing:
        print(f"Missing packages: pip install {' '.join(missing)}")
        return False

    return True


def check_audio_devices():
    """Check audio devices."""
    try:
        import sounddevice as sd
        default_input = sd.query_devices(kind='input')
        print(f"Microphone: {default_input['name']}")
        return True
    except Exception as e:
        print(f"Audio device error: {e}")
        return False


def run_latency_test():
    """Run a quick latency test of all components."""
    print("\nLatency Test\n")

    results = {}

    # Test STT initialization
    print("Testing STT...")
    start = time.perf_counter()
    try:
        from stt import SpeechToText
        stt = SpeechToText()
        results["stt_init"] = (time.perf_counter() - start) * 1000
        print(f"  STT init: {results['stt_init']:.1f} ms")
    except Exception as e:
        print(f"  STT failed: {e}")
        results["stt_init"] = -1

    # Test LLM
    print("Testing LLM...")
    try:
        from llm import BankingLLM
        llm = BankingLLM(model="llama-3.1-8b-instant")

        start = time.perf_counter()
        response, metrics = llm.get_response("What are your savings account rates?")
        results["llm_response"] = metrics.get("latency_ms", 0)
        print(f"  LLM response: {results['llm_response']:.1f} ms")
    except Exception as e:
        print(f"  LLM failed: {e}")
        results["llm_response"] = -1

    # Test TTS
    print("Testing TTS...")
    try:
        from tts import get_tts_engine
        tts = get_tts_engine()

        start = time.perf_counter()
        if hasattr(tts, 'synthesize'):
            audio, synth_time = tts.synthesize("Hello, this is a test.")
            results["tts_synthesis"] = synth_time
        else:
            results["tts_synthesis"] = 0
        print(f"  TTS synthesis: {results.get('tts_synthesis', 0):.1f} ms")
    except Exception as e:
        print(f"  TTS failed: {e}")
        results["tts_synthesis"] = -1

    # Summary
    total = sum(v for v in results.values() if v >= 0)
    print(f"\nEstimated round-trip: {total:.1f} ms\n")

    return results


def print_banner():
    """Print welcome banner."""
    print("\n" + "=" * 50)
    print("   ASKRI BANK LIMITED - VOICE AGENT")
    print("=" * 50 + "\n")


def print_help():
    """Print help information."""
    help_text = """
    Usage: python main.py [OPTIONS]
    
    Options:
        --help, -h          Show this help message
        --test              Run latency test only
        --no-latency        Disable latency display
        --model MODEL       LLM model (default: llama-3.1-8b-instant)
        --voice VOICE       TTS voice (default: af_heart)
    
    Examples:
        python main.py
        python main.py --test
        python main.py --no-latency
    
    Environment Variables:
        GROQ_API_KEY        Your Groq API key (required)
    """
    print(help_text)


def parse_args():
    """Parse command line arguments."""
    args = {
        "test": False,
        "show_latency": True,
        "llm_model": "llama-3.1-8b-instant",
        "tts_voice": "af_heart",
        "help": False
    }

    argv = sys.argv[1:]
    i = 0

    while i < len(argv):
        arg = argv[i]

        if arg in ["--help", "-h"]:
            args["help"] = True
        elif arg == "--test":
            args["test"] = True
        elif arg == "--no-latency":
            args["show_latency"] = False
        elif arg == "--model" and i + 1 < len(argv):
            i += 1
            args["llm_model"] = argv[i]
        elif arg == "--voice" and i + 1 < len(argv):
            i += 1
            args["tts_voice"] = argv[i]

        i += 1

    return args


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()

    # Print banner
    print_banner()

    # Handle help
    if args["help"]:
        print_help()
        sys.exit(0)

    # Check dependencies
    if not check_dependencies():
        print("\n⚠ Please install missing dependencies and try again.")
        sys.exit(1)

    # Check audio devices
    if not check_audio_devices():
        print("\n⚠ Please check your audio configuration.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

    # Run latency test only
    if args["test"]:
        run_latency_test()
        sys.exit(0)

    # Create and start the agent
    try:
        print("\nStarting Voice Agent...\n")

        agent = VoiceAgent(
            llm_model=args["llm_model"],
            tts_voice=args["tts_voice"],
            show_latency=args["show_latency"]
        )

        agent.start()

    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()