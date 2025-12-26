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
            "vad_ms": 0,
            "denoise_ms": 0,
            "stt_ms": 0,
            "llm_ms": 0,
            "tts_synthesis_ms": 0,
            "tts_total_ms":  0,
            "total_processing_ms": 0,
            "total_turn_ms": 0
        }

    def record(self, component: str, latency_ms: float):
        """Record latency for a component."""
        self. current_turn[component] = latency_ms

    def end_turn(self):
        """End the current turn and calculate totals."""
        self.current_turn["total_turn_ms"] = (
            time.perf_counter() - self.current_turn["start_time"]
        ) * 1000

        self.current_turn["total_processing_ms"] = (
            self.current_turn["vad_ms"] +
            self.current_turn["denoise_ms"] +
            self.current_turn["stt_ms"] +
            self.current_turn["llm_ms"] +
            self.current_turn["tts_synthesis_ms"]
        )

        self.history.append(self.current_turn.copy())
        return self.current_turn

    def display_metrics(self, metrics: dict):
        """Display formatted latency metrics."""
        print("\n┌─────────────────────────────────────┐")
        print("│         LATENCY BREAKDOWN           │")
        print("├─────────────────────────────────────┤")
        print(f"│  VAD Processing:       {metrics. get('vad_ms', 0):>7. 1f} ms   │")
        print(f"│  Noise Reduction:     {metrics.get('denoise_ms', 0):>7.1f} ms   │")
        print(f"│  STT (Groq):          {metrics.get('stt_ms', 0):>7.1f} ms   │")
        print(f"│  LLM (Groq):          {metrics.get('llm_ms', 0):>7.1f} ms   │")
        print(f"│  TTS Synthesis:       {metrics.get('tts_synthesis_ms', 0):>7.1f} ms   │")
        print("├─────────────────────────────────────┤")
        print(f"│  Total Processing:     {metrics.get('total_processing_ms', 0):>7.1f} ms   │")
        print(f"│  Total Turn Time:     {metrics. get('total_turn_ms', 0):>7.1f} ms   │")
        print("└─────────────────────────────────────┘")

    def display_session_summary(self):
        """Display session summary statistics."""
        if not self.history:
            return

        avg_processing = sum(t["total_processing_ms"] for t in self.history) / len(self.history)
        avg_stt = sum(t["stt_ms"] for t in self.history) / len(self.history)
        avg_llm = sum(t["llm_ms"] for t in self.history) / len(self.history)
        avg_tts = sum(t["tts_synthesis_ms"] for t in self.history) / len(self. history)

        min_processing = min(t["total_processing_ms"] for t in self.history)
        max_processing = max(t["total_processing_ms"] for t in self.history)

        print("\n" + "=" * 50)
        print("           SESSION LATENCY SUMMARY")
        print("=" * 50)
        print(f"  Total Turns:             {len(self. history)}")
        print(f"  Avg STT Latency:        {avg_stt:. 1f} ms")
        print(f"  Avg LLM Latency:         {avg_llm:.1f} ms")
        print(f"  Avg TTS Latency:        {avg_tts:.1f} ms")
        print("-" * 50)
        print(f"  Avg Total Processing:   {avg_processing:.1f} ms")
        print(f"  Min Processing:          {min_processing:.1f} ms")
        print(f"  Max Processing:         {max_processing:.1f} ms")
        print("=" * 50)


class VoiceAgent:
    """
    Main voice agent with VAD, noise cancellation, and latency tracking.
    """

    def __init__(
        self,
        vad_type: str = "silero",
        enable_noise_reduction: bool = True,
        llm_model: str = "llama-3.1-8b-instant",
        tts_voice: str = "af_heart",
        show_latency: bool = True
    ):
        print("=" * 60)
        print("       XYZ BANK VOICE CUSTOMER SUPPORT AGENT")
        print("       Enhanced with VAD & Noise Cancellation")
        print("=" * 60)
        print("\nInitializing components...\n")

        # Initialize STT with VAD and noise cancellation
        self. stt = SpeechToText(
            vad_type=vad_type,
            enable_noise_reduction=enable_noise_reduction,
            vad_threshold=0.5,
            silence_duration=1.0,
            max_duration=30.0
        )
        print("✓ STT ready (Groq Whisper + Silero VAD + Noise Reduction)")

        # Initialize LLM
        self.llm = BankingLLM(model=llm_model)
        print(f"✓ LLM ready (Groq {llm_model})")

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
        signal.signal(signal. SIGINT, self._signal_handler)

        self._print_instructions()

        # Initial greeting
        greeting, _ = self.llm.get_greeting()
        print(f"\n🤖 Agent: {greeting}\n")
        self. tts.speak(greeting)

        # Main loop
        self._conversation_loop()

    def _print_instructions(self):
        """Print usage instructions."""
        print("╔" + "═" * 58 + "╗")
        print("║" + " " * 15 + "VOICE SESSION STARTED" + " " * 22 + "║")
        print("╠" + "═" * 58 + "╣")
        print("║  Features:                                                ║")
        print("║  • Voice Activity Detection (Silero VAD)                 ║")
        print("║  • Noise Cancellation enabled                            ║")
        print("║  • Real-time latency tracking                            ║")
        print("╠" + "═" * 58 + "╣")
        print("║  Commands:                                               ║")
        print("║  • Say 'goodbye' or 'exit' to end                        ║")
        print("║  • Wait for the beep/prompt before speaking              ║")
        print("╚" + "═" * 58 + "╝\n")

    def _conversation_loop(self):
        """Main conversation loop."""
        while self.running:
            try:
                # Start tracking this turn
                self. latency_tracker. start_turn()

                print("\n" + "-" * 50)

                # === STT Phase ===
                user_input, stt_metrics = self.stt.listen_and_transcribe()

                self.latency_tracker.record("vad_ms", stt_metrics.get("vad_ms", 0))
                self. latency_tracker. record("denoise_ms", stt_metrics.get("denoise_ms", 0))
                self.latency_tracker.record("stt_ms", stt_metrics.get("stt_ms", 0))

                if not user_input:
                    print("   (No speech detected)")
                    continue

                print(f"\n👤 Customer: {user_input}")

                # Check for exit
                if self._check_exit(user_input):
                    self._end_session()
                    break

                # === LLM Phase ===
                print("   Processing...")
                response, llm_metrics = self.llm.get_response(user_input)
                self.latency_tracker.record("llm_ms", llm_metrics.get("latency_ms", 0))

                print(f"\n🤖 Agent: {response}")

                # === TTS Phase ===
                tts_metrics = self.tts.speak(response)
                self.latency_tracker.record("tts_synthesis_ms", tts_metrics.get("synthesis_ms", 0))
                self.latency_tracker.record("tts_total_ms", tts_metrics.get("total_ms", 0))

                # End turn and show metrics
                turn_metrics = self. latency_tracker. end_turn()

                if self.show_latency:
                    self.latency_tracker.display_metrics(turn_metrics)

            except KeyboardInterrupt:
                self._end_session()
                break
            except Exception as e:
                print(f"\n⚠ Error:  {e}")
                import traceback
                traceback.print_exc()
                self.tts.speak("I'm sorry, I encountered an issue.  Could you please repeat that?")

    def _check_exit(self, text: str) -> bool:
        """Check if user wants to exit."""
        exit_phrases = [
            'goodbye', 'bye', 'exit', 'quit', 'end call',
            'hang up', 'stop', "that's all", 'thanks bye',
            'thank you bye', 'end session', 'close'
        ]
        text_lower = text. lower().strip()
        return any(phrase in text_lower for phrase in exit_phrases)

    def _end_session(self):
        """End the session gracefully."""
        self.running = False

        farewell = "Thank you for contacting XYZ Bank!  Have a wonderful day!"
        print(f"\n🤖 Agent: {farewell}")
        self.tts.speak(farewell)

        # Show session summary
        self.latency_tracker.display_session_summary()

        # Session duration
        if self.session_start:
            duration = datetime.now() - self.session_start
            print(f"\nSession Duration: {duration}")

        print("\n" + "=" * 60)
        print("           SESSION ENDED")
        print("=" * 60 + "\n")

    def _signal_handler(self, signum, frame):
        """Handle interrupt signal."""
        print("\n\n⚠ Interrupt received...")
        self._end_session()
        sys.exit(0)


def check_dependencies() -> bool:
    """Check required dependencies."""
    print("Checking dependencies.. .\n")

    missing = []
    warnings = []

    # Required packages
    try:
        import numpy
        print("  ✓ numpy")
    except ImportError:
        missing. append("numpy")
        print("  ✗ numpy")

    try:
        import sounddevice
        print("  ✓ sounddevice")
    except ImportError:
        missing.append("sounddevice")
        print("  ✗ sounddevice")

    try:
        from groq import Groq
        print("  ✓ groq")
    except ImportError:
        missing. append("groq")
        print("  ✗ groq")

    try:
        from dotenv import load_dotenv
        print("  ✓ python-dotenv")
    except ImportError:
        missing.append("python-dotenv")
        print("  ✗ python-dotenv")

    # Optional but recommended
    try:
        import torch
        print("  ✓ torch (for Silero VAD)")
    except ImportError:
        warnings.append("torch (Silero VAD will use fallback)")
        print("  ⚠ torch (optional - for Silero VAD)")

    try:
        import noisereduce
        print("  ✓ noisereduce")
    except ImportError:
        warnings.append("noisereduce (will use basic noise gate)")
        print("  ⚠ noisereduce (optional - for better noise reduction)")

    try:
        import webrtcvad
        print("  ✓ webrtcvad")
    except ImportError:
        print("  ⚠ webrtcvad (optional - alternative VAD)")

    # TTS
    try:
        from kokoro import KPipeline
        print("  ✓ kokoro (TTS)")
    except ImportError:
        try:
            import pyttsx3
            print("  ⚠ kokoro not found, using pyttsx3 fallback")
            warnings.append("kokoro (using pyttsx3 fallback)")
        except ImportError:
            warnings.append("No TTS engine (kokoro or pyttsx3)")
            print("  ⚠ No TTS engine found")

    # Check API key
    print("\nChecking configuration...")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("  ✗ GROQ_API_KEY not found in environment!")
        print("\n    Please create a .env file with:")
        print("    GROQ_API_KEY=your_api_key_here")
        print("\n    Get your API key from:  https://console.groq.com/")
        return False
    else:
        print(f"  ✓ GROQ_API_KEY found ({api_key[: 8]}... )")

    # Summary
    print()
    if missing:
        print("❌ Missing required packages:")
        print(f"   pip install {' '.join(missing)}")
        return False

    if warnings:
        print("⚠ Warnings (optional packages):")
        for w in warnings:
            print(f"   - {w}")

    print("\n✓ All required dependencies are installed!")
    return True


def check_audio_devices():
    """Check and display available audio devices."""
    try:
        import sounddevice as sd

        print("\nAudio Devices:")
        print("-" * 40)

        # Input devices
        default_input = sd. query_devices(kind='input')
        print(f"  Default Input:   {default_input['name']}")

        # Output devices
        default_output = sd.query_devices(kind='output')
        print(f"  Default Output: {default_output['name']}")

        print("-" * 40)
        return True

    except Exception as e:
        print(f"\n⚠ Audio device error: {e}")
        print("  Please check your microphone and speaker connections.")
        return False


def run_latency_test():
    """Run a quick latency test of all components."""
    print("\n" + "=" * 50)
    print("         RUNNING LATENCY TEST")
    print("=" * 50 + "\n")

    results = {}

    # Test STT initialization
    print("Testing STT...")
    start = time.perf_counter()
    try:
        from stt import SpeechToText
        stt = SpeechToText(vad_type="silero", enable_noise_reduction=True)
        results["stt_init"] = (time.perf_counter() - start) * 1000
        print(f"  ✓ STT initialized in {results['stt_init']:.1f} ms")
    except Exception as e:
        print(f"  ✗ STT failed: {e}")
        results["stt_init"] = -1

    # Test LLM
    print("\nTesting LLM...")
    try:
        from llm import BankingLLM
        llm = BankingLLM(model="llama-3.1-8b-instant")

        start = time. perf_counter()
        response, metrics = llm.get_response("What are your savings account rates?")
        results["llm_response"] = metrics.get("latency_ms", 0)
        print(f"  ✓ LLM response in {results['llm_response']:.1f} ms")
        print(f"    Response: {response[: 80]}...")
    except Exception as e:
        print(f"  ✗ LLM failed: {e}")
        results["llm_response"] = -1

    # Test TTS
    print("\nTesting TTS...")
    try:
        from tts import get_tts_engine
        tts = get_tts_engine()

        start = time. perf_counter()
        # Just synthesize, don't play
        if hasattr(tts, 'synthesize'):
            audio, synth_time = tts.synthesize("Hello, this is a test.")
            results["tts_synthesis"] = synth_time
        else:
            results["tts_synthesis"] = 0
        print(f"  ✓ TTS synthesis in {results. get('tts_synthesis', 0):.1f} ms")
    except Exception as e:
        print(f"  ✗ TTS failed: {e}")
        results["tts_synthesis"] = -1

    # Summary
    print("\n" + "=" * 50)
    print("         LATENCY TEST RESULTS")
    print("=" * 50)

    total = 0
    for component, latency in results.items():
        if latency >= 0:
            total += latency
            status = "✓"
        else:
            status = "✗"
        print(f"  {status} {component}: {latency:.1f} ms" if latency >= 0 else f"  {status} {component}:  FAILED")

    print("-" * 50)
    print(f"  Estimated round-trip:  {total:.1f} ms")
    print("=" * 50 + "\n")

    return results


def print_banner():
    """Print welcome banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║     ██╗  ██╗██╗   ██╗███████╗    ██████╗  █████╗ ███╗  ██╗║
    ║     ╚██╗██╔╝╚██╗ ██╔╝╚══███╔╝    ██╔══██╗██╔══██╗████╗ ██║║
    ║      ╚███╔╝  ╚████╔╝   ███╔╝     ██████╦╝███████║██╔██╗██║║
    ║      ██╔██╗   ╚██╔╝   ███╔╝      ██╔══██╗██╔══██║██║╚████║║
    ║     ██╔╝╚██╗   ██║   ███████╗    ██████╦╝██║  ██║██║ ╚███║║
    ║     ╚═╝  ╚═╝   ╚═╝   ╚══════╝    ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚══╝║
    ║                                                           ║
    ║           VOICE CUSTOMER SUPPORT AGENT v1.0               ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_help():
    """Print help information."""
    help_text = """
    Usage: python main.py [OPTIONS]
    
    Options: 
        --help, -h          Show this help message
        --test              Run latency test only
        --no-latency        Disable latency display during conversation
        --vad TYPE          VAD type:  silero (default), webrtc, or energy
        --model MODEL       LLM model:  llama-3.1-8b-instant (default),
                           llama-3.1-70b-versatile, mixtral-8x7b-32768
        --voice VOICE       TTS voice: af_heart (default), af_bella,
                           am_adam, bf_emma, bm_george
        --no-noise          Disable noise reduction
    
    Examples:
        python main.py                      # Start with defaults
        python main. py --test               # Run latency test
        python main. py --no-latency         # Hide latency metrics
        python main. py --model mixtral-8x7b-32768 --voice af_bella
    
    Environment Variables:
        GROQ_API_KEY        Your Groq API key (required)
    
    For more information, visit: https://github.com/your-repo/voice-agent
    """
    print(help_text)


def parse_args():
    """Parse command line arguments."""
    args = {
        "test": False,
        "show_latency": True,
        "vad_type": "silero",
        "llm_model": "llama-3.1-8b-instant",
        "tts_voice":  "af_heart",
        "noise_reduction": True,
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
        elif arg == "--no-noise":
            args["noise_reduction"] = False
        elif arg == "--vad" and i + 1 < len(argv):
            i += 1
            args["vad_type"] = argv[i]
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
        print("\n" + "=" * 60)
        print("Starting Voice Agent...")
        print("=" * 60 + "\n")

        agent = VoiceAgent(
            vad_type=args["vad_type"],
            enable_noise_reduction=args["noise_reduction"],
            llm_model=args["llm_model"],
            tts_voice=args["tts_voice"],
            show_latency=args["show_latency"]
        )

        agent.start()

    except KeyboardInterrupt:
        print("\n\nShutting down...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()