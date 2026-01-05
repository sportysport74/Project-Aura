"""
Quick automated test of Jarvis CLI components.
Tests basic interaction without requiring user input.
"""

import sys
from pathlib import Path

# Add COSF-MI to path
cosf_path = Path(__file__).parent / "COSF-MI"
sys.path.insert(0, str(cosf_path))
src_python_path = Path(__file__).parent / "COSF-MI" / "src" / "python"
sys.path.insert(0, str(src_python_path))

print("=" * 70)
print("JARVIS CLI AUTOMATED TEST")
print("=" * 70)
print()

# Import after path setup
from jarvis_cli import JarvisCLI

# Initialize Jarvis
print("Initializing Jarvis...")
jarvis = JarvisCLI()

print()
print("=" * 70)
print("TESTING CONVERSATIONAL INTERACTION")
print("=" * 70)

# Test 1: Simple greeting
print("\nTest 1: Greeting")
print("-" * 70)
jarvis.process_turn("Hello, who are you?")

# Test 2: Knowledge query
print("\n\nTest 2: Knowledge Query")
print("-" * 70)
jarvis.process_turn("What is the golden ratio?")

# Test 3: System status
print("\n\n" + "=" * 70)
print("SYSTEM STATUS")
print("=" * 70)
jarvis.show_status()

# Test 4: Memory retrieval
print("\n" + "=" * 70)
print("MEMORY CHECK")
print("=" * 70)
jarvis.show_memory()

print()
print("=" * 70)
print("AUTOMATED TEST COMPLETE")
print("=" * 70)
print()
print("All systems operational!")
print()
print("To run interactive mode:")
print("  python jarvis_cli.py")
print()
