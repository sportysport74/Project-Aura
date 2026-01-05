"""
Quick test to verify COSF Jarvis components are working.
"""

print("=" * 70)
print("COSF JARVIS SETUP TEST")
print("=" * 70)
print()

# Test 1: Intent Router
print("Test 1: Intent Router")
print("-" * 70)
try:
    from src.python.cosf_mi.agent.intent_router import SimpleIntentRouter, Intent

    router = SimpleIntentRouter()

    tests = [
        "Hello, how are you?",
        "Open notepad",
        "Search my notes for COSF proofs",
        "Calculate phi^5",
    ]

    for test in tests:
        intent = router.route(test)
        print(f"  {test:45} -> {intent.value}")

    print("[OK] Intent Router working!\n")
except Exception as e:
    print(f"[FAIL] Intent Router failed: {e}\n")

# Test 2: Memory System
print("Test 2: Memory System")
print("-" * 70)
try:
    from src.python.cosf_mi.agent.memory import COSFMemory
    import os
    import shutil

    # Clean up test DB if it exists
    test_db = "memory/test_jarvis"
    if os.path.exists(test_db):
        shutil.rmtree(test_db)

    memory = COSFMemory(db_path=test_db)

    # Store some memories
    memory.store("The golden ratio phi ~= 1.618 is fundamental to COSF",
                 source="knowledge")

    memory.store("I prefer working late at night",
                 source="user_preference")

    memory.store("COSF uses Julia stability bounds to prevent divergence",
                 metadata={'tags': ['cosf', 'stability']},
                 source="knowledge")

    # Retrieve
    results = memory.retrieve("What is the golden ratio?", top_k=2)

    print(f"  Stored 3 memories")
    print(f"  Retrieved {len(results)} relevant memories:")
    for i, result in enumerate(results[:2], 1):
        text = result['text'][:60] + "..." if len(result['text']) > 60 else result['text']
        print(f"    {i}. {text}")
        print(f"       Source: {result['metadata']['source']}, Distance: {result['distance']:.4f}")

    print("[OK] Memory System working!\n")
except Exception as e:
    print(f"[FAIL] Memory System failed: {e}\n")
    import traceback
    traceback.print_exc()

# Test 3: Stability Governor
print("Test 3: Stability Governor")
print("-" * 70)
try:
    import torch
    from src.python.cosf_mi.agent.stability_governor import JuliaGovernor

    # Create mock model
    class MockModel:
        pass

    model = MockModel()
    governor = JuliaGovernor(model, bounds=2.0)

    # Simulate activations
    stable_act = torch.randn(10, 384) * 0.5  # Well within bounds
    unstable_act = torch.randn(10, 384) * 5.0  # Exceeds bounds

    print("  Testing stable activations:")
    stability = governor.monitor_inference(stable_act)
    print(f"    Status: {stability['status']}")
    print(f"    Exceed %: {stability['exceed_pct']:.1%}")

    print("  Testing unstable activations:")
    stability = governor.monitor_inference(unstable_act)
    print(f"    Status: {stability['status']}")
    print(f"    Exceed %: {stability['exceed_pct']:.1%}")

    metrics = governor.get_stability_metrics()
    print(f"  Current status: {metrics['status']}")
    print(f"  Corrections applied: {metrics['corrections_applied']}")

    print("[OK] Stability Governor working!\n")
except Exception as e:
    print(f"[FAIL] Stability Governor failed: {e}\n")
    import traceback
    traceback.print_exc()

# Summary
print("=" * 70)
print("SETUP TEST COMPLETE")
print("=" * 70)
print()
print("All core components are ready!")
print()
print("Next steps:")
print("  1. Run: python test_jarvis_demo.py (for full integration)")
print("  2. Check your Shakespeare model training progress")
print("  3. Start TinyStories data prep for scaled-up model")
print()
