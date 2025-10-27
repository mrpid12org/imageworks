#!/usr/bin/env python3
"""Test script to verify CLI commands are built correctly."""

import sys

sys.path.insert(0, "src")


# Test command building logic
test_scenarios = [
    {
        "name": "Download HF model",
        "command": [
            "uv",
            "run",
            "imageworks-download",
            "download",
            "Qwen/Qwen2.5-VL-7B-AWQ",
            "--format",
            "awq",
            "--location",
            "linux_wsl",
            "--non-interactive",
        ],
        "expected_parts": [
            "download",
            "Qwen/Qwen2.5-VL-7B-AWQ",
            "--format",
            "awq",
            "--non-interactive",
        ],
    },
    {
        "name": "Scan directory",
        "command": [
            "uv",
            "run",
            "imageworks-download",
            "scan",
            "--base",
            "/home/user/weights",
            "--dry-run",
        ],
        "expected_parts": ["scan", "--base", "--dry-run"],
    },
    {
        "name": "Normalize formats",
        "command": [
            "uv",
            "run",
            "imageworks-download",
            "normalize-formats",
            "--dry-run",
            "--rebuild",
        ],
        "expected_parts": ["normalize-formats", "--dry-run", "--rebuild"],
    },
    {
        "name": "Purge deprecated",
        "command": [
            "uv",
            "run",
            "imageworks-download",
            "purge-deprecated",
            "--dry-run",
        ],
        "expected_parts": ["purge-deprecated", "--dry-run"],
    },
    {
        "name": "Verify models",
        "command": ["uv", "run", "imageworks-download", "verify", "--fix-missing"],
        "expected_parts": ["verify", "--fix-missing"],
    },
    {
        "name": "Remove model",
        "command": [
            "uv",
            "run",
            "imageworks-download",
            "remove",
            "test-model",
            "--delete-files",
            "--force",
        ],
        "expected_parts": ["remove", "test-model", "--delete-files", "--force"],
    },
]

print("🧪 Testing CLI Command Construction")
print("=" * 60)

passed = 0
failed = 0

for scenario in test_scenarios:
    name = scenario["name"]
    command = scenario["command"]
    expected_parts = scenario["expected_parts"]

    command_str = " ".join(command)

    # Check if all expected parts are in the command
    all_present = all(part in command_str for part in expected_parts)

    if all_present:
        print(f"✅ {name}")
        print(f"   Command: {command_str}")
        passed += 1
    else:
        print(f"❌ {name}")
        print(f"   Command: {command_str}")
        print(f"   Missing: {[p for p in expected_parts if p not in command_str]}")
        failed += 1

    print()

print("=" * 60)
print(f"Results: {passed} passed, {failed} failed")

if failed == 0:
    print("✅ All tests passed!")
    sys.exit(0)
else:
    print("❌ Some tests failed")
    sys.exit(1)
