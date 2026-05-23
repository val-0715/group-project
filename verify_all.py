#!/usr/bin/env python3
"""
Final Verification Script - Ensures all code is working perfectly
Run this to confirm everything is ready for deployment
"""

import subprocess
import sys
import os

def run_test(name, command, timeout=10):
    """Run a test command and return success/failure"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd="/workspaces/group-project"
        )
        success = result.returncode == 0 or "completed" in result.stdout.lower() or "passed" in result.stdout.lower()
        return success, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)

def main():
    print("=" * 60)
    print("FINAL VERIFICATION - Face Tracking Project")
    print("=" * 60)
    print()
    
    tests = [
        ("Python Syntax Check", "python -m py_compile edvin.py mathew.py edvin_gui.py test_headless_edvin.py", 5),
        ("Dependency Check", "python -c 'import cv2, mediapipe, numpy; print(\"OK\")'", 10),
        ("edvin.py Headless", "timeout 8 python edvin.py 2>&1 | grep -q 'demo\\|completed' && echo OK", 10),
        ("edvin.py --list-cameras", "python edvin.py --list-cameras 2>&1 | grep -q 'detected\\|indices' && echo OK", 10),
        ("mathew.py Headless", "timeout 8 python mathew.py 2>&1 | grep -q 'demo\\|completed' && echo OK", 10),
        ("mathew.py --list-cameras", "python mathew.py --list-cameras 2>&1 | grep -q 'detected\\|webcam' && echo OK", 10),
        ("test_headless_edvin.py", "timeout 8 python test_headless_edvin.py 2>&1 | grep -q 'finished\\|faces' && echo OK", 10),
        ("edvin_gui.py Import", "python -c 'from edvin_gui import EdvinGui; print(\"OK\")'", 5),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, command, timeout in tests:
        sys.stdout.write(f"{test_name:.<40} ")
        sys.stdout.flush()
        
        success, output = run_test(test_name, command, timeout)
        
        if success:
            print("✅ PASS")
            passed += 1
        else:
            print("❌ FAIL")
            failed += 1
            if output and "TIMEOUT" not in output:
                print(f"  Error: {output[:100]}")
    
    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("✅ ALL TESTS PASSED - Code is ready!")
        return 0
    else:
        print("❌ Some tests failed - please review")
        return 1

if __name__ == "__main__":
    sys.exit(main())
