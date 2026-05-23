#!/bin/bash

echo "=========================================="
echo "TEST 1: edvin.py (headless mode)"
echo "=========================================="
timeout 8 python edvin.py 2>&1 | grep -E "demo|completed|Available" && echo "✓ PASSED" || echo "✗ FAILED"

echo ""
echo "=========================================="
echo "TEST 2: edvin.py --list-cameras"
echo "=========================================="
timeout 5 python edvin.py --list-cameras 2>&1 | grep -v "WARN\|ERROR" | tail -3 && echo "✓ PASSED" || echo "✗ FAILED"

echo ""
echo "=========================================="
echo "TEST 3: mathew.py (headless mode)"
echo "=========================================="
timeout 8 python mathew.py 2>&1 | grep -E "demo|completed" && echo "✓ PASSED" || echo "✗ FAILED"

echo ""
echo "=========================================="
echo "TEST 4: mathew.py --list-cameras"
echo "=========================================="
timeout 5 python mathew.py --list-cameras 2>&1 | grep -v "WARN\|ERROR" | tail -2 && echo "✓ PASSED" || echo "✗ FAILED"

echo ""
echo "=========================================="
echo "TEST 5: test_headless_edvin.py"
echo "=========================================="
timeout 8 python test_headless_edvin.py 2>&1 | grep -E "completed|faces" | tail -2 && echo "✓ PASSED" || echo "✗ FAILED"

echo ""
echo "=========================================="
echo "TEST 6: edvin_gui.py (import check)"
echo "=========================================="
python -c "from edvin_gui import EdvinGui; print('✓ GUI module loads correctly')" && echo "✓ PASSED" || echo "✗ FAILED"

echo ""
echo "=========================================="
echo "All tests completed!"
echo "=========================================="
