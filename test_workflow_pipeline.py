#!/usr/bin/env python3
"""
Test Script for AgentOS Workflow Pipeline
Tests the complete workflow with sample intents
"""

import json
import os
import sys
from datetime import datetime

# Test intents
TEST_INTENTS = [
    {
        "name": "Test 1: Simple Coverage Improvement",
        "intent": "Improve coverage in cell TX0 to at least -85 dBm",
        "strategy": "PRIORITY"
    },
    {
        "name": "Test 2: SINR Optimization",
        "intent": "Optimize signal quality for site ABC123, SINR should be above 15 dB with high priority",
        "strategy": "PRIORITY"
    },
    {
        "name": "Test 3: Throughput with Weighted Merge",
        "intent": "Increase throughput to 50 Mbps minimum in downtown area",
        "strategy": "WEIGHTED_MERGE"
    },
    {
        "name": "Test 4: Load Balancing",
        "intent": "Balance user load between 80-120 users per cell in cluster C1",
        "strategy": "PRIORITY"
    }
]


def clear_active_intents():
    """Clear active intents file."""
    active_file = "active_intents_workflow.json"
    if os.path.exists(active_file):
        os.remove(active_file)
        print("🗑️  Cleared active intents\n")


def run_workflow_test(intent_text: str, strategy: str, test_name: str):
    """Run workflow with a specific intent."""
    print("\n" + "="*80)
    print(f"🧪 {test_name}")
    print("="*80)
    print(f"Intent: {intent_text}")
    print(f"Strategy: {strategy}")
    print("="*80)
    
    # Import workflow
    from agno_workflow_pipeline import optimization_workflow, PipelineState
    
    # Create initial state
    initial_state = PipelineState(
        natural_language_intent=intent_text,
        resolution_strategy=strategy
    )
    
    try:
        # Run workflow
        result = optimization_workflow.run(input=initial_state)
        
        # Save result
        result_file = f"test_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        if hasattr(result, 'model_dump'):
            result_dict = result.model_dump()
        else:
            result_dict = result
            
        with open(result_file, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        
        print(f"\n✅ TEST PASSED")
        print(f"📁 Result saved to: {result_file}")
        
        # Print summary
        if hasattr(result, 'execution_log'):
            print(f"\n📋 Execution Log:")
            for log in result.execution_log:
                print(f"   {log}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test AgentOS Workflow Pipeline")
    parser.add_argument("--clear", action="store_true", help="Clear active intents before testing")
    parser.add_argument("--test", type=int, help="Run specific test number (1-4)")
    parser.add_argument("--all", action="store_true", help="Run all tests sequentially")
    parser.add_argument("--custom", type=str, help="Run with custom intent text")
    parser.add_argument("--strategy", type=str, choices=["PRIORITY", "WEIGHTED_MERGE"],
                       default="PRIORITY", help="Resolution strategy")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🧪 AgentOS Workflow Pipeline Test Suite")
    print("="*80)
    
    # Clear active intents if requested
    if args.clear:
        clear_active_intents()
    
    # Run custom test
    if args.custom:
        run_workflow_test(args.custom, args.strategy, "Custom Test")
    
    # Run specific test
    elif args.test:
        if 1 <= args.test <= len(TEST_INTENTS):
            test = TEST_INTENTS[args.test - 1]
            run_workflow_test(test["intent"], test["strategy"], test["name"])
        else:
            print(f"❌ Invalid test number. Choose 1-{len(TEST_INTENTS)}")
    
    # Run all tests
    elif args.all:
        results = []
        for i, test in enumerate(TEST_INTENTS, 1):
            success = run_workflow_test(test["intent"], test["strategy"], test["name"])
            results.append((test["name"], success))
            
            if i < len(TEST_INTENTS):
                print("\n" + "-"*80)
                input("Press Enter to continue to next test...")
        
        # Print summary
        print("\n" + "="*80)
        print("📊 TEST SUMMARY")
        print("="*80)
        for name, success in results:
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{status}: {name}")
        
        passed = sum(1 for _, s in results if s)
        print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    else:
        # Show menu
        print("\nAvailable Tests:")
        for i, test in enumerate(TEST_INTENTS, 1):
            print(f"  {i}. {test['name']}")
            print(f"     Intent: {test['intent']}")
            print(f"     Strategy: {test['strategy']}\n")
        
        print("\nUsage:")
        print("  Run specific test:  python test_workflow_pipeline.py --test 1")
        print("  Run all tests:      python test_workflow_pipeline.py --all")
        print("  Custom intent:      python test_workflow_pipeline.py --custom 'your intent' --strategy PRIORITY")
        print("  Clear intents:      python test_workflow_pipeline.py --clear --test 1")


if __name__ == "__main__":
    main()
