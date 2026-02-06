#!/usr/bin/env python3
"""
run_full_pipeline.py

Complete pipeline to train surrogate models and test optimization agent.

Usage:
    # Train model (sample 1M rows for speed)
    python run_full_pipeline.py --train --sample-size 1000000
    
    # Test optimization only (uses existing model)
    python run_full_pipeline.py --test
    
    # Train and test
    python run_full_pipeline.py --train --test --sample-size 500000
"""
import argparse
import json
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()


def train_surrogate(sample_size: int | None = None):
    """Train surrogate models."""
    print("\n" + "="*80)
    print("TRAINING SURROGATE MODELS")
    print("="*80 + "\n")
    
    from train_surrogate_v2 import SurrogateTrainer
    
    data_path = os.getenv("DATA_PATH", "").strip()
    model_path = os.getenv("MODEL_PATH", "./models/surrogate.joblib").strip()
    
    if not data_path:
        raise ValueError("DATA_PATH must be set in .env file")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    print(f"Configuration:")
    print(f"  Data path: {data_path}")
    print(f"  Model path: {model_path}")
    print(f"  Sample size: {sample_size or 'ALL'}")
    print()
    
    start_time = time.time()
    
    # Train surrogate
    trainer = SurrogateTrainer(
        data_path=data_path,
        sample_size=sample_size,
        random_seed=42,
        chunk_size=100000,
    )
    
    df = trainer.load_data()
    trainer.train_models(df)
    trainer.save(model_path)
    
    elapsed = time.time() - start_time
    print(f"\n✓ Training complete in {elapsed:.1f} seconds!")
    print(f"✓ Model saved to: {model_path}")
    
    return model_path


def test_optimization():
    """Test optimization agent with multiple scenarios."""
    print("\n" + "="*80)
    print("TESTING OPTIMIZATION AGENT")
    print("="*80 + "\n")
    
    from optimization_agent_v2 import SurrogateModel, OptimizationAgent
    
    model_path = os.getenv("MODEL_PATH", "./models/surrogate.joblib").strip()
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Please train the model first using: python run_full_pipeline.py --train"
        )
    
    print(f"Loading surrogate model from: {model_path}")
    surrogate = SurrogateModel.load(model_path)
    print(f"✓ Loaded model with {len(surrogate.models)} KPI predictors")
    print(f"✓ Parameter ranges: {surrogate.param_ranges}\n")
    
    # Initialize agent
    agent = OptimizationAgent(
        surrogate=surrogate,
        search_iterations=1000,
        random_seed=42,
    )
    
    # Test scenarios
    test_cases = [
        {
            "name": "Scenario 1: Improve RX Power",
            "intent": {
                "test_name": "improve_rx_power",
                "target_area": "sector_A",
                "target_kpis": ["RX_POWER"],
                "kpi_thresholds": [
                    {"kpi": "RX_POWER", "op": "GTE", "value": -95.0, "unit": "dBm"},
                ],
                "priority": "HIGH",
                "confidence": 0.95,
                "current_config": {
                    "tx0_on": True, "tx0_P_dBm": 43.0, "tx0_dAz": 0.0, "tx0_dEl": 0.0,
                    "tx1_on": True, "tx1_P_dBm": 43.0, "tx1_dAz": 120.0, "tx1_dEl": 0.0,
                    "tx2_on": True, "tx2_P_dBm": 43.0, "tx2_dAz": -120.0, "tx2_dEl": 0.0,
                    "tx3_on": False, "tx3_P_dBm": 0.0, "tx3_dAz": 0.0, "tx3_dEl": 0.0,
                },
                "k_users": 800,
                "user_set_id": 0,
            }
        },
        {
            "name": "Scenario 2: Optimize Throughput",
            "intent": {
                "test_name": "optimize_throughput",
                "target_area": "sector_B",
                "target_kpis": ["THROUGHPUT_5P", "SINR"],
                "kpi_thresholds": [
                    {"kpi": "SINR", "op": "GTE", "value": 5.0, "unit": "dB"},
                    {"kpi": "THROUGHPUT_5P", "op": "GTE", "value": 20.0, "unit": "Mbps"},
                ],
                "priority": "MEDIUM",
                "confidence": 0.90,
                "current_config": {
                    "tx0_on": True, "tx0_P_dBm": 40.0, "tx0_dAz": 0.0, "tx0_dEl": 2.0,
                    "tx1_on": True, "tx1_P_dBm": 40.0, "tx1_dAz": 120.0, "tx1_dEl": 2.0,
                    "tx2_on": True, "tx2_P_dBm": 40.0, "tx2_dAz": -120.0, "tx2_dEl": 2.0,
                    "tx3_on": True, "tx3_P_dBm": 40.0, "tx3_dAz": 60.0, "tx3_dEl": 2.0,
                },
                "k_users": 800,
                "user_set_id": 0,
            }
        },
        {
            "name": "Scenario 3: Balance Load",
            "intent": {
                "test_name": "balance_load",
                "target_area": "sector_C",
                "target_kpis": ["SERVED_USERS", "RX_POWER"],
                "kpi_thresholds": [
                    {"kpi": "RX_POWER", "op": "GTE", "value": -100.0, "unit": "dBm"},
                ],
                "priority": "LOW",
                "confidence": 0.85,
                "current_config": {
                    "tx0_on": True, "tx0_P_dBm": 46.0, "tx0_dAz": 0.0, "tx0_dEl": 0.0,
                    "tx1_on": True, "tx1_P_dBm": 40.0, "tx1_dAz": 120.0, "tx1_dEl": 0.0,
                    "tx2_on": True, "tx2_P_dBm": 40.0, "tx2_dAz": -120.0, "tx2_dEl": 0.0,
                    "tx3_on": False, "tx3_P_dBm": 0.0, "tx3_dAz": 0.0, "tx3_dEl": 0.0,
                },
                "k_users": 800,
                "user_set_id": 0,
            }
        },
        {
            "name": "Scenario 4: From Scratch (No Current Config)",
            "intent": {
                "test_name": "from_scratch",
                "target_area": "sector_D",
                "target_kpis": ["RX_POWER", "SINR"],
                "kpi_thresholds": [
                    {"kpi": "RX_POWER", "op": "GTE", "value": -90.0, "unit": "dBm"},
                    {"kpi": "SINR", "op": "GTE", "value": 8.0, "unit": "dB"},
                ],
                "priority": "CRITICAL",
                "confidence": 0.95,
                "current_config": None,  # No baseline
                "k_users": 800,
                "user_set_id": 0,
            }
        },
    ]
    
    results = []
    
    for test_case in test_cases:
        print("\n" + "-"*80)
        print(f"Running: {test_case['name']}")
        print("-"*80)
        
        start_time = time.time()
        result = agent.optimize(test_case["intent"])
        elapsed = time.time() - start_time
        
        results.append(result)
        
        # Print summary
        output = result["output"]
        print(f"\n✓ Optimization complete in {elapsed:.2f}s")
        print(f"  Test: {result['test_name']}")
        print(f"  Passed: {result['passed']}")
        print(f"  Constraints satisfied: {output['constraints_satisfied']}")
        print(f"  Number of changes: {len(output['changes'])}")
        
        print(f"\n  Expected KPIs:")
        for kpi_name, kpi_value in output['expected_kpis'].items():
            if kpi_value is not None:
                print(f"    {kpi_name}: {kpi_value:.3f}")
        
        if output['current_kpis']:
            print(f"\n  Current KPIs:")
            for kpi_name, kpi_value in output['current_kpis'].items():
                if kpi_value is not None:
                    print(f"    {kpi_name}: {kpi_value:.3f}")
        
        if output['changes']:
            print(f"\n  Key changes:")
            for change in output['changes'][:5]:  # Show first 5
                if change['before'] is not None:
                    print(f"    {change['param']}: {change['before']} → "
                          f"{change['change']:+.2f} {change['unit'] or ''}")
                else:
                    print(f"    {change['param']}: {change['change']} {change['unit'] or ''}")
    
    # Save results
    output_dir = "./optimization_results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"results_{int(time.time())}.json")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print(f"✓ All tests complete! Results saved to: {output_file}")
    print("="*80)
    
    # Summary statistics
    passed_count = sum(1 for r in results if r['passed'])
    print(f"\nSummary: {passed_count}/{len(results)} tests passed")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train and test optimization pipeline")
    parser.add_argument("--train", action="store_true", help="Train surrogate models")
    parser.add_argument("--test", action="store_true", help="Test optimization agent")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of rows to sample for training (default: use all)",
    )
    
    args = parser.parse_args()
    
    if not args.train and not args.test:
        print("Error: Must specify --train and/or --test")
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.train:
            train_surrogate(sample_size=args.sample_size)
        
        if args.test:
            test_optimization()
        
        print("\n✓ Pipeline complete!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
