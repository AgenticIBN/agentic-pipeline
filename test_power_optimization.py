#!/usr/bin/env python3
"""
test_power_optimization.py

Test optimization agent with a specific scenario:
- Current config: All transmitters at -80 dBm
- Intent: Increase RX_POWER to minimum -75 dBm
"""
import json
from optimization_agent_v2 import load_agent


def main():
    """Test power increase scenario."""
    
    print("="*80)
    print("TEST SCENARIO: Power Increase from -53 dBm to -51 dBm")
    print("="*80)
    print()
    
    # Load optimization agent
    print("Loading optimization agent...")
    agent = load_agent(search_iterations=2000)
    print(f"✓ Agent loaded with parameter ranges:")
    print(f"  Power: {agent.surrogate.param_ranges['power']}")
    print(f"  Azimuth: {agent.surrogate.param_ranges['dAz']}")
    print(f"  Elevation: {agent.surrogate.param_ranges['dEl']}")
    print()
    print(f"Dataset RX_POWER range: -140 to -45.42 dBm (best achievable: ~-45 dBm)")
    print()
    
    # Define current configuration - Suboptimal scenario
    # This should give approximately -56 dBm RX_POWER
    current_config = {
        "tx0_on": True,
        "tx0_P_dBm": 43.0,  # Minimum power
        "tx0_dAz": 0.0,
        "tx0_dEl": 0.0,
        
        "tx1_on": True,
        "tx1_P_dBm": 43.0,  # Minimum power
        "tx1_dAz": 0.0,
        "tx1_dEl": 0.0,
        
        "tx2_on": True,
        "tx2_P_dBm": 43.0,  # Minimum power
        "tx2_dAz": 0.0,
        "tx2_dEl": 0.0,
        
        "tx3_on": True,
        "tx3_P_dBm": 43.0,  # Minimum power
        "tx3_dAz": 0.0,
        "tx3_dEl": 0.0,
    }
    
    # Define intent: Increase RX_POWER to minimum -50 dBm
    intent = {
        "test_name": "power_increase_scenario",
        "target_area": "sector_test",
        "target_kpis": ["RX_POWER", "THROUGHPUT_5P"],
        "kpi_thresholds": [
            {
                "kpi": "RX_POWER",
                "op": "GTE",
                "value": -51.0,  # Target: minimum -50 dBm
                "unit": "dBm",
            },
        ],
        "priority": "HIGH",
        "confidence": 0.95,
        "current_config": current_config,
        "k_users": 800,
        "user_set_id": 0,
    }
    
    print("="*80)
    print("INPUT INTENT:")
    print("="*80)
    print(f"Target KPIs: {intent['target_kpis']}")
    print(f"Constraint: RX_POWER >= {intent['kpi_thresholds'][0]['value']:.1f} dBm")
    print(f"Priority: {intent['priority']}")
    print()
    print("Current Config:")
    for i in range(4):
        print(f"  TX{i}: ON={current_config[f'tx{i}_on']}, "
              f"Power={current_config[f'tx{i}_P_dBm']:.1f} dBm, "
              f"Az={current_config[f'tx{i}_dAz']:.1f}°, "
              f"El={current_config[f'tx{i}_dEl']:.1f}°")
    print()
    
    # Run optimization
    print("="*80)
    print("RUNNING OPTIMIZATION...")
    print("="*80)
    result = agent.optimize(intent)
    
    # Extract results
    output = result["output"]
    
    print()
    print("="*80)
    print("OPTIMIZATION RESULTS:")
    print("="*80)
    
    # Current KPIs
    if output["current_kpis"]:
        print()
        print("Current KPIs:")
        print(f"  RX_POWER:         {output['current_kpis']['RX_POWER']:.2f} dBm")
        print(f"  SINR:             {output['current_kpis']['SINR']:.2f} dB")
        print(f"  THROUGHPUT_5P:    {output['current_kpis']['THROUGHPUT_5P']:.2f} Mbps")
        print(f"  LOAD_IMBALANCE:   {output['current_kpis']['LOAD_IMBALANCE']:.2f}")
        print(f"  RX_COVERAGE:      {output['current_kpis']['RX_COVERAGE_RATIO']:.4f}")
    
    # Expected KPIs after optimization
    print()
    print("Expected KPIs after optimization:")
    print(f"  RX_POWER:         {output['expected_kpis']['RX_POWER']:.2f} dBm")
    print(f"  SINR:             {output['expected_kpis']['SINR']:.2f} dB")
    print(f"  THROUGHPUT_5P:    {output['expected_kpis']['THROUGHPUT_5P']:.2f} Mbps")
    print(f"  LOAD_IMBALANCE:   {output['expected_kpis']['LOAD_IMBALANCE']:.2f}")
    print(f"  RX_COVERAGE:      {output['expected_kpis']['RX_COVERAGE_RATIO']:.4f}")
    
    # Improvement
    if output["current_kpis"]:
        print()
        print("Improvement:")
        rx_diff = output['expected_kpis']['RX_POWER'] - output['current_kpis']['RX_POWER']
        thr_diff = output['expected_kpis']['THROUGHPUT_5P'] - output['current_kpis']['THROUGHPUT_5P']
        print(f"  RX_POWER:         {rx_diff:+.2f} dBm")
        print(f"  THROUGHPUT_5P:    {thr_diff:+.2f} Mbps")
    
    # Constraints
    print()
    print(f"Constraints satisfied: {'✓ YES' if output['constraints_satisfied'] else '✗ NO'}")
    
    # Configuration changes
    print()
    print(f"Configuration changes ({len(output['changes'])} total):")
    for change in output["changes"]:
        param = change["param"]
        before = change["before"]
        delta = change["change"]
        unit = change["unit"] or ""
        
        if unit:
            print(f"  {param:20s}: {before:8.2f} → {before + delta:8.2f} ({delta:+.2f} {unit})")
        else:
            print(f"  {param:20s}: {before} → {delta}")
    
    print()
    print("="*80)
    
    # Save full result
    output_file = "test_power_optimization_result.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Full result saved to: {output_file}")
    print("="*80)


if __name__ == "__main__":
    main()
