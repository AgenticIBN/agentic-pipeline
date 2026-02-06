#!/usr/bin/env python3
"""
AgentOS Playground Starter for 6G Network Optimization
Start playground and test with multiple intents to verify state management

Run: python start_playground.py
Then: Open http://localhost:7777 and use Workflows tab
"""

import os
import json
from agno_workflow_pipeline import optimization_workflow, PipelineState, ACTIVE_INTENTS_FILE
from agno.os import AgentOS

def show_active_intents():
    """Display currently active intents."""
    if os.path.exists(ACTIVE_INTENTS_FILE):
        with open(ACTIVE_INTENTS_FILE, 'r') as f:
            intents = json.load(f)
        print(f"\n📊 Active Intents: {len(intents)}")
        for i, intent in enumerate(intents, 1):
            parsed = intent.get('parsed_intent', {})
            print(f"   {i}. Target: {parsed.get('target_area', 'N/A')} | Priority: {parsed.get('priority', 'N/A')}")
    else:
        print("\n📊 No active intents yet")

def main():
    print("\n" + "="*80)
    print("🚀 6G NETWORK OPTIMIZATION PIPELINE - AGENTΟΣ PLAYGROUND")
    print("="*80)
    
    # Show current state
    show_active_intents()
    
    print("\n" + "="*80)
    print("🌐 PLAYGROUND BAŞLATILIYOR...")
    print("="*80)
    print("\n📍 URL: http://localhost:7777")
    print("\n📖 KULLANIM TALİMATLARI:")
    print("   1. Browser'da http://localhost:7777 aç")
    print("   2. 'Workflows' sekmesine git")
    print("   3. '6G_Network_Optimization_Pipeline' workflow'unu seç")
    print("   4. 'Run' butonuna tıkla")
    print("   5. Input alanına şu formatı kullan:")
    print("\n   📝 ÖRNEK INPUT:")
    print('   {')
    print('     "natural_language_intent": "improve coverage in cell1",')
    print('     "resolution_strategy": "PRIORITY"')
    print('   }')
    print("\n   📝 DAHA FAZLA TEST İÇİN:")
    print('   Intent 1: "improve coverage in cell1"')
    print('   Intent 2: "increase throughput in cell2"')
    print('   Intent 3: "reduce power consumption in cell1"  <- Conflict!')
    print("\n   ⚠️  İkinci intent çalıştırdığında ilk intent\'i görecek!")
    print("   ⚠️  Conflict olursa resolution strategy devreye girecek!")
    print("\n" + "="*80)
    print("\n💡 İPUCU: Active intents'i görmek için:")
    print("   Terminal'de: cat active_intents_workflow.json")
    print("   veya: python -c 'from start_playground import show_active_intents; show_active_intents()'")
    print("\n🗑️  Active intents'i temizlemek için:")
    print("   rm active_intents_workflow.json")
    print("\n" + "="*80)
    
    # Create AgentOS instance
    agent_os = AgentOS(
        name="6G_Network_Optimizer",
        workflows=[optimization_workflow]
    )
    
    # Start playground
    print("\n⏳ Playground server başlatılıyor...\n")
    try:
        # Get the FastAPI app and serve it
        app = agent_os.get_app()
        agent_os.serve(app=app, host="localhost", port=7777)
    except KeyboardInterrupt:
        print("\n\n👋 Playground kapatıldı!")
        show_active_intents()

if __name__ == "__main__":
    main()
