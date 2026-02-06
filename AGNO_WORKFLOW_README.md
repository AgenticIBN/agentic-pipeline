# 🚀 AgentOS Workflow Pipeline - 6G Network Optimization

Complete end-to-end workflow system for 6G network optimization using **AgentOS Workflow Architecture**. 

## 🏗️ Architecture

This system uses **AgentOS Workflow** to orchestrate multiple specialized agents in a sequential pipeline:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AgentOS Workflow Pipeline                        │
└─────────────────────────────────────────────────────────────────────┘

   Intent Input (Natural Language)
          │
          ▼
┌─────────────────────────┐
│  STEP 1: Intent Parser  │  ← Groq LLM-based agent
│  - Parse NL to JSON     │     Converts natural language to structured intent
│  - Extract KPIs         │
│  - Set priorities       │
└───────────┬─────────────┘
            │ IntentParse
            ▼
┌─────────────────────────┐
│  STEP 2: Optimization   │  ← Surrogate model + local search
│  - Load surrogate       │     Generates optimal configuration
│  - Run optimization     │
│  - Predict KPIs         │
└───────────┬─────────────┘
            │ OptimizationResult
            ▼
┌─────────────────────────┐
│  STEP 3: Conflict       │  ← Deterministic detector
│    Detection            │     Compares with active intents
│  - Load active intents  │
│  - Detect conflicts     │
└───────────┬─────────────┘
            │ ConflictReport
            ▼
┌─────────────────────────┐
│  STEP 4: Resolution     │  ← Priority OR Weighted Merge
│  (if conflicts exist)   │     Resolves conflicts based on strategy
│  - Priority-based OR    │
│  - Weighted merge       │
└───────────┬─────────────┘
            │ ResolvedConfig
            ▼
┌─────────────────────────┐
│  STEP 5: Finalize       │  ← Storage & execution
│  - Save to active       │     Stores final configuration
│  - Return final config  │
└─────────────────────────┘
            │
            ▼
   Final Configuration + Execution Log
```

## 📦 Components

### 1. **Intent Parser Agent** (`intent_parser/intent_parser_agent.py`)
- **Type**: LLM-based agent (Groq)
- **Input**: Natural language text
- **Output**: Structured `IntentParse` JSON
- **Function**: Converts user intent to machine-readable format

### 2. **Optimization Agent V2** (`optimization_agent_v2.py`)
- **Type**: Surrogate model + local search
- **Input**: `IntentParse` JSON
- **Output**: Optimal configuration with predicted KPIs
- **Function**: Generates base-station parameter changes

### 3. **Conflict Detector Agent** (`conflict_detector_agent.py`)
- **Type**: Deterministic rule-based
- **Input**: New optimization result + active intents
- **Output**: `ConflictReport` with details
- **Function**: Detects parameter conflicts

### 4. **Resolution Agents**
   - **Priority-based** (`priority_based_resolution_agent.py`): Chooses highest priority
   - **Weighted Merge** (`weighted_merge_resolution_agent.py`): Merges configurations by weight

### 5. **Workflow Orchestrator** (`agno_workflow_pipeline.py`)
- **Type**: AgentOS Workflow
- **Function**: Coordinates all agents in sequence
- **Features**: 
  - State management across steps
  - Error handling
  - Execution logging
  - AgentOS Playground integration

## 🚀 Quick Start

### Option 1: AgentOS Playground (Recommended)

```bash
# Start playground server
python agno_workflow_pipeline.py --playground

# Open browser
# http://localhost:7777
# Navigate to: Workflows → 6G_Network_Optimization_Pipeline
```

In the playground, you can:
- ✅ Visualize the entire workflow
- ✅ See each step's input/output
- ✅ Track state transitions
- ✅ Debug individual steps
- ✅ Monitor execution flow

### Option 2: Command Line

```bash
# Basic usage
python agno_workflow_pipeline.py \
  --intent "Improve coverage in cell TX0 to at least -85 dBm" \
  --strategy PRIORITY

# With clear active intents
python agno_workflow_pipeline.py \
  --clear \
  --intent "Optimize SINR above 15 dB in site ABC" \
  --strategy WEIGHTED_MERGE

# Available strategies:
# - PRIORITY: Choose highest priority configuration
# - WEIGHTED_MERGE: Merge configurations by weight
```

### Option 3: Test Suite

```bash
# Run specific test
python test_workflow_pipeline.py --test 1

# Run all tests
python test_workflow_pipeline.py --all

# Custom test
python test_workflow_pipeline.py \
  --custom "Your intent here" \
  --strategy PRIORITY

# Clear and test
python test_workflow_pipeline.py --clear --test 2
```

## 📋 Example Usage

### Example 1: Simple Coverage Improvement

```bash
python agno_workflow_pipeline.py \
  --intent "Improve coverage in cell TX0 to at least -85 dBm" \
  --strategy PRIORITY
```

**Output:**
```
======================================================================
📋 STEP 1: INTENT PARSING
======================================================================
✅ Intent Parsed:
   Target Area: TX0
   Target KPIs: ['RX_POWER']
   Priority: MEDIUM
   Thresholds: RX_POWER GTE -85.0 dBm

======================================================================
🔧 STEP 2: CONFIGURATION OPTIMIZATION
======================================================================
✅ Optimization Result:
   Configuration Changes (12)
   Predicted KPIs: RX_POWER: -83.2 dBm

======================================================================
🔍 STEP 3: CONFLICT DETECTION
======================================================================
✅ NO CONFLICTS - Configuration can be applied directly

======================================================================
⚖️  STEP 4: CONFLICT RESOLUTION
======================================================================
⏭️  Skipped: No conflicts to resolve

======================================================================
✨ STEP 5: FINALIZE CONFIGURATION
======================================================================
✅ Configuration added to active intents

📊 WORKFLOW SUMMARY
Workflow ID: workflow_20260206_120000
Final Status: COMPLETED
```

### Example 2: With Conflicts (Priority Resolution)

```bash
# First intent
python agno_workflow_pipeline.py \
  --intent "Maximize throughput in cell TX0" \
  --strategy PRIORITY

# Second conflicting intent (HIGH priority)
python agno_workflow_pipeline.py \
  --intent "Minimize power consumption in TX0 - HIGH priority" \
  --strategy PRIORITY
```

**Output:**
```
======================================================================
🔍 STEP 3: CONFLICT DETECTION
======================================================================
⚠️  CONFLICTS DETECTED!
   Number of conflicts: 3
   Conflict Details:
      1. PARAMETER_CONFLICT - tx_power: opposite directions

======================================================================
⚖️  STEP 4: CONFLICT RESOLUTION
======================================================================
Resolution Strategy: PRIORITY

✅ Priority-Based Resolution:
   Winning Result: result_20260206_120100
   Winning Priority: HIGH
   Rejected Results: 1
   Notes: Selected highest priority configuration
```

### Example 3: Weighted Merge Strategy

```bash
python agno_workflow_pipeline.py \
  --intent "Balance coverage and capacity in downtown" \
  --strategy WEIGHTED_MERGE
```

## 🔍 AgentOS Playground Features

When you run `--playground`, you get:

1. **Workflow Visualization**: See all 5 steps in a flowchart
2. **State Inspector**: View `PipelineState` at each step
3. **Step-by-Step Execution**: Run workflow one step at a time
4. **Input/Output Viewer**: See what each agent receives and returns
5. **Execution Logs**: Track progress with detailed logs
6. **Error Debugging**: Catch and debug errors at specific steps

### Playground View Structure:

```
Workflows Tab
├── 6G_Network_Optimization_Pipeline
│   ├── parse_intent
│   │   ├── Input: natural_language_intent
│   │   └── Output: parsed_intent (IntentParse)
│   ├── optimize_configuration
│   │   ├── Input: parsed_intent
│   │   └── Output: optimization_result
│   ├── detect_conflicts
│   │   ├── Input: optimization_result
│   │   └── Output: conflict_report
│   ├── resolve_conflicts
│   │   ├── Input: conflict_report
│   │   └── Output: resolution_output
│   └── finalize_configuration
│       ├── Input: resolution_output
│       └── Output: final_configuration
```

## 📊 State Model

The `PipelineState` flows through all steps:

```python
class PipelineState(BaseModel):
    # Input
    natural_language_intent: str
    resolution_strategy: Literal["PRIORITY", "WEIGHTED_MERGE"]
    
    # Step outputs
    parsed_intent: Optional[IntentParse] = None
    optimization_result: Optional[Dict[str, Any]] = None
    conflict_report: Optional[Dict[str, Any]] = None
    resolution_output: Optional[Dict[str, Any]] = None
    final_configuration: Optional[Dict[str, Any]] = None
    
    # Tracking
    execution_log: List[str] = []
    workflow_id: str
    timestamp: str
    current_step: str
```

## 🎯 Key Differences from Old `workflow_steps.py`

| Feature | Old System | New AgentOS Workflow |
|---------|-----------|---------------------|
| Architecture | Monolithic function calls | AgentOS Workflow Steps |
| Agents | Tightly coupled | Loosely coupled, independent |
| State Management | Manual passing | Automated by AgentOS |
| Visualization | None | AgentOS Playground |
| Debugging | Print statements | Step-by-step inspection |
| Testing | Manual | Test suite included |
| Extensibility | Hard to modify | Easy to add/remove steps |
| Error Handling | Basic | Per-step with rollback |

## 🔧 Configuration

### Environment Variables

Create `.env` file:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### Required Files

- `models/surrogate.joblib` - Trained surrogate model
- `intent_parser/intent_parser_agent.py` - Intent parser
- `optimization_agent_v2.py` - Optimization agent
- `conflict_detector_agent.py` - Conflict detector
- `priority_based_resolution_agent.py` - Priority resolution
- `weighted_merge_resolution_agent.py` - Weighted merge resolution

### Storage

- `active_intents_workflow.json` - Stores active optimization results
- `workflow_result_YYYYMMDD_HHMMSS.json` - Workflow execution results

## 🧪 Testing

### Test Cases

1. **Test 1**: Simple coverage improvement (no conflicts)
2. **Test 2**: SINR optimization with high priority
3. **Test 3**: Throughput with weighted merge
4. **Test 4**: Load balancing between cells

### Run Tests

```bash
# Single test
python test_workflow_pipeline.py --test 1

# All tests with interactive mode
python test_workflow_pipeline.py --all

# Custom test
python test_workflow_pipeline.py \
  --custom "Improve signal quality in downtown, SINR > 20 dB" \
  --strategy WEIGHTED_MERGE
```

## 📈 Monitoring & Logs

Each workflow execution produces:

1. **Console Output**: Real-time step-by-step progress
2. **Execution Log**: Stored in `PipelineState.execution_log`
3. **Result Files**: JSON files with complete state
4. **Active Intents**: Updated in `active_intents_workflow.json`

### Example Log Output:

```
Execution Log:
  1. ✅ Intent parsed successfully
  2. ✅ Optimization completed
  3. ⚠️  Conflicts detected: 2
  4. ✅ Conflicts resolved using PRIORITY
  5. ✅ Workflow completed successfully
```

## 🔄 Workflow Execution Flow

```python
# 1. Create initial state
state = PipelineState(
    natural_language_intent="Your intent",
    resolution_strategy="PRIORITY"
)

# 2. Run workflow (AgentOS handles step-by-step execution)
result = optimization_workflow.run(input=state)

# 3. Access results
print(result.final_configuration)
print(result.execution_log)
```

## 🎨 AgentOS Playground Benefits

1. **Visual Debugging**: See exactly where errors occur
2. **State Inspection**: View state at any step
3. **Replay**: Re-run workflows with saved inputs
4. **Comparison**: Compare different resolution strategies
5. **Performance**: Monitor step execution times
6. **Documentation**: Auto-generated workflow docs

## 🚨 Error Handling

Each step has try-except blocks:

- **Parse Error**: Invalid intent format
- **Optimization Error**: Surrogate model issues
- **Conflict Detection Error**: Invalid result format
- **Resolution Error**: Strategy execution failure
- **Finalization Error**: Storage issues

Errors are logged to `execution_log` and workflow stops gracefully.

## 🔮 Future Enhancements

- [ ] Parallel conflict detection for multiple new intents
- [ ] Machine learning-based conflict prediction
- [ ] Real-time KPI monitoring integration
- [ ] Rollback capability for failed configurations
- [ ] A/B testing for resolution strategies
- [ ] Performance optimization caching
- [ ] Multi-site optimization support

## 📚 Related Files

- `WORKFLOW_SETUP_GUIDE.md` - Setup instructions
- `OPTIMIZATION_AGENT_README.md` - Optimization agent details
- `CONFLICT_RESOLUTION_ARCHITECTURE.md` - Conflict resolution architecture
- `README.md` - Main project README

## 🤝 Contributing

To add a new step to the workflow:

1. Create step function: `def step_N_your_step(step_input) -> PipelineState`
2. Add to workflow steps list
3. Update `PipelineState` model if needed
4. Add tests to `test_workflow_pipeline.py`

## 📞 Support

For issues or questions:
- Check AgentOS documentation: https://docs.agno.com
- Review workflow execution logs
- Use playground for debugging
- Check `execution_log` in results

---

**Built with AgentOS Workflow** 🚀
