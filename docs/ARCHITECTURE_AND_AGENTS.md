# Architecture and Agent Responsibilities

## 1. System purpose

The Agentic IBN system converts a natural-language network objective into a validated four-transmitter configuration. It predicts the resulting KPIs, detects interactions with active intents, applies an appropriate conflict-resolution strategy, and stores a complete execution record.

The system is designed as a hybrid agentic architecture:

- **Agno Agents** perform semantic interpretation, configuration proposal, policy selection, auditing, and explanation.
- **Deterministic services** enforce exact constraints, validation rules, numerical calculations, and persistence behavior.
- **Scenario-specific surrogate models** estimate wireless KPIs for candidate and final configurations.
- **Agno Workflow** provides the top-level execution graph through named, ordered steps.

This separation allows language models to handle reasoning-intensive tasks without assigning exact arithmetic or safety-critical decisions to probabilistic generation.

## 2. End-to-end execution graph

```text
Agno Workflow
│
├─ Step 1: Intent Parser Agent
│    └─ natural language → structured ParsedIntent
│
├─ Step 2: Optimization Agent
│    └─ propose → validate → predict → score → refine
│
├─ Step 3: Conflict Detector Agent
│    └─ compare the candidate with active intents
│
├─ Step 4: Meta Agent Resolution
│    ├─ Priority Resolution Agent (CRS-1)
│    └─ Weighted Merge Agent (CRS-2)
│
├─ Step 5: Final Surrogate Validation
│    └─ evaluate the resolved final configuration
│
└─ Step 6: Reasoning Agent and Persistence
     └─ explain, update active state, and store run artifacts
```

The main implementation is located in:

```text
agentic_ibn/orchestration/workflow.py
```

`AgenticWorkflow` creates an `agno.workflow.Workflow` containing six named `Step` objects. The workflow fixes the execution order so that parsing, validation, conflict handling, final prediction, and persistence cannot be skipped or arbitrarily reordered by an LLM.

## 3. Architectural layers

### 3.1 Agent layer

```text
agentic_ibn/agents/
├── runtime.py
├── intent_parser_agent.py
├── optimization_agent.py
├── conflict_detector_agent.py
├── meta_agent.py
├── priority_resolution_agent.py
├── weighted_merge_agent.py
└── reasoning_agent.py
```

Each named specialist exposes a factory that returns an actual `agno.agent.Agent`:

```text
create_intent_parser_agent
create_optimization_agent
create_conflict_detector_agent
create_meta_agent
create_priority_resolution_agent
create_weighted_merge_agent
create_reasoning_agent
```

### 3.2 Deterministic decision-support layer

```text
agentic_ibn/constraints.py
agentic_ibn/optimization/objective.py
agentic_ibn/optimization/validator.py
agentic_ibn/surrogate/model.py
agentic_ibn/storage/state_store.py
agentic_ibn/storage/result_store.py
```

These modules provide exact operations that should not depend on free-form LLM output.

### 3.3 Shared schema layer

```text
agentic_ibn/schemas.py
```

Pydantic schemas define the contracts between agents, workflow steps, deterministic services, storage, and command-line entry points. Important schemas include:

```text
IntentParse
ParsedIntent
NetworkConfig
CandidateProposal
KPIPrediction
OptimizationResult
ConflictReport
ResolutionDecision
ResolutionResult
StrategicNarrative
WorkflowResult
```

## 4. Agno agent construction pattern

The agents use a consistent construction pattern:

```python
from agno.agent import Agent
from agno.models.groq import Groq


def create_example_agent(model_id: str) -> Agent:
    return Agent(
        name="Example Agent",
        model=Groq(id=model_id),
        instructions=[
            "Follow the assigned specialist role.",
            "Return only information allowed by the output schema.",
        ],
        output_schema=ExampleSchema,
        structured_outputs=True,
    )
```

The essential elements are:

- a specialist name and responsibility;
- explicit instructions defining allowed behavior;
- a Groq-hosted model through `agno.models.groq.Groq`;
- a Pydantic `output_schema`;
- structured output enforcement.


## 5. Intent Parser Agent

**File:** `agentic_ibn/agents/intent_parser_agent.py`  
**Factory:** `create_intent_parser_agent()`  
**Output schema:** `IntentParse`

The Intent Parser Agent converts the operator's natural-language request into a structured intent representation.

Responsibilities:

- identify the target area;
- map language to supported KPI names;
- extract comparison operators and numeric thresholds;
- identify open-ended improvement goals;
- parse time constraints when sufficiently specified;
- infer intent priority and parsing confidence;
- avoid proposing transmitter actions or configurations.

Example input:

```text
Improve coverage in Kadikoy and keep RX power above -95 dBm.
```

Conceptual structured result:

```json
{
  "area": "Kadikoy",
  "objectives": [
    {"kpi": "coverage", "operator": "TARGET"},
    {"kpi": "rx_power_dbm", "operator": ">=", "value": -95.0}
  ],
  "priority": "MEDIUM",
  "confidence": 0.94
}
```

`agentic_ibn/intent_adapter.py` attaches the original text and scenario and converts the parser output into the internal `ParsedIntent` schema.

### Hard transmitter commands

Explicit transmitter commands are scanned separately by `agentic_ibn/constraints.py`.

Examples:

```text
Turn off TX2.
TX2 is under maintenance.
Use all available transmitters.
```

These commands become hard TX-state constraints. Candidate generation and conflict resolution must preserve them.

## 6. Optimization Agent

**File:** `agentic_ibn/agents/optimization_agent.py`  
**Factory:** `create_optimization_agent()`  
**Output schema:** `CandidateProposal`

The Optimization Agent proposes a complete absolute `NetworkConfig` rather than an informal list of changes. Its input contains:

- the parsed intent;
- the baseline or current configuration;
- the current surrogate prediction;
- objective-gap information;
- hard TX-state constraints;
- parameter ranges supported by the surrogate artifact;
- feedback from earlier candidate evaluations.

The agent proposes network actions but does not invent KPI evidence. KPI values are produced only by the surrogate model.

### 6.1 Candidate configuration

A candidate contains the complete state of each transmitter, including fields such as:

```text
tx_on
tx_power_dbm
azimuth_deg
elevation_deg
```

Using absolute configurations avoids ambiguity about whether a value is a delta or a final setting.

### 6.2 Optimization loop

`OptimizationLoop` is the deterministic executor surrounding the Agno Optimization Agent:

```text
1. Ask the Optimization Agent for a candidate.
2. Apply hard transmitter-state constraints.
3. validate parameter bounds and reject invalid configurations.
4. Run surrogate prediction.
5. Score the prediction against the parsed intent.
6. Return objective gaps and validation feedback to the agent.
7. Retain the best evaluated candidate.
8. Stop when targets are satisfied or the iteration limit is reached.
```

The default maximum number of candidate evaluations is controlled by:

```text
MAX_OPTIMIZATION_ITERATIONS=3
```

### 6.3 Validation rules

The candidate validator enforces:

- model-supported parameter ranges;
- valid transmitter states;
- hard ON/OFF commands;
- valid numeric values;
- rejection of an all-OFF configuration;
- a complete four-transmitter configuration.

## 7. Surrogate prediction and objective scoring

The surrogate layer is located in:

```text
agentic_ibn/surrogate/
├── features.py
├── model.py
└── training.py
```

For every validated candidate, the surrogate predicts the configured KPI set. The objective scorer in `agentic_ibn/optimization/objective.py` then evaluates whether the prediction satisfies the parsed intent.

The surrogate is authoritative for model-predicted KPI values. Agents may interpret these values but may not replace them with generated estimates.

The runtime and training layers communicate through a stable artifact contract:

- scenario identifier;
- ordered feature names;
- model-supported feature ranges;
- target KPI names;
- one trained regressor per target;
- preprocessing and artifact metadata.

Retraining is required when the dataset, feature definition, target set, scenario contract, model family, or training configuration changes. Agent instructions and orchestration logic do not by themselves alter the surrogate artifact contract.

## 8. Conflict Detector Agent

**File:** `agentic_ibn/agents/conflict_detector_agent.py`  
**Factory:** `create_conflict_detector_agent()`  
**Output schema:** `ConflictReport`

The Conflict Detector Agent examines the candidate result together with active or suppressed intent records. Conflict checks are relevant when spatial and temporal scopes overlap.

The deterministic `ConflictDetector` computes the authoritative predicate scan for:

- `BOOLEAN_CONFLICT`;
- `PARAMETER_CONFLICT`;
- `RESOURCE_CONTENTION`;
- `BASE_STATION_INTERACTION`;
- `KPI_DOMAIN_CONFLICT`.

The resulting report contains:

- whether a conflict exists;
- conflict type and severity;
- affected parameters;
- participating intents;
- whether each conflict is mergeable;
- the recommended resolution strategy;
- an explanation of the detected interaction.

The Agno agent audits the structured report and can improve its explanation. A structural signature check protects conflict existence, severity, mergeability, participants, affected parameters, and the recommended strategy. When the agent output disagrees with the verified predicate result, the deterministic report remains authoritative.

## 9. Dynamic Meta Agent

**File:** `agentic_ibn/agents/meta_agent.py`  
**Factory:** `create_meta_agent()`  
**Output schema:** `ResolutionDecision`

The Meta Agent is invoked when a conflict exists. It selects the appropriate conflict-resolution specialist based on the verified `ConflictReport`.

Supported policies:

```text
PRIORITY
WEIGHTED_MERGE
```

Policy rules:

- `PRIORITY` is used for non-mergeable conflicts, boolean contradictions, high-severity conflicts, or critical conflicts.
- `WEIGHTED_MERGE` is eligible only when every conflict detail is mergeable.
- a user-requested strategy is still checked against safety and mergeability rules;
- weighted merge is rejected when any verified conflict detail is non-mergeable.

The Meta Agent chooses a policy; the selected resolver performs the exact computation.

## 10. Priority Resolution Agent — CRS-1

**File:** `agentic_ibn/agents/priority_resolution_agent.py`  
**Factory:** `create_priority_resolution_agent()`  
**Output schema:** `ResolutionResult`

The priority policy resolves non-mergeable intent interactions by selecting one authoritative configuration.

Priority order:

```text
CRITICAL > HIGH > MEDIUM > LOW
```

Tie-break order:

1. higher priority;
2. earlier registration time;
3. lexicographically smaller result ID.

`PriorityResolver` computes:

- the winning result;
- suppressed result IDs;
- participating intents;
- the final configuration;
- the formal resolution rationale.

The Priority Resolution Agent receives the computed result and generates a clear policy explanation. A structural signature check prevents changes to the winner, suppressed records, participants, strategy, or final configuration.

## 11. Weighted Merge Agent — CRS-2

**File:** `agentic_ibn/agents/weighted_merge_agent.py`  
**Factory:** `create_weighted_merge_agent()`  
**Output schema:** `ResolutionResult`

The weighted-merge policy combines compatible configurations according to intent priority.

Priority weights:

```text
LOW=1
MEDIUM=2
HIGH=3
CRITICAL=4
```

Resolution behavior:

- numeric absolute values are combined by weighted average;
- Boolean TX states use weighted voting;
- explicit hard TX-state constraints override voting;
- the merged configuration is validated before use;
- the merged configuration is passed through the surrogate again.

The final surrogate evaluation is necessary because a numerically valid merge may still produce unacceptable wireless KPIs.

The Weighted Merge Agent explains the computed result. Structural signature verification prevents it from modifying the strategy, participants, merged configuration, or other authoritative fields.

## 12. Final surrogate validation

After conflict resolution, the workflow performs a separate surrogate prediction on the final resolved configuration.

This stage ensures that the stored result corresponds to the configuration that will actually become active, rather than only to the Optimization Agent's original candidate.

The final prediction is included in `WorkflowResult` and is available to the Reasoning Agent and the external Colab validation process.

## 13. Reasoning Agent

**File:** `agentic_ibn/agents/reasoning_agent.py`  
**Factory:** `create_reasoning_agent()`  
**Output schema:** `StrategicNarrative`

The Reasoning Agent produces the operator-facing explanation after all numerical and conflict-resolution decisions are complete.

It explains:

- how the intent was interpreted;
- which configuration was proposed;
- the optimization attempts and target status;
- surrogate-predicted KPI values;
- detected conflicts;
- the selected resolution strategy;
- the final configuration;
- hard constraints and safety notes;
- the difference between surrogate prediction and simulator validation.

The Reasoning Agent does not alter the final configuration or conflict result. A deterministic narrative is available as a fallback when the LLM call fails.

## 14. Deterministic safety boundary

The following operations are intentionally deterministic:

- explicit TX ON/OFF command extraction;
- model-domain validation and clamping;
- rejection of invalid or all-OFF configurations;
- feature construction;
- surrogate inference;
- objective scoring;
- conflict predicate calculation;
- CRS-1 winner calculation;
- CRS-2 weighted arithmetic;
- structural signature verification;
- active-state updates;
- run-result persistence.

Agno Agents operate around these functions by interpreting, proposing, selecting, auditing, and explaining. They do not replace exact calculations with generated text.

## 15. Active state and intent lifecycle

Active state is stored in:

```text
state/active_intents.json
```

The state includes:

- active and suppressed intent records;
- their priorities and registration timestamps;
- their proposed or resolved configurations;
- the currently effective network configuration.

Each workflow run loads the state before optimization. Conflict detection compares the new result with relevant active or suppressed records. After resolution, the state store applies the result and writes the updated state atomically.

## 16. Run history and persistence

Each execution creates a timestamped directory:

```text
results/YYYY-MM-DD_HH-MM-SS_runN/
├── input.json
├── agent_history.json
├── active_intents_before.json
├── active_intents_after.json
└── final_result.json
```

The files provide:

- the original command and runtime settings;
- ordered agent and workflow-step history;
- input and output payloads for each recorded stage;
- state before and after resolution;
- the final structured workflow result.

When an exception occurs, the run directory contains:

```text
error.json
```

Completed history entries remain available for debugging.