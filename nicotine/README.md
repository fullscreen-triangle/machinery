# Nicotine: Context Validation and Coherence Maintenance System

> "Just as humans take cigarette breaks to step back and refocus, AI systems need periodic 'context breaks' to validate their understanding and refresh their mental model."

## 🚬 Philosophy

The Nicotine framework addresses a critical challenge in long-running AI systems: **context drift** and **loss of coherence**. AI systems are prone to losing track of exactly what they are supposed to be doing, gradually drifting from their intended purpose as they process more information.

Nicotine solves this by implementing a "cigarette break" system that periodically pauses the AI system and presents it with machine-readable puzzles. If the system solves these puzzles correctly, it proves it still understands the context and can continue. If not, the system refreshes and recalibrates its understanding.

## 🎯 Key Features

### Context Validation
- **Puzzle-Based Validation**: Machine-readable puzzles that test understanding
- **Context Tracking**: Continuous monitoring of system context and coherence
- **Adaptive Difficulty**: Puzzles adjust difficulty based on system performance
- **Multi-Modal Testing**: Logic, memory, context, and integration puzzles

### Break Scheduling
- **Process-Count Triggers**: Breaks after processing N items
- **Time-Based Triggers**: Regular intervals regardless of activity
- **Coherence Triggers**: Breaks when coherence drops below threshold
- **Adaptive Scheduling**: Intervals adjust based on system performance

### System Integration
- **Mzekezeke Integration**: Health-triggered context breaks
- **Spectacular Integration**: Extraordinary event validation
- **Diggiden Integration**: Adversarial challenge modes
- **Hatata Integration**: Decision coherence tracking

### Coherence Monitoring
- **Real-time Tracking**: Continuous coherence level monitoring
- **Context Snapshots**: Historical context preservation
- **Drift Detection**: Early warning for context degradation
- **Automatic Recovery**: Context refresh and recalibration

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    NicotineEngine                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐│
│  │  Context        │  │    Puzzle       │  │   Break         ││
│  │  Tracker        │  │  Generator      │  │ Scheduler       ││
│  └─────────────────┘  └─────────────────┘  └─────────────────┘│
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐│
│  │  Puzzle         │  │  Coherence      │  │  Integration    ││
│  │  Solver         │  │  Validator      │  │  Manager        ││
│  └─────────────────┘  └─────────────────┘  └─────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### Core Components

1. **NicotineEngine**: Main orchestrator for context validation
2. **PuzzleGenerator**: Creates context-specific validation puzzles
3. **PuzzleSolver**: Attempts to solve generated puzzles
4. **ContextTracker**: Monitors and tracks context changes
5. **BreakScheduler**: Manages timing of context breaks
6. **CoherenceValidator**: Validates system coherence levels

## 🧩 Puzzle Types

### Context Puzzles
Test understanding of recent processing context:
- Recent process recall
- Process counting and statistics
- Coherence analysis
- Context summarization

### Logic Puzzles
Test reasoning and pattern recognition:
- Sequence completion
- Logical deduction
- Pattern matching
- Mathematical reasoning

### Memory Puzzles
Test retention of recent information:
- Process detail recall
- Sequence memory
- Temporal ordering
- Pattern memory

### Integration Puzzles
Test knowledge of system components:
- Component identification
- System purpose validation
- Integration awareness
- Cross-system understanding

## 🚀 Quick Start

### Installation

```bash
# Install from source
cd nicotine
pip install -e .

# Or install with optional dependencies
pip install -e ".[gpu,nlp]"
```

### Basic Usage

```python
from nicotine import NicotineEngine, NicotineConfig

# Initialize with default configuration
config = NicotineConfig()
engine = NicotineEngine(config)

# Start the engine
engine.start()

# Register process completion
engine.register_process(
    process_id="process_001",
    input_data="input data",
    output_data="output result",
    context={"task": "data_processing"}
)

# Check system status
status = engine.get_status()
print(f"Context state: {status['context_state']}")
print(f"Coherence level: {status['coherence_level']}")
print(f"Processes since last break: {status['process_count']}")

# Stop the engine
engine.stop()
```

### Advanced Configuration

```python
from nicotine import NicotineConfig

config = NicotineConfig(
    # Break triggers
    break_trigger_process_count=50,
    break_trigger_time_seconds=600,  # 10 minutes
    break_trigger_coherence_threshold=0.4,
    
    # Puzzle settings
    min_puzzle_confidence=0.8,
    puzzle_timeout_seconds=45,
    
    # Integration settings
    integration_settings={
        'spectacular': {
            'enabled': True,
            'extraordinary_event_break_trigger': True
        },
        'mzekezeke': {
            'enabled': True,
            'critical_health_break_trigger': True
        }
    }
)
```

## 🎮 Use Cases

### 1. Long-Running Data Processing
```python
# Process large datasets with context validation
for batch in large_dataset:
    result = process_batch(batch)
    
    # Register with nicotine
    engine.register_process(
        process_id=f"batch_{batch.id}",
        input_data=batch,
        output_data=result
    )
    
    # Engine automatically handles breaks and validation
```

### 2. AI Agent Monitoring
```python
# Monitor AI agent context during extended operation
class AIAgent:
    def __init__(self):
        self.nicotine = NicotineEngine()
        self.nicotine.start()
    
    def process_task(self, task):
        result = self.execute_task(task)
        
        # Validate context after each task
        self.nicotine.register_process(
            process_id=task.id,
            input_data=task,
            output_data=result,
            context={'agent_state': self.get_state()}
        )
        
        return result
```

### 3. Stream Processing with Context Breaks
```python
# Handle streaming data with periodic validation
async def stream_processor(data_stream):
    async for item in data_stream:
        processed = await process_item(item)
        
        engine.register_process(
            process_id=item.id,
            input_data=item,
            output_data=processed
        )
        
        # Check if break is needed
        status = engine.get_status()
        if status['context_state'] == 'break_time':
            await asyncio.sleep(1)  # Wait for break to complete
```

## ⚙️ Configuration

### Break Triggers
```python
# Process count trigger
config.break_trigger_process_count = 25

# Time-based trigger (5 minutes)
config.break_trigger_time_seconds = 300

# Coherence threshold trigger
config.break_trigger_coherence_threshold = 0.3
```

### Puzzle Configuration
```python
config.puzzle_generation_settings = {
    'context_puzzle': {
        'enabled': True,
        'difficulty_adaptive': True,
        'memory_depth': 5
    },
    'logic_puzzle': {
        'enabled': True,
        'complexity_levels': ['basic', 'intermediate', 'advanced']
    }
}
```

### Integration Settings
```python
config.integration_settings = {
    'spectacular': {
        'enabled': True,
        'extraordinary_event_break_trigger': True,
        'extraordinariness_threshold': 0.8
    },
    'mzekezeke': {
        'enabled': True,
        'health_context_weight': 0.8,
        'critical_health_break_trigger': True
    }
}
```

## 🔧 Development

### Project Structure
```
nicotine/
├── src/nicotine/
│   ├── __init__.py          # Public API
│   ├── core.py              # NicotineEngine
│   ├── config.py            # Configuration
│   ├── puzzles.py           # Puzzle generation
│   ├── solvers.py           # Puzzle solving
│   ├── context.py           # Context management
│   ├── schedulers.py        # Break scheduling
│   ├── validators.py        # Coherence validation
│   ├── integrations.py      # System integrations
│   └── utils.py             # Utilities
├── tests/                   # Test suite
├── examples/                # Usage examples
└── data/                    # Data files
```

### Running Tests
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=nicotine --cov-report=html

# Run specific test category
pytest tests/test_puzzles.py -v
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 🔗 Integration with Machinery Framework

Nicotine seamlessly integrates with other Machinery components:

- **Mzekezeke**: Health monitoring triggers context breaks
- **Spectacular**: Extraordinary events trigger validation
- **Diggiden**: Adversarial challenges enhance puzzle difficulty
- **Hatata**: Decision coherence influences break scheduling

## 📊 Monitoring and Metrics

```python
# Get detailed status
status = engine.get_status()
print(f"Total processes: {status['statistics']['total_processes']}")
print(f"Breaks taken: {status['statistics']['breaks_taken']}")
print(f"Puzzles solved: {status['statistics']['puzzles_solved']}")
print(f"Average coherence: {status['statistics']['avg_coherence_score']}")

# Get context history
history = engine.get_context_history()
for snapshot in history:
    print(f"Time: {snapshot['timestamp']}")
    print(f"Coherence: {snapshot['coherence_level']}")
    print(f"Processes: {snapshot['process_count']}")
```

## 🚨 Error Handling

```python
try:
    engine.register_process(
        process_id="risky_process",
        input_data=risky_input,
        output_data=risky_output
    )
except Exception as e:
    # Nicotine handles errors gracefully
    logger.error(f"Process registration failed: {e}")
    
    # Check if context refresh is needed
    if engine.get_status()['coherence_level'] == 'CRITICAL_FAILURE':
        engine.force_context_refresh()
```

## 🎯 Best Practices

1. **Regular Process Registration**: Register all significant processes
2. **Appropriate Break Intervals**: Balance frequency with performance
3. **Context-Rich Registration**: Provide meaningful context data
4. **Monitor Coherence Levels**: Watch for degradation patterns
5. **Integration Utilization**: Leverage other Machinery components
6. **Adaptive Configuration**: Adjust settings based on use case

## 📚 Examples

See the `examples/` directory for comprehensive usage examples:
- Basic usage patterns
- Integration with other systems
- Custom puzzle types
- Advanced configuration scenarios
- Performance optimization

## 🔮 Future Enhancements

- **Neural Puzzle Generation**: AI-generated context puzzles
- **Semantic Coherence Analysis**: NLP-based context validation
- **Distributed Context Tracking**: Multi-node coherence monitoring
- **Visual Context Dashboards**: Real-time coherence visualization
- **Predictive Break Scheduling**: ML-based break prediction

---

*Nicotine: Because even AI systems need to take a break and remember what they're doing.* 