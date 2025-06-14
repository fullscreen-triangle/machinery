# Zengeza: Noise Reduction and Attention Space Optimization

**Making processes tenable by reducing the attention space**

Zengeza is a sophisticated noise reduction and attention space optimization system for the Machinery bioinformatics framework. It addresses a fundamental challenge: not all information is equally valuable, and focusing on the signal while filtering out noise is essential for efficient processing.

## Philosophy

*"Even valid information can be full of noise - information that does not help."*

Consider a 24-hour heart rate dataset. While all data points are valid, not every measurement is necessary for every process. Zengeza identifies what information truly matters and reduces the attention space to focus on the most informative segments.

## Key Features

- **Intelligent Noise Detection**: Statistical, temporal, and spatial noise analysis
- **Attention Space Optimization**: Importance-based data segment selection  
- **Smart Filtering**: Adaptive filtering based on noise profiles
- **Data Type Specialization**: Time series, images, text, signals
- **Machinery Integration**: Works with all Machinery components

## Quick Start

```python
from zengeza import ZengezaEngine, AttentionMode

# Initialize engine
engine = ZengezaEngine()
engine.start()

# Process data
result = engine.process_data(
    data=heart_rate_data,
    data_type="timeseries", 
    attention_mode=AttentionMode.BALANCED
)

print(f"Compression: {result.compression_ratio:.2f}x")
print(f"SNR: {result.noise_profile.snr_db:.1f} dB")
```

## Architecture

The system consists of noise detectors, filters, attention optimizers, data reducers, and integration components that work together to identify important information and reduce noise.

## Use Cases

- Biomedical signal processing (ECG, EEG compression)
- Large dataset analysis (genomic feature filtering)  
- Real-time monitoring (sensor stream optimization)
- Time series compression (preserving key events)

## Installation

```bash
pip install -e .
pip install -e .[gpu,research,timeseries]  # Optional components
```

For complete documentation and examples, see the full README. 