# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About vLLM

vLLM is a high-throughput and memory-efficient inference and serving engine for Large Language Models (LLMs). It provides state-of-the-art serving throughput with PagedAttention, supports 50+ popular models, and handles tensor/pipeline parallelism for distributed inference.

## Development Commands

### Setup
```bash
# Install development dependencies
pip install -r requirements/dev.txt

# Install pre-commit hooks (handles all linting/formatting)
pre-commit install --hook-type pre-commit --hook-type commit-msg
```

### Linting and Formatting
```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Manual mypy run (included in pre-commit)
./tools/mypy.sh

# Check specific Python version
pre-commit run mypy-3.9 --hook-stage manual --all-files
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test file with verbose output
pytest -s -v tests/test_logger.py

# Run tests for specific component
pytest tests/core/
pytest tests/model_executor/
```

### Documentation
```bash
# Install docs dependencies
pip install -r requirements/docs.txt

# Serve docs locally
mkdocs serve  # Available at http://127.0.0.1:8000/
```

### Building
```bash
# Build from source
pip install -e .

# For development without CUDA compilation
pip install -e . --no-build-isolation
```

## Core Architecture

vLLM recently underwent a major rearchitecture called "V1". This applies to the core parts of the system and the code for this has been arranged under `vllm/v1/`. The preexising "V0" code has been kept for now but is considered deprecated - this is much of the code outside of `vllm/v1`.

Some components were not changed and are common between V0 and V1. This includes the front-end parts, most notably the API Server (under `vllm/entrypoints`), much of the configuration (`vllm/config.py`) and CLI handling and various utilities. There are one or two other exceptions such as `WorkerBase` which is shared between V0 and V1.

You should focus primarily on the V1 code paths, especially for new functionality.

### Configuration System
- **VllmConfig**: Central unified configuration object passed throughout the system
- All models use: `__init__(*, vllm_config: VllmConfig, prefix: str = "")`
- Key configs: ModelConfig, ParallelConfig, SchedulerConfig, LoRAConfig

### Entrypoints
1. **LLM Class** (`vllm/entrypoints/llm.py`) - Offline inference interface
2. **API Server** (`vllm/entrypoints/openai/api_server.py`) - OpenAI-compatible server via `vllm serve`
3. **CLI** (`vllm/entrypoints/cli/main.py`) - Command-line interface

### Engine Architecture
- **LLMEngine** (`vllm/engine/llm_engine.py`) - Core synchronous engine
- **AsyncLLMEngine** (`vllm/engine/async_llm_engine.py`) - Asynchronous wrapper for serving
- **Flow**: Input processing → Scheduling → Model execution → Output processing

### Execution Model
- **Workers** (`vllm/worker/`) - Process-per-device pattern for distributed execution
- **Model Runners** (`vllm/worker/model_runner.py`) - Handle model loading and execution
- **Executors** (`vllm/executor/`) - Coordinate across workers (uniproc, multiproc, ray)

### Key Subsystems
- **Core Scheduling** (`vllm/core/`) - Block management, request scheduling, memory allocation
- **Attention** (`vllm/attention/`) - PagedAttention implementation and backends
- **Distributed** (`vllm/distributed/`) - Multi-GPU/multi-node coordination
- **Quantization** (`vllm/model_executor/layers/quantization/`) - GPTQ, AWQ, FP8, etc.
- **LoRA** (`vllm/lora/`) - Low-rank adaptation support
- **Multimodal** (`vllm/multimodal/`) - Vision/audio input processing

### V1 Architecture
- **Location**: `vllm/v1/` - New optimized architecture (1.7x speedup)
- **Features**: Zero-overhead prefix caching, clean execution loop, enhanced multimodal

## Development Guidelines

### Code Quality
- **DCO Required**: All commits must be signed off with `git commit -s`
- **SPDX Headers**: Required for all Python files (checked by pre-commit)
- **Style**: YAPF for Python formatting, clang-format for C++/CUDA
- **Type Checking**: MyPy with gradual type coverage expansion

### File Organization
- `/vllm/` - Main package
- `/csrc/` - CUDA/C++ kernels
- `/tests/` - Test suite organized by component
- `/docs/` - MkDocs documentation
- `/examples/` - Usage examples
- `/benchmarks/` - Performance benchmarking

### Testing Patterns
- Use pytest with markers: `core_model`, `distributed`, `skip_v1`, `optional`
- GPU tests require actual hardware (CI handles this)
- Component-specific test directories match source structure

### Import Conventions
- Use `import regex as re` (enforced by pre-commit)
- Avoid direct triton imports (use `vllm.triton_utils`)
- Import vLLM modules with full paths

### Model Implementation
- Follow `ModelRegistry` pattern for new models
- Use quantization-aware layer implementations
- Implement both single and multi-GPU execution paths

### PR Classification
Use prefixes: `[Bugfix]`, `[Model]`, `[Core]`, `[Kernel]`, `[Hardware]`, `[Frontend]`, `[Misc]`

## Common Tasks

### Adding New Models
1. Check `vllm/model_executor/models/` for similar architectures
2. Follow registration pattern in `ModelRegistry`
3. Add tests in `tests/models/`
4. Update supported models documentation

### Kernel Development
- C++/CUDA code in `/csrc/`
- Use CMake build system
- Follow existing patterns for Python bindings
- Extensive benchmarking in `/benchmarks/kernels/`

### Quantization Implementation
- Base classes in `vllm/model_executor/layers/quantization/`
- Follow existing patterns (AWQ, GPTQ examples)
- Implement both inference and weight loading

### Multi-modal Features
- Registry pattern in `vllm/multimodal/`
- Asset handling in `vllm/assets/`
- Input processing and validation

## Performance Considerations

- **Memory Management**: PagedAttention for KV cache efficiency
- **Batching**: Continuous batching of requests
- **Parallelism**: Tensor parallel for large models, pipeline parallel for memory constraints
- **Kernels**: Custom CUDA kernels for attention, quantization, and activation functions
- **Compilation**: CUDA graphs for inference speedup

## Notes for Claude Code

- The codebase has extensive type hints but mypy coverage is still expanding
- CUDA development requires appropriate GPU setup
- Many tests require GPU hardware - rely on CI for validation
- Pre-commit hooks handle most formatting/linting automatically
- Focus on component-level testing rather than end-to-end unless necessary
- V1 architecture is the future direction - prefer V1 patterns for new code