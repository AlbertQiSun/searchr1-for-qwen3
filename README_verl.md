# Search-R1 with verl Integration

This document explains how to train Search-R1 using verl's agentic RL framework.

## Overview

Search-R1 now integrates with [verl](https://github.com/volcengine/verl), a flexible RL training framework for LLMs. The integration uses verl's agentic RL capabilities for multi-turn search interactions.

## Architecture

- **Agent Loop**: `search_r1_agent.py` - Implements the SearchAgentLoop extending verl's RectAgentLoop
- **Search Tool**: Integrated search capabilities with BM25/E5 retrievers
- **Reward Function**: `search_r1_reward.py` - Computes rewards based on answer quality
- **Configuration**: `search_r1_config.yaml` - Training configuration for verl

## Files Added/Modified

### New Files:
- `search_r1_config.yaml` - verl training configuration
- `search_r1_agent.py` - Search agent implementation
- `search_r1_reward.py` - Reward function for training
- `data_preprocess_search_r1.py` - Data preprocessing for verl
- `train.sh` - Training launch script
- `README_verl.md` - This documentation

### Modified Files:
- `main.py` - Removed old verl.online_update code, added comments about separate training

## Setup

1. **Install dependencies**:
   ```bash
   conda activate train
   pip install verl mlflow sglang
   ```

2. **Prepare data**:
   ```bash
   python data_preprocess_search_r1.py \
     --dataset-path datasets/hotpotqa/dev.jsonl \
     --output-train data/search_r1_train.parquet \
     --output-val data/search_r1_val.parquet \
     --max-samples 1000
   ```

## Training

Run the training script:

```bash
./train.sh
```

This will:
1. Preprocess the data with agent_name field
2. Launch verl training with the Search-R1 configuration

## Key Components

### SearchAgentLoop
- Extends verl's `RectAgentLoop`
- Handles multi-turn search interactions
- Manages conversation state and tool calls
- Integrates with search retrievers

### Reward Function
- Computes F1 score against golden answers
- Bonuses for proper reasoning format
- Penalties for malformed responses
- Rewards effective search usage

### Configuration
- Uses sglang for inference engine
- Configured for agentic RL with async rollout
- Supports multi-turn conversations
- Includes tool calling capabilities

## Training Flow

1. **Data Preparation**: Convert dataset to verl format with agent_name field
2. **Model Initialization**: Load base model and tokenizer
3. **Rollout**: Generate responses using the SearchAgentLoop
4. **Reward Computation**: Score responses using SearchR1RewardModel
5. **Policy Update**: Update model using PPO/GRPO
6. **Iteration**: Repeat with improved policy

## Inference with Trained Model

After training, use the original `main.py` for inference:

```bash
python main.py --reasoner-model /path/to/trained/model [other args]
```

The trajectory collection remains for analysis, but actual training happens via verl.

## Configuration Options

### Agent Configuration
- `max_turns`: Maximum search turns (default: 5)
- `retriever_type`: "bm25" or "e5"
- `top_k_docs`: Number of documents to retrieve

### Training Configuration
- `model.path`: Base model path
- `actor_rollout_ref.rollout.mode`: "async" for agentic RL
- `data.return_raw_chat`: True for proper chat formatting

## Troubleshooting

### Common Issues:
1. **Import errors**: Ensure verl is installed in the train environment
2. **CUDA out of memory**: Reduce batch sizes in config
3. **No agent_name field**: Run data preprocessing first
4. **Search failures**: Check retriever index paths

### Debug Mode:
Enable mlflow tracing for debugging agent interactions:
```bash
pip install mlflow
mlflow ui -h 0.0.0.0 -p 5000
```

## Performance Notes

- Start with smaller models (Qwen-1.7B) for testing
- Use sglang backend for better performance
- Monitor GPU memory usage during training
- Adjust reward function weights based on your dataset

## Integration with Existing Code

The original Search-R1 inference code remains unchanged. Training now happens separately via verl, allowing you to:

- Train multiple models in parallel
- Use different RL algorithms
- Scale to larger datasets
- Integrate with existing model serving infrastructure


