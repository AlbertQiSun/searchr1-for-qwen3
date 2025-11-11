#!/bin/bash
# Search-R1 Training Script using verl

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Ensure we use the correct Python environment
export PATH="/gpfsnyu/scratch/qs2196/.conda/envs/train/bin:$PATH"

# Create necessary directories
mkdir -p data
mkdir -p checkpoints/search_r1
mkdir -p output/verl_training

echo "=== Search-R1 verl Training Setup ==="
echo ""

# Step 1: Preprocess data for verl training
echo "Step 1: Preprocessing data..."
python data_preprocess_search_r1.py \
    --dataset-path datasets/hotpotqa/dev.jsonl \
    --output-train data/search_r1_train.parquet \
    --output-val data/search_r1_val.parquet \
    --max-samples 10

if [ $? -eq 0 ]; then
    echo "✓ Data preprocessing completed successfully"
else
    echo "✗ Data preprocessing failed"
    exit 1
fi

echo ""

# Step 2: Verify components
echo "Step 2: Verifying components..."
echo "Checking verl installation..."
python -c "import verl; print('verl version:', verl.__version__)"

echo "Checking agent..."
python -c "from search_r1_agent import SearchAgentLoop; print('✓ Agent loaded successfully')"

echo "Checking reward function..."
python -c "from search_r1_reward import compute_score; print('✓ Reward function loaded successfully')"

echo "Checking data files..."
if [ -f "data/search_r1_train.parquet" ] && [ -f "data/search_r1_val.parquet" ]; then
    echo "✓ Data files exist"
else
    echo "✗ Data files missing"
    exit 1
fi

echo ""

# Step 3: Launch verl training
echo "Step 3: Launching verl training..."
echo "Note: This is a basic setup. You may need to adjust the verl command"
echo "based on your verl version and available trainers."
echo ""
echo "Testing agent initialization first..."
echo "Running: python -c \"from search_r1_agent import SearchAgentLoop; print('Agent import OK')\""
echo ""

# Test agent import first
python -c "from search_r1_agent import SearchAgentLoop; print('Agent import OK')" || exit 1

echo "Testing reward function..."
echo "Running: python -c \"from search_r1_reward import compute_score; print('Reward function OK')\""
echo ""

# Test reward function
python -c "from search_r1_reward import compute_score; print('Reward function OK')" || exit 1

echo "Testing data loading..."
echo "Running: python -c \"import pandas as pd; df = pd.read_parquet('data/search_r1_train.parquet'); print(f'Data OK: {len(df)} samples')\""
echo ""

# Test data loading
python -c "import pandas as pd; df = pd.read_parquet('data/search_r1_train.parquet'); print(f'Data OK: {len(df)} samples')" || exit 1

echo "All components verified. Now testing verl training..."
echo ""

# First try single process to debug with smaller model and sync rollout
export HYDRA_FULL_ERROR=1
python -m verl.trainer.main_ppo \
    +data.train_files='["data/search_r1_train.parquet"]' \
    +data.val_files='["data/search_r1_val.parquet"]' \
    ++actor_rollout_ref.model.path='/gpfsnyu/scratch/qs2196/.cache/models/Qwen3-0.6B' \
    ++critic.model.path='/gpfsnyu/scratch/qs2196/.cache/models/Qwen3-0.6B' \
    ++data.return_raw_chat=True \
    ++data.dataloader_num_workers=0 \
    ++data.train_batch_size=4 \
    ++data.max_prompt_length=128 \
    ++data.max_response_length=128 \
    ++actor_rollout_ref.rollout.mode=sync \
    ++actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    ++actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
    ++actor_rollout_ref.rollout.data_parallel_size=1 \
    ++actor_rollout_ref.rollout.agent.default_agent_loop='single_turn_agent' \
    ++actor_rollout_ref.rollout.agent.num_workers=1 \
    ++actor_rollout_ref.rollout.log_prob_micro_batch_size=1 \
    ++actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    ++actor_rollout_ref.rollout.prompt_length=128 \
    ++actor_rollout_ref.rollout.response_length=128 \
    ++actor_rollout_ref.rollout.max_num_batched_tokens=512 \
    ++actor_rollout_ref.rollout.max_num_seqs=4 \
    ++actor_rollout_ref.rollout.dtype=fp16 \
    ++trainer.device=cpu \
    ++trainer.n_gpus_per_node=0 \
    ++actor_rollout_ref.actor.ppo_micro_batch_size=1 \
    ++actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    ++critic.forward_micro_batch_size=1 \
    ++critic.ppo_mini_batch_size=2 \
    ++actor_rollout_ref.model.enable_gradient_checkpointing=True \
    ++actor_rollout_ref.model.use_torch_compile=False \
    ++trainer.logger='["console"]' \
    ++trainer.total_training_steps=5 \
    ++trainer.ppo_epochs=1

# If that works, uncomment the distributed version below
# echo "Starting distributed training..."
# torchrun --nproc_per_node=1 \
#     --master_port=12345 \
#     -m verl.trainer.main_ppo \
#     +data.train_files='["data/search_r1_train.parquet"]' \
#     +data.val_files='["data/search_r1_val.parquet"]' \
#     ++actor_rollout_ref.model.path='/gpfsnyu/scratch/qs2196/.cache/models/Qwen3-1.7B' \
#     ++data.return_raw_chat=True \
#     ++actor_rollout_ref.rollout.mode=async

echo ""
echo "=== Search-R1 Training Started! ==="

