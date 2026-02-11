
PROJECT_DIR="$(pwd)"
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
MODEL_NAME=Qwen3-mua-grpo

BATCH_SIZE=16
MINI_BATCH_SIZE=4
ROLLOUT_N=8
EPOCH_NUM=30
TEMPERATURE=1.2
MODEL_PATH="" 


# reward&log
ENABLE_THINKING="nothink" # think / nothink
TRAINSET_VERSION="apigen"
REWARD="naive"
REWARD_FUNC="sql_compute_score"
TRAIN_NAME="apigen8b"

# # user model
BASE_URL=""
API_KEY=""        
CHAT_MODEL="Qwen3-32B"
#######################################################################

if [ "$ENABLE_THINKING" == "think" ]; then
    ENABLE_THINKING_BOOL="True"
else
    ENABLE_THINKING_BOOL="False"
fi

SUFFIX="b${BATCH_SIZE}_mb${MINI_BATCH_SIZE}_n${ROLLOUT_N}_${ENABLE_THINKING}_${TRAINSET_VERSION}_R${REWARD_FUNC}_T${TEMPERATURE}_${TRAIN_NAME}"

CKPT_DIR=${CHECKPOINT_SAVE}/apigen/CKPT/${MODEL_NAME}-${SUFFIX}
export TENSORBOARD_DIR=${TENSORBOARD_PATH}/MUA-RL/tensorboard/${MODEL_NAME}-${SUFFIX}
ROLLOUT_LOG_PATH=${CHECKPOINT_SAVE}/apigen/log/${MODEL_NAME}-${SUFFIX}/rollout_log
VALID_LOG_PATH=${CHECKPOINT_SAVE}/apigen/log/${MODEL_NAME}-${SUFFIX}/valid_log

mkdir -p "$CKPT_DIR" "$TENSORBOARD_DIR" "$ROLLOUT_LOG_PATH" "$VALID_LOG_PATH"

export VERL_LOGGING_LEVEL=INFO
export HYDRA_FULL_ERROR=1
export CHAT_MODEL=$CHAT_MODEL
export API_KEY=$API_KEY
export BASE_URL=$BASE_URL
# : # Enable automatic padding for DataProto chunking to avoid len % chunks assertion
# export VERL_AUTO_PADDING=TRUE
# echo "VERL_AUTO_PADDING=$VERL_AUTO_PADDING"
#: > output.log

airline_empty_output=$PROJECT_DIR/data/airline_train.parquet
retail_empty_output=$PROJECT_DIR/data/retail_train.parquet

airline_path=$PROJECT_DIR/data/airline_test.parquet
retail_path=$PROJECT_DIR/data/retail_test.parquet
train_files="['$airline_empty_output','$retail_empty_output']"
test_files="['$airline_path','$retail_path']"

echo "RANK=$RANK"
echo "WORLD_SIZE=$WORLD_SIZE"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='mua_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=$BATCH_SIZE \
    data.max_prompt_length=5000 \
    data.max_response_length=13384 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.response_length_one_turn=8192 \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.model.enable_activation_offload=True\
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.validation_data_dir="$VALID_LOG_PATH"\
    trainer.logger=['console','tensorboard'] \
    trainer.rollout_data_dir="$ROLLOUT_LOG_PATH" \
    trainer.project_name=$MODEL_NAME \
    trainer.experiment_name=$MODEL_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$WORLD_SIZE \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/sqlbench_apigen_tool_config.yaml" \
    trainer.total_epochs=$EPOCH_NUM \
    trainer.default_local_dir=$CKPT_DIR \
    hydra.run.dir=$CKPT_DIR \
    actor_rollout_ref.rollout.enable_thinking=${ENABLE_THINKING_BOOL} \
    actor_rollout_ref.rollout.multi_turn.enable_tokenization_sanity_check=False \
    reward_model.reward_manager=$REWARD \
    custom_reward_function.path=verl/utils/reward_score/sqlbench.py \
    custom_reward_function.name=$REWARD_FUNC



 #actor_rollout_ref.rollout.engine_kwargs.disable_cascade_attn=True 