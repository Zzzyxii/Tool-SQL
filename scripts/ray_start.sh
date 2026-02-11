#export VLLM_ATTENTION_BACKEND=FLASHMLA 
# export VLLM_TEST_ENABLE_EP=1 
# export VLLM_FLASH_ATTN_VERSION=2
# export VLLM_USE_V1=0
# export VLLM_USE_FLASHINFER_SAMPLER=1
export RAY_memory_usage_threshold=0.98
# export MASTER_ADDR=$(hostname -I | awk '{print $1}')

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
WORK_DIR=`dirname $SCRIPT_DIR`

cd $WORK_DIR
echo "workspace_dir=$PWD"
echo "$SCRIPT_DIR"
export PYTHONPATH="$WORK_DIR:$PYTHONPATH"

curr_ip=$(python $WORK_DIR/scripts/get_host_ip.py)
if [ "$RANK" == "0" ]; then
    master_ip=$curr_ip
else 
    master_ip=$(python3 $WORK_DIR/scripts/get_domain_ip.py $MASTER_ADDR)
fi

echo $curr_ip > $CHECKPOINT_SAVE/ip_"$RANK".txt

echo $master_ip $curr_ip

if [ "$master_ip" = "$curr_ip" ]; then
    echo "run ray!!!!!!!!!!!!!!!!!"
    ray start --head --num-gpus 8 --max-worker-port 12800 --runtime-env-agent-port 20100 --dashboard-agent-grpc-port 20101 --dashboard-agent-listen-port 20102 --metrics-export-port 20103
    sleep 50s
    echo "start job!!!!!!!!!!!!!!!!!" 
else
    sleep 20s
    echo "connect to ray!!!!!!!!!!!!!!!!!"
    ray start --address $master_ip:6379 --block --max-worker-port 12800  --runtime-env-agent-port 20100 --dashboard-agent-grpc-port 20101 --dashboard-agent-listen-port 20102 --metrics-export-port 20103 
fi