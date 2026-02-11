#!/bin/bash

# unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
# export no_proxy=127.0.0.1,localhost

# 获取环境变量 WORLD_SIZE 和 RANK
WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}

# 定义检查点保存路径
CHECKPOINT_DIR="/checkpoint_save"

# 创建一个空文件，命名为 _SUCCESS_RANK

SUCCESS_FILE="${CHECKPOINT_DIR}/_SUCCESS_${RANK}"
echo $SUCCESS_FILE >> $CHECKPOINT_DIR/log.log
touch "${SUCCESS_FILE}"

# 函数：检查 /checkpoint_save/ 路径下 _SUCCESS_ 文件的数量
check_success_files() {
    success_files_count=$(ls ${CHECKPOINT_DIR}/_SUCCESS_* 2>/dev/null | wc -l)
    echo ${success_files_count}
}

# 循环判断直到 success 文件数量等于 WORLD_SIZE
while true; do
    success_files_count=$(check_success_files)
    if [ "${success_files_count}" -eq "${WORLD_SIZE}" ]; then
        echo "All _SUCCESS files are present. Continuing with the script."
        break
    else
        echo "Waiting for all _SUCCESS files. Current count: ${success_files_count} out of ${WORLD_SIZE}."
        sleep 1
    fi
done

# 继续执行脚本的其他部分
echo "All _SUCCESS files are present. Proceeding with the rest of the script."
