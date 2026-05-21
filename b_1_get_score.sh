model_name_list=(
    "qa_GPT-5"
)

for model_name in "${model_name_list[@]}"; do
    if [[ $model_name == \#* ]]; then
        continue
    fi
    
    log_path="logs/b_1_get_score_${model_name}.log"
    echo "================================================================== ${log_path}"
    echo "Starting evaluation for: ${model_name}"
    nohup python -u b_1_get_score.py \
        --model_name "$model_name" \
        > ${log_path} 2>&1 &
    
    sleep 1
done

echo "All model evaluation jobs have been started in the background."
echo "Check logs in the logs/ directory for progress."