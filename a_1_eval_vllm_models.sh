cd /mmu_vcg_wjc_hdd/lizhuoqun/metaphor_video_eval

source /mmu_vcg_wjc_hdd/lizhuoqun/anaconda/bin/activate ads_understanding

eval_method=qa
api_type=wanqing
model_name="GPT-5"
max_images=50

#######################################################################################################################################
echo "api_type: ${api_type}; model_name: ${model_name}; eval_method: ${eval_method}"

log_file_path=logs/a_1_eval_vllm_models_${eval_method}_${model_name}.log
echo "log file path: ${log_file_path}"

nohup python -u a_1_eval_vllm_models.py \
    --api_type $api_type \
    --model_name $model_name \
    --max_images $max_images \
    --eval_method $eval_method \
    > ${log_file_path} 2>&1 &