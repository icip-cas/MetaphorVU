eval_method=qa
api_type=wanqing
model_name="GPT-5"
max_images=50
aug_mode=mkg query_mode=keywords reference_mode=simple top_k=10 indexing_type=word num_hops=2

#######################################################################################################################################
echo "eval_method: ${eval_method}; model_name: ${model_name}; aug_mode: ${aug_mode}; query_mode: ${query_mode}; reference_mode: ${reference_mode}; top_k: ${top_k}; indexing_type: ${indexing_type}; num_hops: ${num_hops}"

log_file_path=logs/a_0.99_eval_metaphorvu_boost_${eval_method}_${model_name}_${aug_mode}_${query_mode}_${reference_mode}_${top_k}_${indexing_type}_${num_hops}.log
echo "log file path: ${log_file_path}"

nohup python -u a_0_eval_metaphorvu_boost.py \
    --api_type $api_type \
    --model_name $model_name \
    --max_images $max_images \
    --eval_method $eval_method \
    --aug_mode $aug_mode \
    --query_mode $query_mode \
    --reference_mode $reference_mode \
    --top_k $top_k \
    --indexing_type $indexing_type \
    --num_hops $num_hops \
    > ${log_file_path} 2>&1 &