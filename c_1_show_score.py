import json
import pandas as pd
import numpy as np
import glob
import os


if __name__ == '__main__':
    with open('/mmu_vcg_wjc_hdd/lizhuoqun/metaphor_video_eval/benchmark/datas.jsonl', 'r', encoding='utf-8') as f:
        original_data = [json.loads(line.strip()) for line in f if line.strip()]

    model_names = [
        "qa_GPT-5",
    ]

    video_to_types = {}
    for item in original_data:
        video_to_types[item['video_id']] = item.get('metaphor_type', [])
    
    all_models_avg_scores = []
    all_models_sample_counts = []
    
    short_names = {
        "Atmosphere_Language": "AL",
        "Body_Language": "BL",
        "Naturalistic_Symbol": "NS",
        "Cultural_Symbol": "CS",
        "Analogical_Montage": "AM",
        "Causal_Montage": "CM",
        "Performative_Narrative": "PN",
        "Surreal_Narrative": "SN",
        "overall": "overall",
    }
    
    for model_name in model_names:
        print(f"正在处理模型: {model_name}")

        score_file = f'/mmu_vcg_wjc_hdd/lizhuoqun/metaphor_video_eval/score/{model_name}.jsonl'
        
        total_metrics = {
            "Body_Language": [],
            "Atmosphere_Language": [],
            "Cultural_Symbol": [],
            "Naturalistic_Symbol": [],
            "Causal_Montage": [],
            "Analogical_Montage": [],
            "Surreal_Narrative": [],
            "Performative_Narrative": [],
            "overall": [],
        }
        
        if not os.path.exists(score_file):
            print(f"文件不存在: {score_file}")
            avg_scores = {}
            sample_counts = {}
            
            for key in total_metrics.keys():
                avg_scores[key] = np.nan
                sample_counts[key] = 0
            
            avg_scores_short = {short_names[k]: v for k, v in avg_scores.items()}
            sample_counts_short = {short_names[k]: v for k, v in sample_counts.items()}
            
            avg_scores_short['Model'] = model_name
            sample_counts_short['Model'] = model_name
            
            all_models_avg_scores.append(avg_scores_short)
            all_models_sample_counts.append(sample_counts_short)
            continue
        
        with open(score_file, 'r', encoding='utf-8') as f:
            score_data = [json.loads(line.strip()) for line in f if line.strip()]

        for score_item in score_data:
            video_id = score_item['video_id']
            content_score = score_item['Content_Mining_Score']
            thought_score = score_item['Thought_Analysis_Score']
            
            total_metrics["overall"].append(thought_score)

            metaphor_types = video_to_types.get(video_id, [])
            
            for m_type in metaphor_types:
                thought_key = f"{m_type}-Thought_Analysis_Score"
                
                if thought_key in total_metrics:
                    total_metrics[thought_key].append(thought_score)

        avg_scores = {}
        sample_counts = {}
        
        for key, scores in total_metrics.items():
            if scores: 
                avg_scores[key] = round(np.mean(scores) * 10, 2) 
                sample_counts[key] = len(scores)
            else: 
                avg_scores[key] = np.nan
                sample_counts[key] = 0
        
        avg_scores_short = {short_names[k]: v for k, v in avg_scores.items()}
        sample_counts_short = {short_names[k]: v for k, v in sample_counts.items()}
        
        avg_scores_short['Model'] = model_name
        sample_counts_short['Model'] = model_name
        
        all_models_avg_scores.append(avg_scores_short)
        all_models_sample_counts.append(sample_counts_short)
    
    df_all_avg = pd.DataFrame(all_models_avg_scores) 
    df_all_count = pd.DataFrame(all_models_sample_counts)
    
    cols = ['Model'] + [col for col in df_all_avg.columns if col != 'Model']
    df_all_avg = df_all_avg[cols]
    df_all_count = df_all_count[cols]

    print(df_all_avg)
    print(df_all_count)

    df_all_avg.to_csv('c_1_show_score.csv', index=False, encoding='utf-8-sig')
    df_all_count.to_csv('c_1_show_score.count.csv', index=False, encoding='utf-8-sig')

    print("完成！")