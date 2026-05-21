import os
import json
import tqdm
import random
import argparse
from utlis.MetaphorVU_Boost import MetaphorVU_Boost
from utlis.use_wanqing_api import WangQingClient
from utlis.use_vllm_api import VLLMClient


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--api_type", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--max_images", type=int)
    parser.add_argument("--eval_method", type=str)
    parser.add_argument("--base_url", type=str, default="")
    parser.add_argument("--base_url_2", type=str, default="")
    parser.add_argument("--base_url_3", type=str, default="")
    parser.add_argument("--base_url_4", type=str, default="")
    parser.add_argument("--base_url_5", type=str, default="")
    parser.add_argument("--base_url_6", type=str, default="")
    parser.add_argument("--base_url_7", type=str, default="")
    parser.add_argument("--base_url_8", type=str, default="")
    parser.add_argument("--aug_mode", type=str)
    parser.add_argument("--query_mode", type=str)
    parser.add_argument("--reference_mode", type=str)
    parser.add_argument("--top_k", type=int)
    parser.add_argument("--indexing_type", type=str)
    parser.add_argument("--num_hops", type=int)
    args = parser.parse_args()

    assert args.api_type in ["vllm", "wanqing"], f"api_type {args.api_type} not supported"

    for k, v in vars(args).items():
        print(f"{k}: {v}")

    if args.api_type == "wanqing":
        client = WangQingClient(
            model_name=args.model_name,
            max_size=1024, max_images=args.max_images
        )
        client_list = [client]
    else:
        client_list = []
        for url in [args.base_url, args.base_url_2, args.base_url_3, args.base_url_4, args.base_url_5, args.base_url_6, args.base_url_7, args.base_url_8]:
            if url != "":
                client = VLLMClient(
                    base_url=url, 
                    model_name=args.model_name,
                    max_size=1024, max_images=args.max_images
                )
                client_list.append(client)
        print(f"using {len(client_list)} clients")

    metaphor_brain = MetaphorVU_Boost(vlm_client_list=client_list, aug_mode=args.aug_mode, query_mode=args.query_mode, reference_mode=args.reference_mode, top_k=args.top_k, indexing_type=args.indexing_type, num_hops=args.num_hops)

    datas = [json.loads(line) for line in open("/mmu_vcg_wjc_hdd/lizhuoqun/metaphor_video_eval/benchmark/datas.jsonl")]
    random.shuffle(datas)
    print(f"total {len(datas)} data")

    save_path = f"/mmu_vcg_wjc_hdd/lizhuoqun/metaphor_video_eval/output/metaphorvu_boost_{args.eval_method}_{args.model_name}_{args.aug_mode}_{args.query_mode}_{args.reference_mode}_{args.top_k}_{args.indexing_type}_{args.num_hops}.jsonl"
    fw = open(save_path, "a")
    alredy_processed_data_ids = [data['video_id'] for data in [json.loads(line) for line in open(save_path, "r").readlines()]]
    print(f"already processed {len(alredy_processed_data_ids)} / {len(datas)} data")

    for data in tqdm.tqdm(datas):

        if data['video_id'] in alredy_processed_data_ids:
            print(f"\nalready processed {data['video_id']}")
            continue
        
        print(f"\nwill process {data['video_id']}")

        output = "init_output"
        total_output = "init_total_output"
        output_clear = "init_output_clear"
        for try_time in range(1): # already has try in process
            # print(f"try time: {try_time}")

            try:
                image_dir_path=f"/mmu_vcg_wjc_hdd/lizhuoqun/metaphor_video_eval/benchmark/videos/{data['video_id']}"
                assert os.path.exists(image_dir_path), f"!!! {image_dir_path} not exists"

                if args.eval_method == "qa":
                    if data['title'] != "":
                        input_text = "The video title is: " + data['title']
                    else:
                        input_text = ""

                    output, total_output = metaphor_brain.process(title=input_text, image_dir_path=image_dir_path)
                    output_clear_json = {"video_id": data['video_id'], "analysis_dict": output["analysis_dict"], "total_output": total_output}
                else:
                    raise NotImplementedError
                
                fw.write(json.dumps(output_clear_json, ensure_ascii=False) + "\n")
                fw.flush()
                break
            except Exception as e:
                print(f"error: {e}")
                print(f"output: ", output)
                print(f"total_output: ", total_output)
                continue

    fw.close()