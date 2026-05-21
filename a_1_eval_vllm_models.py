import os
import json
import tqdm
import random
import argparse
from utlis.use_vllm_api import VLLMClient
from utlis.use_wanqing_api import WangQingClient
from utlis.eval_prompts import prompt


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--api_type", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--max_images", type=int)
    parser.add_argument("--eval_method", type=str)
    parser.add_argument("--base_url", type=str, default="")
    args = parser.parse_args()

    assert args.api_type in ["vllm", "wanqing"], f"api_type {args.api_type} not supported"

    if args.api_type == "wanqing":
        client = WangQingClient(
            model_name=args.model_name,
            max_size=1024, max_images=args.max_images
        )
    else:
        client = VLLMClient(
            base_url=args.base_url, 
            model_name=args.model_name,
            max_size=1024, max_images=args.max_images
        )

    datas = [json.loads(line) for line in open("/mmu_vcg_wjc_hdd/lizhuoqun/metaphor_video_eval/benchmark/datas.jsonl")]
    random.shuffle(datas)
    print(f"total {len(datas)} data")

    save_path = f"/mmu_vcg_wjc_hdd/lizhuoqun/metaphor_video_eval/output/{args.eval_method}_{args.model_name}.jsonl"
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
        for try_time in range(8):
            print(f"try time: {try_time}")

            try:
                image_dir_path=f"/mmu_vcg_wjc_hdd/lizhuoqun/metaphor_video_eval/benchmark/videos/{data['video_id']}"
                assert os.path.exists(image_dir_path), f"!!! {image_dir_path} not exists"

                if args.eval_method == "qa":
                    if data['title'] != "":
                        input_text = "The video title is: " + data['title'] + prompt
                    else:
                        input_text = prompt

                    output, total_output = client.call_openai_vl(prompt=input_text, image_dir_path=image_dir_path)

                    if "</think>" in output:
                        output = output.split("</think>")[-1]
                    if output.startswith('<|begin_of_box|>'):
                        output = output.replace('<|begin_of_box|>', '').replace('<|end_of_box|>', '')

                    output_clear = output.split("```json")[-1].split("```")[0]
                    output_clear = ''.join(char for char in output_clear if ord(char) >= 32 or char in '\n\r\t')
                    output_clear_json = json.loads(output_clear)
                    assert "analysis_dict" in output_clear_json, f"no analysis_dict in output_clear_json: {output_clear_json}"
                    output_clear_json = {"video_id": data['video_id'], "analysis_dict": output_clear_json["analysis_dict"], "total_output": str(total_output)}
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