# MetaphorVU

MetaphorVU: Towards Metaphorical Video Understanding (ICML 2026)

Paper: https://openreview.net/forum?id=yKcBAJMPXZ

Dataset: https://huggingface.co/datasets/lzq2021/MetaphorVU-Bench

# Environment
```bash
python=3.11.13
pip install -r requirements.txt
```

# Evaluation
```bash

# Download Benchmark from Hugging Face
mkdir benchmark
download test.jsonl and videos_deface from https://huggingface.co/datasets/lzq2021/MetaphorVU-Bench/tree/main
mv test.jsonl ./benchmark/datas.jsonl && mv videos_deface ./benchmark/videos

# Prepare LLM API
setting your LLM API and repalce utlis/use_wanqing_api.py

# Run Evalution
mkdir output
bash a_1_eval_vllm_models.sh # will get output file under the ./output, such as qa_GPT-5.jsonl

# Get Score
mkdir score
bash b_1_get_score.sh # will do LLM judge and get score file under the ./score, such as qa_GPT-5.jsonl

# Show Score Table
python c_1_show_score.py # will get csv file of all evaluated MLLMs' score
```

# MetaphorBoost
```bash
# Download Metaphorical Knowledge Graph
download from metaphor_graph_embedding_word.pt and metaphor_graph.json from https://drive.google.com/drive/folders/1nN_ApXmwZW-XIZLREJfPOucIT16R8zlL
mv metaphor_graph_embedding_word.pt ./utlis/metaphor_graph_embedding_word.pt && mv metaphor_graph.json ./utlis/metaphor_graph.json

# Run MetaphorBoost
bash a_0_eval_metaphorvu_boost.sh # will get output file under the ./output, such as metaphorvu_boost_qa_GPT-5_mkg_keywords_simple_10_word_2.jsonl

# Get Score
same as the process in Evaluation
```
