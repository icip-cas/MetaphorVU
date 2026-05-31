# MetaphorVU

MetaphorVU: Towards Metaphorical Video Understanding (ICML 2026)

Email: lizhuoqun2021@iscas.ac.cn

Paper: https://huggingface.co/papers/2605.25461

Dataset: https://huggingface.co/datasets/lzq2021/MetaphorVU-Bench


Metaphorical videos are prevalent across various real-world scenarios to convey complex ideas, and understanding them typically requires high-order cognitive capabilities. The lack of systematic studies on metaphorical video understanding not only constrains the real-world applicability of MLLMs but also impedes the thorough assessment of their high-order cognitive capabilities. To bridge this gap, we propose MetaphorVU-Bench, the first systematic and comprehensive benchmark dedicated to metaphorical video understanding. Through experiments, we find current MLLMs struggle with accurate metaphorical video understanding, lagging far behind human level, primarily due to defective cross-domain mapping. Motivated by this finding, we construct a metaphor knowledge graph as mapping augmentation and propose MetaphorBoost, an inference-time enhancement framework achieving consistent performance improvement. Our benchmark, analysis, and method provide useful insights and a foundation for future research on advancing MLLMs.


<img width="1162" height="709" alt="截屏2026-05-31 15 45 19" src="https://github.com/user-attachments/assets/e2ec7b4c-429b-447f-8760-73b7b311f333" />


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
