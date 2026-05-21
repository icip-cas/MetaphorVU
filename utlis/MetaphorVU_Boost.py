import json
import torch
import random
import torch.nn.functional as F
from torch import Tensor
from typing import List, Dict, Tuple, Optional
from collections import Counter
import os
import requests
from openai import OpenAI


def check_busy(api_url: str):
    print(f"api_url: {api_url}")
    try:
        resp = requests.get(api_url.replace('v1', 'metrics'))
        for line in resp.text.split('\n'):
            if 'num_requests_running' in line and not line.startswith('#'):
                running = int(float(line.split()[-1]))
            elif 'num_requests_waiting' in line and not line.startswith('#'):
                waiting = int(float(line.split()[-1]))
        return running, waiting
    except Exception as e:
        print(f"Error: {e}")
        return 999, 999

# ==================== Embedding API ====================

embedding_client = OpenAI(
    base_url="http://localhost:1225/v1",
    api_key="empty"
)


def get_embedding(text: str) -> list[float]:
    """获取单条文本的 embedding"""
    response = embedding_client.embeddings.create(
        model="Qwen3-Embedding-8B",
        input=text
    )
    return response.data[0].embedding


# ==================== Prompts ====================

PROMPT_EXTRACT_KEYWORDS = \
"""<< Instruction >>
Watch this video carefully and extract all key content elements that appear in the video. These elements will be used for metaphor understanding analysis.

<< Requirements >>
1. Extract all significant visual elements, objects, actions, scenes, symbols, and any notable content.
2. Be comprehensive - don't miss any potentially meaningful elements.
3. Each keyword should be concise but descriptive.
4. Include both concrete objects and abstract concepts if they are clearly presented.
5. Output in JSON format.

<< Output Format >>
```json
{
    "keywords": ["keyword1", "keyword2", "keyword3", ...]
}
```"""

PROMPT_EXTRACT_DESCRIPTIONS = \
"""<< Instruction >>
Watch this video carefully and describe all the key scenes and content that appear in the video. These descriptions will be used for metaphor understanding analysis.

<< Requirements >>
1. Describe all significant visual elements, actions, scenes, symbols, and any notable content.
2. Be comprehensive - don't miss any potentially meaningful scenes or details.
3. Each description should be a short sentence (10-30 words) that captures a specific aspect of the video.
4. Focus on what is actually shown in the video, not interpretations.
5. Include descriptions of both static elements and dynamic actions/transitions.
6. Output in JSON format.

<< Output Format Example >>
```json
{
    "descriptions": [
        "A woman is shown putting on a wedding ring.",
        "The scene transitions to a woman doing household chores.",
        "A child is being cared for by the woman.",
        ...
    ]
}
```"""

PROMPT_FINAL_ANALYSIS = \
"""<< Instruction >>
Analyze the metaphorical logic in this video, i.e., what ideas are implicitly expressed through the content presented.

{title}

<< Requirements >>
1. Thoroughly identify all video content that contains metaphors.
2. Analyze the underlying ideas of the metaphors deeply and accurately.
3. You may refer to the external knowledge below for reference, but your analysis must be grounded in the actual video content, the reference is just for inspiration, do not rely on the reference completely.
4. Avoid baseless assumptions or forced interpretations.
5. If there are multiple elements in the video that contain metaphorical logic, list them separately, with each entry as a concise sentence.
6. Output in JSON format as a dictionary. Each analysis entry should follow the sentence structure: "The video presents *** content, implicitly expressing *** idea."

<< Here are some examples >>
{{
    "analysis_dict": {{
        "analysis_1": "The video presents an out-of-focus rose with raindrops in the foreground, implicitly expressing that the once-beautiful love has become a blurry past, leaving only tears and sadness.",
        "analysis_2": "The video presents a gloomy rainy scene, implicitly expressing that the protagonist's inner world is filled with darkness, sorrow, and despair after the breakup.",
        "analysis_3": "The video presents a view from indoors, looking outward through obstructing flowers, implicitly expressing that the protagonist is trapped in sad memories, isolated from the outside world and unable to move forward.",
        "analysis_4": "The video presents pedestrians and vehicles receding into the background, implicitly expressing that the other person has already left, life goes on, but the protagonist has been left behind.",
        "analysis_5": "The video presents the process of the screen gradually darkening from bright, implicitly expressing the protagonist's sinking mood and slowly extinguishing hope after the breakup."
    }}
}}

{{
    "analysis_dict": {{
        "analysis_1": "The video shows the protagonist excitedly running in place at the school entrance on the first day of enrollment, implicitly expressing the student's hopeful, optimistic, and prepared-to-embrace-challenges positive mindset at the beginning of the new semester.",
        "analysis_2": "The video shows the protagonist dancing freely on the playground one week after enrollment, implicitly expressing that the student is still in a relatively relaxed and enjoyable stage of new life shortly after the semester starts.",
        "analysis_3": "The video shows the protagonist screaming desperately in the snow one month after enrollment, implicitly expressing that the student, after experiencing continuous academic pressure, has exhausted their initial passion and fallen into a state of mental fatigue and breakdown.",
        "analysis_4": "The video presents the complete process of the protagonist transitioning from excited running to relaxed dancing and then to desperate screaming, as a whole implicitly expressing the brutal process of the student's mental state rapidly sliding from idealism to reality under the weight of academic pressure after the semester begins."
    }}
}}

{{
    "analysis_dict": {{
        "analysis_1": "The video presents a scene of a group of adults imitating childhood games, playing and running on the road, implicitly expressing the idea that friendship which allows people to put aside their adult identities and relive the pure joy of childhood is incredibly precious.",
        "analysis_2": "The video combines the real-life scene of young people playing with animated childhood memories they share, implicitly expressing the idea of profound nostalgia and cherishing for that carefree, fantasy-filled collective time of childhood."
    }}
}}

<< External knowledge for reference >>
Based on the video content, here are some relevant metaphorical associations from a knowledge graph that may help your analysis:
NOTE: these associations are just for reference, do not completely rely them.
{external_reference}

<< Output Format >>
```json
{{
    "analysis_dict": {{
        "analysis_1": "The video presents the *** content, implicitly expressing the *** idea",
        "analysis_2": "The video presents the *** content, implicitly expressing the *** idea"
    }}
}}
```"""


class MetaphorVU_Boost:
    def __init__(
        self,
        vlm_client_list,

        max_retry: int = 5,

        aug_mode: str = "mkg", 
        query_mode: str = "keywords", 
        reference_mode: str = "simple",
        top_k: int = 10,
        indexing_type: str = "word",
        num_hops: int = 1 
    ):
        self.vlm_client_list = vlm_client_list
        self.vlm_client = self.vlm_client_list[0]

        self.max_retry = max_retry

        self.aug_mode = aug_mode
        self.query_mode = query_mode
        self.reference_mode = reference_mode
        self.top_k = top_k
        self.indexing_type = indexing_type
        self.num_hops = num_hops 

        assert aug_mode in ["mkg", "ckg", "rag", "self"], f'aug_mode must be "mkg" or "ckg" or "rag" or "self", got "{aug_mode}"'
        assert query_mode in ["keywords", "descriptions"], f'query_mode must be "keywords" or "descriptions", got "{query_mode}"'
        assert reference_mode in ["simple", "examples"], f'reference_mode must be "simple" or "examples", got "{reference_mode}"'
        assert indexing_type in ["word", "word_neighbors"], f'indexing_type must be "word" or "word_neighbors", got "{indexing_type}"'
        assert num_hops >= 1, f'num_hops must be >= 1, got {num_hops}'  
        
        print(f">>>>> Initializing MetaphorVU_Boost with parameters:")
        
        if self.aug_mode == "mkg":

            print("Loading metaphor graph")
            with open('/mmu_vcg_wjc_hdd/lizhuoqun/metaphor_video_eval/utlis/metaphor_graph.json', 'r', encoding='utf-8') as f:
                self.graph = json.load(f)
            print(f"Graph loaded with {len(self.graph)} nodes")

            print("Loading graph embeddings")
            self.graph_embeddings = torch.load(f'/mmu_vcg_wjc_hdd/lizhuoqun/metaphor_video_eval/utlis/metaphor_graph_embedding_{self.indexing_type}.pt', map_location='cpu')
            print(f"Graph embeddings shape: {self.graph_embeddings.shape}")
            
            self.word_to_idx = {}
            for idx_str, node in self.graph.items():
                self.word_to_idx[node["word"]] = int(idx_str)
                
        else:
            raise NotImplementedError(f"Augmentation mode '{self.aug_mode}' is not implemented yet.")

        print(f"> Aug mode: {aug_mode}")
        print(f"> Query mode: {query_mode}")
        print(f"> Reference mode: {reference_mode}")
        print(f"> Top K: {top_k}")
        print(f"> Indexing type: {indexing_type}")
        print(f"> Num hops: {num_hops}")  
        print(f">>>>> Initialization completed")
    
    def select_best_vlm_client(self):

        best_client = random.choice(self.vlm_client_list)
        min_load = float('inf')
        random.shuffle(self.vlm_client_list)
        
        for client in self.vlm_client_list:
            running, waiting = check_busy(client.base_url)
            load = running + waiting
            if load < min_load:
                min_load = load
                best_client = client

        print(f"best_client: {best_client.base_url}")
        return best_client

    def get_text_embedding(self, texts: List[str]) -> Tensor:

        all_embeddings = []
        
        for text in texts:
            emb = get_embedding(text)
            all_embeddings.append(emb)
        
        embeddings = torch.tensor(all_embeddings, dtype=torch.float32, device=torch.device('cpu'))
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def retrieve_subnetwork(self, queries: List[str]) -> Tuple[List[Dict], List[Tuple[str, int, Dict]]]:

        if not queries:
            return [], []
        
        query_texts = [f"This is a scene description related to metaphor understanding: {q}" for q in queries]

        query_embeddings = self.get_text_embedding(query_texts)  # [num_queries, dim]

        similarities = query_embeddings @ self.graph_embeddings.T  # [num_queries, num_nodes]

        retrieved_indices = []
        for i in range(len(queries)):
            top_idx = similarities[i].argmax().item()
            retrieved_indices.append((queries[i], top_idx))

        unique_indices = list(set([idx for _, idx in retrieved_indices]))

        retrieved_nodes = []
        for idx in unique_indices:
            node = self.graph[str(idx)]
            retrieved_nodes.append({
                "node_id": idx,
                "word": node["word"],
                "adjacency": node.get("adjacency", {})
            })

        adjacency_counter = Counter() 
        adjacency_info = {}  
        
        for query, idx in retrieved_indices:
            node = self.graph[str(idx)]
            adjacency = node.get("adjacency", {})
            
            for adj_word, adj_info in adjacency.items():
                adjacency_counter[adj_word] += 1
                
                if adj_word not in adjacency_info:
                    adjacency_info[adj_word] = {
                        "source_words": [],
                        "sentences": []
                    }

                adjacency_info[adj_word]["source_words"].append(query)
                
                sentences = adj_info.get("sentences", [])
                for sent in sentences:
                    if sent not in adjacency_info[adj_word]["sentences"]:
                        adjacency_info[adj_word]["sentences"].append(sent)
        
        sorted_adjacencies = adjacency_counter.most_common(self.top_k)
        
        top_adjacencies = []
        for adj_word, count in sorted_adjacencies:
            info = adjacency_info[adj_word]
            top_adjacencies.append((adj_word, count, info))
        
        return retrieved_nodes, top_adjacencies
    
    def retrieve_subnetwork_multi_hop(self, queries: List[str]) -> Tuple[List[Dict], List[Tuple[str, int, Dict]]]:

        if not queries:
            return [], []
        
        query_texts = [f"This is a scene description related to metaphor understanding: {q}" for q in queries]

        query_embeddings = self.get_text_embedding(query_texts)  # [num_queries, dim]
        
        similarities = query_embeddings @ self.graph_embeddings.T  # [num_queries, num_nodes]

        query_node_pairs = []
        for i in range(len(queries)):
            top_idx = similarities[i].argmax().item()
            query_node_pairs.append((queries[i], top_idx))

        unique_start_indices = list(set([idx for _, idx in query_node_pairs]))

        retrieved_nodes = []
        for idx in unique_start_indices:
            node = self.graph[str(idx)]
            retrieved_nodes.append({
                "node_id": idx,
                "word": node["word"],
                "adjacency": node.get("adjacency", {})
            })

        adjacency_counter = Counter()  
        adjacency_info = {}  

        for query, start_idx in query_node_pairs:  

            visited = {start_idx: 0}  
            queue = [(start_idx, 0, query)] 
            
            while queue:
                current_idx, current_hop, source_query = queue.pop(0)

                if current_hop < self.num_hops:
                    current_node = self.graph[str(current_idx)]
                    adjacency = current_node.get("adjacency", {})
                    
                    for adj_word, adj_info in adjacency.items():

                        adjacency_counter[adj_word] += 1

                        if adj_word not in adjacency_info:
                            adjacency_info[adj_word] = {
                                "source_words": [],
                                "sentences": [],
                                "min_hop": current_hop + 1  
                            }

                        adjacency_info[adj_word]["source_words"].append(source_query)
                        adjacency_info[adj_word]["min_hop"] = min(
                            adjacency_info[adj_word]["min_hop"], 
                            current_hop + 1
                        )

                        sentences = adj_info.get("sentences", [])
                        for sent in sentences:
                            if sent not in adjacency_info[adj_word]["sentences"]:
                                adjacency_info[adj_word]["sentences"].append(sent)

                        if adj_word in self.word_to_idx:
                            adj_idx = self.word_to_idx[adj_word]
                            if adj_idx not in visited:
                                visited[adj_idx] = current_hop + 1
                                queue.append((adj_idx, current_hop + 1, source_query))

        sorted_adjacencies = adjacency_counter.most_common(self.top_k)

        top_adjacencies = []
        for adj_word, count in sorted_adjacencies:
            info = adjacency_info[adj_word]
            top_adjacencies.append((adj_word, count, info))
        
        return retrieved_nodes, top_adjacencies

    def format_external_reference(
        self, 
        retrieved_nodes: List[Dict], 
        top_adjacencies: List[Tuple[str, int, Dict]],
        max_examples_per_adj: int = 1
    ) -> str:

        if not top_adjacencies:
            return ""
        
        parts = []
        if self.reference_mode == "simple":

            for adj_word, count, info in top_adjacencies:
                source_words = list(set(info["source_words"])) 
                source_words_str = ", ".join([f'"{w}"' for w in source_words])
                entry = f"• The concept {source_words_str} is possiblely associated with \"{adj_word}\""
                parts.append(entry)
        elif self.reference_mode == "examples":

            for adj_word, count, info in top_adjacencies:
                source_words = list(set(info["source_words"]))
                source_words_str = ", ".join([f'"{w}"' for w in source_words])
                sentences = info["sentences"][:max_examples_per_adj]
                entry = f"• The concept {source_words_str} is possiblely associated with \"{adj_word}\""
                if sentences:
                    entry += "\n  Example usage:"
                    for i, sent in enumerate(sentences, 1):

                        clean_sent = sent.replace('\n', ' ').strip()
                        entry += f"\n    {i}. {clean_sent}"
                parts.append(entry)
        else:
            raise ValueError(f"Unknown reference_mode: {self.reference_mode}")

        result = "\n".join(parts)
        if len(result) > 4000:
            result = result[:4000] + "\n... (more associations omitted)"
        
        return result
    
    def extract_queries(self, image_dir_path: str) -> Tuple[List[str], str, any]:

        print("=" * 50)
        print(f"Step 1: Extracting {'keywords' if self.query_mode == 'keywords' else 'descriptions'} from video")
        print("=" * 50)
        
        if self.query_mode == "keywords":
            prompt = PROMPT_EXTRACT_KEYWORDS
            output_key = "keywords"
        else:  
            prompt = PROMPT_EXTRACT_DESCRIPTIONS
            output_key = "descriptions"
        
        output, total_output, saved_image_messages = self.vlm_client.call_openai_vl(
            prompt=prompt,
            image_dir_path=image_dir_path,
            return_image_messages=True
        )
        
        output_json = self.get_clear_output_json(output)
        
        assert output_key in output_json, f'"{output_key}" not in output_json: {output_json}'
        assert isinstance(output_json[output_key], list), f'{output_key} is not a list: {output_json}'
        
        queries = output_json[output_key]
        print(f"Extracted {len(queries)} {self.query_mode}:")
        for i, q in enumerate(queries):
            print(f"  {i+1}. {q}")
        
        return queries, output, saved_image_messages
    
    def final_analysis(
        self,
        external_reference: str,
        saved_image_messages: any,
        image_dir_path: str,
        title: str
    ) -> Tuple[Dict, str]:

        print("=" * 50)
        print("Step 3: Final metaphor analysis with external reference")
        print("=" * 50)
        
        prompt = PROMPT_FINAL_ANALYSIS.format(external_reference=external_reference, title=title)
        
        if saved_image_messages is not None:
            output, total_output = self.vlm_client.call_openai_vl(
                prompt=prompt,
                saved_image_messages=saved_image_messages
            )
        else:
            output, total_output = self.vlm_client.call_openai_vl(
                prompt=prompt,
                image_dir_path=image_dir_path
            )
        
        output_json = self.get_clear_output_json(output)
        
        assert "analysis_dict" in output_json, f'"analysis_dict" not in output_json: {output_json}'
        
        analysis_dict = output_json["analysis_dict"]
        print(f"Generated {len(analysis_dict)} analysis entries")
        
        return analysis_dict, output
    
    def process(self, image_dir_path: str, title: str) -> Tuple[Dict, Dict]:

        process_record = {
            "config": {
                "aug_mode": self.aug_mode,
                "query_mode": self.query_mode,
                "reference_mode": self.reference_mode,
                "top_k": self.top_k
            }
        }
        
        for retry in range(self.max_retry):
            print(f"\n{'#' * 60}")
            print(f"Process Attempt {retry + 1}/{self.max_retry}")
            print(f"Config: aug_mode={self.aug_mode}, query_mode={self.query_mode}, reference_mode={self.reference_mode}")
            print(f"{'#' * 60}\n")
            
            self.vlm_client = self.select_best_vlm_client()

            try:

                queries, queries_output, saved_image_messages = self.extract_queries(image_dir_path)
                process_record["step1_extraction"] = {
                    "mode": self.query_mode,
                    "queries": queries,
                    "original_output": queries_output
                }
                

                print("=" * 50)
                print(f"Step 2: Do augmentation using {self.aug_mode}")
                print("=" * 50)

                if self.top_k > 0:
                    if self.aug_mode == 'mkg':
                        if self.num_hops > 1:
                            retrieved_nodes, top_adjacencies = self.retrieve_subnetwork_multi_hop(queries)
                        else:
                            retrieved_nodes, top_adjacencies = self.retrieve_subnetwork(queries)
                        external_reference = self.format_external_reference(retrieved_nodes, top_adjacencies)                    
                        process_record["step2_retrieval"] = {
                            "num_retrieved_nodes": len(retrieved_nodes),
                            "retrieved_nodes": retrieved_nodes,
                            "top_adjacencies": [(adj, cnt, {"source_words": info["source_words"], "num_sentences": len(info["sentences"])}) for adj, cnt, info in top_adjacencies],
                            "external_reference": external_reference
                        }
                    else:
                        raise ValueError(f"Unknown aug_mode: {self.aug_mode}")
                else:
                    external_reference = ""
                    
                print(f"\nExternal reference:\n{external_reference}")

                analysis_dict, analysis_output = self.final_analysis(
                    external_reference=external_reference, 
                    saved_image_messages=saved_image_messages, 
                    image_dir_path=image_dir_path,
                    title=title
                )
                
                process_record["step3_analysis"] = {
                    "analysis_dict": analysis_dict,
                    "original_output": analysis_output
                }
                
                final_answer = {"analysis_dict": analysis_dict}
                
                print("=" * 50)
                print("Process completed successfully!")
                print(f"Final answer: {json.dumps(final_answer, ensure_ascii=False, indent=2)}")
                print("=" * 50)
                
                return final_answer, process_record
            
            except Exception as e:
                print(f"Error in process attempt {retry + 1}: {e}")
                process_record[f"error_attempt_{retry + 1}"] = str(e)
                continue
        
        print("=" * 50)
        print("Process failed after all retries!")
        print("=" * 50)
        
        return "error", process_record
    
    def get_clear_output_json(self, output: str) -> Dict:
        if "</think>" in output:
            output = output.split("</think>")[-1]
        if output.startswith('<|begin_of_box|>'):
            output = output.replace('<|begin_of_box|>', '').replace('<|end_of_box|>', '')
        
        output_clear = output.split("```json")[-1].split("```")[0]
        output_clear = ''.join(char for char in output_clear if ord(char) >= 32 or char in '\n\r\t')
        
        output_json = json.loads(output_clear)
        
        return output_json