import json
import os
import re
from typing import List, Dict, Any, Tuple
import numpy as np
import jieba
from rank_bm25 import BM25Okapi
from openai import OpenAI

API_KEY = "your_api_key"
BASE_URL = "your_api_base_url"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

MODEL_LLM = "Qwen/Qwen3.5-4B"
MODEL_EMBEDDING = "Qwen/Qwen3-Embedding-0.6B"

DENSE_WEIGHT = 0.5
BM25_WEIGHT = 0.5
RERANK_TOP_K = 5
REFUSAL_THRESHOLD = 0.35

class LibraryKnowledgeBase:
    def __init__(self):
        self.chunks: List[Dict[str, Any]] = []
        self.bm25: BM25Okapi = None
        self.embeddings: np.ndarray = None

    def load_corpus(self, corpus_path: str):
        print(f"正在加载知识库: {corpus_path}")
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.chunks.append(json.loads(line.strip()))
        
        print(f"成功加载 {len(self.chunks)} 个知识片段。开始构建索引...")
        self._build_bm25_index()
        self._build_embedding_index()

    def _tokenize(self, text: str) -> List[str]:
        return [w for w in jieba.cut(text) if w.strip()]

    def _build_bm25_index(self):
        corpus_tokenized = [self._tokenize(chunk["text"]) for chunk in self.chunks]
        self.bm25 = BM25Okapi(corpus_tokenized)

    def _get_embedding(self, text: str) -> List[float]:
        # 调用 API 获取文本嵌入向量
        response = client.embeddings.create(
            model=MODEL_EMBEDDING,
            input=text
        )
        return response.data[0].embedding

    def _build_embedding_index(self):
        all_embeddings = []
        for chunk in self.chunks:
            # 融合 title 和 text，提高检索召回率
            full_text = f"标题: {chunk['title']} \n正文: {chunk['text']}"
            all_embeddings.append(self._get_embedding(full_text))
        self.embeddings = np.array(all_embeddings)

    def hybrid_search(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        # 1. 密集向量检索 (Cosine Similarity)
        query_emb = np.array(self._get_embedding(query))
        # 归一化后点积即为余弦相似度
        norm_emb = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norm_query = query_emb / np.linalg.norm(query_emb)
        dense_scores = np.dot(norm_emb, norm_query)
        # 归一化到 0-1
        dense_scores = (dense_scores - np.min(dense_scores)) / (np.max(dense_scores) - np.min(dense_scores) + 1e-5)

        # 2. 稀疏文本检索 (BM25)
        query_tokenized = self._tokenize(query)
        bm25_scores = np.array(self.bm25.get_scores(query_tokenized))
        # 归一化到 0-1
        if np.max(bm25_scores) > 0:
            bm25_scores = bm25_scores / np.max(bm25_scores)

        # 3. 混合打分机制
        hybrid_scores = DENSE_WEIGHT * dense_scores + BM25_WEIGHT * bm25_scores
        
        # 4. 排序并截取 Top K
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(hybrid_scores[idx])))
        return results


class LibraryAnsweringSystem:
    def __init__(self, kb: LibraryKnowledgeBase):
        self.kb = kb
        self.default_refusal = "无法根据给定知识库回答"

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        response = client.chat.completions.create(
            model=MODEL_LLM,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=200,
            extra_body={"enable_thinking": False},
        )
        return response.choices[0].message.content.strip()

    def answer_question(self, question: str) -> Dict[str, Any]:
        # 混合检索
        search_results = self.kb.hybrid_search(question, top_k=RERANK_TOP_K)
        
        # 检索强拒答阈值拦截
        max_search_score = search_results[0][1] if search_results else 0.0
        if max_search_score < REFUSAL_THRESHOLD:
            return {"answer": self.default_refusal, "citations": []}

        # 构造上下文
        context_str = ""
        for i, (chunk, score) in enumerate(search_results):
            context_str += f"[文档片段编号: {chunk['chunk_id']}]\n标题: {chunk['title']}\n内容: {chunk['text']}\n\n"

        system_prompt = (
            "你是一个严谨的图书馆问答员。请根据提供的本地知识库片段回答用户问题。\n"
            "【规则要求】:\n"
            "1. 你的回答必须简短、准确、事实完全基于给定的片段。不得发挥，不得包含外部常识。\n"
            "2. 如果给定的片段中没有任何一句话能够回答该问题，或者给定的信息与问题冲突、不完整，你必须直接输出: 无法根据给定知识库回答。\n"
            "3. 你的输出格式必须是严格的 JSON 对象，不能包含任何 markdown 格式标记 (如 ```json) 或其他文字。格式如下：\n"
            '{\n  "answer": "这里写你的精确简短回答",\n  "citations": ["支撑答案的 chunk_id"]\n}'
        )

        user_prompt = f"【本地知识库片段】:\n{context_str}\n【用户问题】:\n{question}\n\n请输出你的JSON回答:"

        llm_output = self._call_llm(system_prompt, user_prompt)

        
        try:
            # 移除可能误生成的 markdown 标记
            clean_output = re.sub(r"```json|```", "", llm_output).strip()
            res_json = json.loads(clean_output)
            
            # 规范格式化字段
            answer = res_json.get("answer", "").strip()
            citations = res_json.get("citations", [])

            # 如果模型在文本里表达了拒绝，或者输出为空，强制归一化
            if self.default_refusal in answer or not answer:
                return {"answer": self.default_refusal, "citations": []}

            # 验证引用确实包含在本次检索到的范围内，防止模型“幻觉伪造”引用
            valid_chunk_ids = {c["chunk_id"] for c, _ in search_results}
            final_citations = [c for c in citations if c in valid_chunk_ids][:5]

            return {
                "answer": answer,
                "citations": final_citations
            }

        except Exception:
            return {"answer": self.default_refusal, "citations": []}



corpus_file = "corpus.jsonl"
test_file = "test_questions.jsonl"
output_file = "results.json"

temp_output_file = "results_checkpoint.jsonl"

kb = LibraryKnowledgeBase()
kb.load_corpus(corpus_file)
qa_system = LibraryAnsweringSystem(kb)


processed_ids = set()
if os.path.exists(temp_output_file):
    #有时会超时卡死，导致之前的结果无法写入最终文件
    with open(temp_output_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line.strip())
                    processed_ids.add(record["id"])
                except Exception:
                    pass
    print(f"已成功跳过 {len(processed_ids)} 个已处理的问题。")


with open(temp_output_file, "a", encoding="utf-8") as f_out:
    with open(test_file, "r", encoding="utf-8") as f_in:
        for line in f_in:
            if not line.strip():
                continue
            item = json.loads(line.strip())
            q_id = item["id"]
            question = item["question"]
            
            if q_id in processed_ids:
                continue
            
            print(f"正在处理问题 [{q_id}]: {question}")
            
            
            result = qa_system.answer_question(question)
            
            record = {
                "id": q_id,
                "answer": result["answer"],
                "citations": result["citations"]
            }
            
            
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush() 


submission_results = []
with open(temp_output_file, "r", encoding="utf-8") as f_temp:
    for line in f_temp:
        if line.strip():
            submission_results.append(json.loads(line.strip()))

with open(output_file, "w", encoding="utf-8") as f_final:
    json.dump(submission_results, f_final, ensure_ascii=False, indent=2)

print(f"处理完成")