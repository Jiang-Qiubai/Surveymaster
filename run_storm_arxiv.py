import os
import re
import argparse
import logging
import arxiv
import dspy
import sys
from sentence_transformers import SentenceTransformer

# --- 1. LiteLLM 静音配置 ---
os.environ["LITELLM_LOG"] = "ERROR" 
import litellm
litellm.suppress_debug_info = True
litellm.set_verbose = False
litellm.drop_params = True

from knowledge_storm import STORMWikiRunnerArguments, STORMWikiRunner, STORMWikiLMConfigs
from knowledge_storm.lm import LitellmModel
from knowledge_storm.utils import load_api_key

# 引入需要 Monkey Patch 的模块
import knowledge_storm.storm_wiki.modules.persona_generator as pg_module
import knowledge_storm.storm_wiki.modules.storm_dataclass as storm_dataclass
# 引入需要修改 Prompt 的模块 (这是关键)
import knowledge_storm.storm_wiki.modules.outline_generation as outline_gen_module
import knowledge_storm.storm_wiki.modules.article_generation as article_gen_module

# ==============================================================================
# 0. 硬编码配置区域
# ==============================================================================
LLM_API_URL = "http://172.17.0.1:8091/v1"
LLM_MODEL_NAME = "openai/qwen3-32b"
# 为了支持长文，检索深度必须增加
SEARCH_TOP_K = 10      # 每一轮搜索更多的论文 (原5)
RETRIEVE_TOP_K = 10    # 每一段写作参考更多的片段 (原5)
MAX_CONV_TURNS = 5     # 增加对话轮数以覆盖更多子话题 (原3)
MAX_TOKENS_WRITE = 16384 # 写作时允许的最大输出长度

# 本地 Embedding 模型路径
LOCAL_EMBEDDING_PATH = "~/models/paraphrase-MiniLM-L6-v2"

# ==============================================================================
# 2. Monkey Patch 区域 (保持不变，确保网络和Embedding正常)
# ==============================================================================
def bypass_wiki_access(url):
    return "Academic Topic Placeholder", "Content skipped: Local Academic Mode."

pg_module.get_wiki_page_title_and_toc = bypass_wiki_access

def local_prepare_table_for_retrieval(self):
    model_path = os.path.expanduser(LOCAL_EMBEDDING_PATH)
    print(f"🧠 [Embedding] Loading local model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Local embedding model not found at {model_path}")
    self.encoder = SentenceTransformer(model_path, device="cpu", local_files_only=True)
    self.collected_urls = []
    self.collected_snippets = []
    for url, information in self.url_to_info.items():
        for snippet in information.snippets:
            self.collected_urls.append(url)
            self.collected_snippets.append(snippet)
    self.encoded_snippets = self.encoder.encode(self.collected_snippets, show_progress_bar=False)

storm_dataclass.StormInformationTable.prepare_table_for_retrieval = local_prepare_table_for_retrieval

# ==============================================================================
# 1. 自定义 LLM 类 (保持清洗逻辑)
# ==============================================================================
class CleanLitellmModel(LitellmModel):
    def __call__(self, prompt, **kwargs):
        kwargs['logger_fn'] = None 
        kwargs['verbose'] = False
        outputs = super().__call__(prompt, **kwargs)
        cleaned_outputs = []
        for out in outputs:
            if not isinstance(out, str):
                cleaned_outputs.append(out)
                continue
            cleaned = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()
            cleaned = re.sub(r"^<think>.*", "", cleaned, flags=re.DOTALL).strip()
            cleaned = re.sub(
                r"^(Okay|Sure|Here is|Certainly|Let's|To answer|Great|I can help|Based on).*?[:\n]", 
                "", cleaned, flags=re.IGNORECASE | re.MULTILINE
            ).strip()
            json_match = re.search(r"```json(.*?)```", cleaned, re.DOTALL)
            if json_match:
                cleaned = json_match.group(1).strip()
            elif "```" in cleaned:
                code_match = re.search(r"```(.*?)```", cleaned, re.DOTALL)
                if code_match:
                    cleaned = code_match.group(1).strip()
            cleaned_outputs.append(cleaned)
        return cleaned_outputs

# ==============================================================================
# 3. ArXiv 检索模块 (保持防崩溃)
# ==============================================================================
class ArXivSearch(dspy.Retrieve):
    def __init__(self, k=5, category="cs.SE"):
        super().__init__(k=k)
        self.k = k
        self.category = category
        self.client = arxiv.Client()

    def forward(self, query_or_queries, exclude_urls=[]):
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        collected_results = []
        for query in queries:
            safe_query = query.replace(':', ' ').replace('-', ' ').replace('"', '').strip()
            safe_query = re.sub(r'\b(query|queries)\b', '', safe_query, flags=re.IGNORECASE).strip()
            if not safe_query or '<' in safe_query: continue
            search_query = f'{safe_query}'
            if self.category:
                search_query += f' AND cat:{self.category}'
            
            print(f"🔍 [ArXiv] Searching: {search_query}")
            try:
                search = arxiv.Search(query=search_query, max_results=self.k, sort_by=arxiv.SortCriterion.Relevance)
                results = list(self.client.results(search))
                for r in results:
                    collected_results.append({
                        'url': r.entry_id,
                        'title': r.title.replace('\n', ' '),
                        'description': r.summary.replace('\n', ' '),
                        'snippets': [r.summary.replace('\n', ' ')] 
                    })
            except Exception as e:
                print(f"⚠️ ArXiv Search Error: {e}")
        
        if not collected_results:
            print("⚠️ [Warning] No results found. Returning placeholder.")
            collected_results.append({
                'url': 'http://placeholder/no-results',
                'title': 'No Academic Papers Found',
                'description': 'Search returned no results.',
                'snippets': ['No relevant information found in ArXiv.']
            })
        return collected_results

# ==============================================================================
# 4. 主程序 (注入核心逻辑)
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="STORM ArXiv Deep Review (Long Version)")
    parser.add_argument('--topic', type=str, required=True, help="论文综述主题")
    parser.add_argument('--output-dir', type=str, default="./results/arxiv_review_long", help="输出目录")
    parser.add_argument('--arxiv-category', type=str, default="cs.AI", help="ArXiv分类")
    
    args = parser.parse_args()

    # 1. 题目增强：强制加上 "Comprehensive Survey" 等词，引导模型往大了写
    survey_topic = f"A Comprehensive and Deep Academic Survey on {args.topic}: Theories, Methodologies, and Future Directions"
    print(f"🚀 启动深度综述任务: {survey_topic}")
    print(f"🤖 连接 LLM: {LLM_MODEL_NAME}")

    # 2. 初始化模型 (增加 max_tokens 以支持长文生成)
    conv_lm = CleanLitellmModel(
        model=LLM_MODEL_NAME, api_key='EMPTY', api_base=LLM_API_URL, model_type='chat',  
        max_tokens=2048, temperature=0.8, top_p=0.9
    )
    article_lm = CleanLitellmModel(
        model=LLM_MODEL_NAME, api_key='EMPTY', api_base=LLM_API_URL, model_type='chat', 
        max_tokens=MAX_TOKENS_WRITE, # 允许生成更长的段落
        temperature=0.7, # 稍微提高温度，增加分析的多样性
        top_p=0.9
    )

    lm_config = STORMWikiLMConfigs()
    lm_config.set_conv_simulator_lm(conv_lm)
    lm_config.set_question_asker_lm(conv_lm)
    lm_config.set_outline_gen_lm(article_lm)
    lm_config.set_article_gen_lm(article_lm)
    lm_config.set_article_polish_lm(article_lm)

    rm = ArXivSearch(k=SEARCH_TOP_K, category=args.arxiv_category)

    runner_args = STORMWikiRunnerArguments(
        output_dir=args.output_dir,
        max_conv_turn=MAX_CONV_TURNS, # 增加轮数，搜集更多信息
        max_search_queries_per_turn=3,
        retrieve_top_k=RETRIEVE_TOP_K, # 增加阅读量，为长文提供素材
        max_thread_num=4, 
        search_top_k=SEARCH_TOP_K
    )

    runner = STORMWikiRunner(runner_args, lm_config, rm)

    # ==============================================================================
    # 🔥 核心修改：注入深度 Prompt (基于你的建议)
    # ==============================================================================
    print("🎨 注入深度学术分析 Prompt...")
    
    # 1. 大纲生成 Prompt：强制要求包含历史、方法论对比、理论框架等章节
    outline_gen_module.WritePageOutline.__doc__ = """
    Write a highly detailed, comprehensive academic literature review outline for the given topic.
    
    REQUIRED STRUCTURE (Must include these perspectives):
    1. **Historical Context**: Evolution of the field, key milestones.
    2. **Theoretical Foundations**: Core definitions, conflicting theories, theoretical evolution.
    3. **Methodological Analysis**: 
       - Compare different approaches (e.g., Qualitative vs Quantitative, Deep Learning vs Traditional).
       - Discuss advantages and limitations of each method.
    4. **Empirical Evidence**: Categorize studies by design, sample, or setting. Explain contradictory results.
    5. **Critical Gaps & Future Directions**: Unsolved problems, emerging trends.
    
    FORMATTING:
    - Use "#" for main sections and "##", "###" for subsections.
    - Ensure the outline is deep and granular (aim for at least 8-10 main sections).
    - Do not include the topic name itself as a section header.
    """

    # 2. 章节写作 Prompt：强制要求“分析性写作”而非“描述性写作”
    # 将你提供的“段落级扩写技巧”写入 Prompt
    article_gen_module.WriteSection.__doc__ = """
    Write an extensive, analytical, and critical academic review section based on the collected information.
    
    CRITICAL WRITING GUIDELINES (Follow strictly):
    1. **Shift from Description to Analysis**:
       - BAD: "Study A found X. Study B found Y."
       - GOOD: "Study A found X, whereas Study B proposed Y. This discrepancy may stem from methodological differences..."
    2. **Paragraph Structure**:
       - Start with a **Core Argument**.
       - Provide **Supporting Evidence** from multiple sources.
       - Introduce **Contrasting Views** or turning points.
       - Analyze the **Methodological Reasons** for differences.
       - Conclude with **Implications** or Gaps.
    3. **Depth**:
       - Each subsection must be substantive (aim for 500-800 words per section if info permits).
       - Compare and contrast theories/methods explicitly.
    4. **Citations**:
       - Use [1], [2] format inline.
       - Do not create a separate Reference list.
    """

    # ==============================================================================
    # 7. 开始运行
    # ==============================================================================
    print("🏁 开始执行深度 STORM 流程 (预计耗时较长)...")
    
    try:
        runner.run(
            topic=survey_topic,
            do_research=True,
            do_generate_outline=True,
            do_generate_article=True,
            do_polish_article=True
        )
        runner.post_run()
        runner.summary()
        print(f"✅ 综述完成！结果已保存至: {os.path.join(args.output_dir, runner.article_dir_name)}")
        
    except Exception as e:
        print(f"❌ 运行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    logging.getLogger("LiteLLM").setLevel(logging.WARNING) 
    logging.getLogger("litellm").setLevel(logging.WARNING) 
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("arxiv").setLevel(logging.WARNING)
    logging.getLogger("dspy").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    main()