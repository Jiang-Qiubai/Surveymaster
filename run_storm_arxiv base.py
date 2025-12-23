import os
import re
import argparse
import logging
import arxiv
import dspy
import sys
# 引入 sentence_transformers 以便我们在 patch 中使用
from sentence_transformers import SentenceTransformer

# --- 1. LiteLLM 静音配置 (防止红色日志刷屏) ---
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
import knowledge_storm.storm_wiki.modules.storm_dataclass as storm_dataclass # <--- 新增引入
import knowledge_storm.storm_wiki.modules.outline_generation as outline_gen_module
import knowledge_storm.storm_wiki.modules.article_generation as article_gen_module

# ==============================================================================
# 0. 硬编码配置区域
# ==============================================================================
LLM_API_URL = "http://172.17.0.1:8091/v1"
LLM_MODEL_NAME = "openai/qwen3-32b"
CONTEXT_WINDOW = 32768
# 本地 Embedding 模型路径 (支持 ~ 展开)
LOCAL_EMBEDDING_PATH = "~/models/paraphrase-MiniLM-L6-v2"

# ==============================================================================
# 2. Monkey Patch 区域 (核心修复)
# ==============================================================================

# --- Patch 1: 屏蔽 Wikipedia ---
def bypass_wiki_access(url):
    return "Academic Topic Placeholder", "Content skipped: Local Academic Mode."

pg_module.get_wiki_page_title_and_toc = bypass_wiki_access

# --- Patch 2: 强制使用本地 Embedding 模型 (修复网络超时) ---
def local_prepare_table_for_retrieval(self):
    # 展开用户目录 ~
    model_path = os.path.expanduser(LOCAL_EMBEDDING_PATH)
    print(f"🧠 [Embedding] Loading local model from: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Local embedding path not found: {model_path}")
        # 如果找不到本地模型，你可以选择抛出异常或回退
        raise FileNotFoundError(f"Local embedding model not found at {model_path}")

    # 关键修改：传入本地路径，并强制 local_files_only=True
    self.encoder = SentenceTransformer(model_path, device="cpu", local_files_only=True)
    
    # 原有逻辑保持不变
    self.collected_urls = []
    self.collected_snippets = []
    for url, information in self.url_to_info.items():
        for snippet in information.snippets:
            self.collected_urls.append(url)
            self.collected_snippets.append(snippet)
    
    # 编码时关闭进度条，减少日志干扰
    self.encoded_snippets = self.encoder.encode(self.collected_snippets, show_progress_bar=False)

# 应用 Patch 覆盖原方法
storm_dataclass.StormInformationTable.prepare_table_for_retrieval = local_prepare_table_for_retrieval


# ==============================================================================
# 1. 自定义 LLM 类 (清洗输出)
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
            
            # 清洗 <think> 和口语
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
# 3. ArXiv 检索模块 (防崩溃)
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
            # 移除 query/queries 这种模型生成的元词汇
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
# 4. 主程序
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="STORM ArXiv Review (Local Embedding Fix)")
    parser.add_argument('--topic', type=str, required=True, help="论文综述主题")
    parser.add_argument('--output-dir', type=str, default="./results/arxiv_review", help="输出目录")
    
    # 搜索配置
    parser.add_argument('--max-conv-turns', type=int, default=3, help="对话轮数")
    parser.add_argument('--max-search-queries', type=int, default=2, help="每轮最大搜索数")
    parser.add_argument('--arxiv-category', type=str, default="cs.AI", help="ArXiv分类")
    
    args = parser.parse_args()

    survey_topic = f"A Comprehensive Academic Literature Review on {args.topic}"
    print(f"🚀 启动任务: {survey_topic}")
    print(f"🤖 连接 LLM: {LLM_MODEL_NAME}")

    # 初始化模型
    conv_lm = CleanLitellmModel(
        model=LLM_MODEL_NAME, api_key='EMPTY', api_base=LLM_API_URL, model_type='chat',  
        max_tokens=2048, temperature=0.8, top_p=0.9
    )
    article_lm = CleanLitellmModel(
        model=LLM_MODEL_NAME, api_key='EMPTY', api_base=LLM_API_URL, model_type='chat', 
        max_tokens=8192, temperature=0.5, top_p=0.9
    )

    lm_config = STORMWikiLMConfigs()
    lm_config.set_conv_simulator_lm(conv_lm)
    lm_config.set_question_asker_lm(conv_lm)
    lm_config.set_outline_gen_lm(article_lm)
    lm_config.set_article_gen_lm(article_lm)
    lm_config.set_article_polish_lm(article_lm)

    rm = ArXivSearch(k=5, category=args.arxiv_category)

    runner_args = STORMWikiRunnerArguments(
        output_dir=args.output_dir,
        max_conv_turn=args.max_conv_turns,
        max_search_queries_per_turn=args.max_search_queries,
        retrieve_top_k=5, 
        max_thread_num=4, 
        search_top_k=5
    )

    runner = STORMWikiRunner(runner_args, lm_config, rm)

    # 注入学术 Prompt
    print("🎨 注入学术综述 Prompt...")
    outline_gen_module.WritePageOutline.__doc__ = """
    Write a comprehensive academic literature review outline for the given topic.
    The outline should be structured logically (e.g., Introduction, Methodology, Key Themes, Discussion, Conclusion).
    Use "#" for section titles and "##" for subsections. Do not include the topic name itself as a section.
    """
    article_gen_module.WriteSection.__doc__ = """
    Write an academic review section based on the collected information.
    Synthesize the findings from the provided papers. Be critical, formal, and objective.
    Use [1], [2], ..., [n] to cite the sources inline.
    Do not include a References section at the end (it will be handled automatically).
    """

    print("🏁 开始执行 STORM 流程...")
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
        print(f"✅ 任务完成！结果已保存至: {os.path.join(args.output_dir, runner.article_dir_name)}")
        
    except Exception as e:
        print(f"❌ 运行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 屏蔽日志
    logging.getLogger("LiteLLM").setLevel(logging.WARNING) 
    logging.getLogger("litellm").setLevel(logging.WARNING) 
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("arxiv").setLevel(logging.WARNING)
    logging.getLogger("dspy").setLevel(logging.WARNING)
    # 屏蔽 sentence_transformers 的 INFO 日志
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    main()