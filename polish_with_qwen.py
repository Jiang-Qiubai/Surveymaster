import os
import re
import json
import argparse
import time
from openai import OpenAI

# ==============================================================================
# ⚙️ 配置部分
# ==============================================================================
# 默认 API 地址 (根据你的环境修改)
DEFAULT_API_BASE = "http://172.17.0.1:8091/v1"
DEFAULT_API_KEY = "EMPTY"
MODEL_NAME = "qwen3-32b" # 或你实际部署的模型名称

# ==============================================================================
# 🧠 引用语义化管理器 (Semantic Citation Manager)
# ==============================================================================
class CitationManager:
    def __init__(self, bib_path):
        self.bib_path = bib_path
        self.ref_to_semantic = {}   # ref1 -> yang2024code
        self.semantic_to_entry = {} # yang2024code -> @article{...} string
        self._parse_and_build_map()

    def _generate_semantic_key(self, meta):
        """生成 authorYearKeyword 格式的键名"""
        # 提取作者姓氏
        author = "unknown"
        if 'author' in meta:
            first_author = meta['author'].split(' and ')[0] # 取第一作者
            # 移除 LaTeX 转义符和非字母字符
            author = re.sub(r'[^a-zA-Z]', '', first_author.lower())
        
        year = meta.get('year', 'nd')
        
        # 提取标题关键词
        keyword = "ref"
        if 'title' in meta:
            # 简单的停用词过滤
            stopwords = {'the', 'a', 'an', 'on', 'in', 'of', 'for', 'to', 'and', 'with', 'survey', 'review'}
            words = re.findall(r'[a-zA-Z]{3,}', meta['title'].lower())
            for w in words:
                if w not in stopwords:
                    keyword = w
                    break
        
        base_key = f"{author}{year}{keyword}"
        return base_key

    def _parse_and_build_map(self):
        if not os.path.exists(self.bib_path):
            print("⚠️ Warning: draft_refs.bib not found.")
            return

        with open(self.bib_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 使用正则提取每个 bib 条目
        # 假设格式标准: @type{key, ...}
        entries = re.findall(r'@\w+\{(ref\d+),(.*?)\n\}', content, re.DOTALL)
        
        print(f"📖 Loaded {len(entries)} citations from draft bib.")

        key_counts = {} # 用于处理键名冲突

        for ref_id, body in entries:
            # 提取元数据用于生成键名
            meta = {}
            for field in ['title', 'author', 'year', 'journal', 'url']:
                match = re.search(f"{field}\s*=\s*[\"{{](.*?)[\"}}]", body, re.IGNORECASE)
                if match: meta[field] = match.group(1)
            
            # 生成语义键名
            sem_key = self._generate_semantic_key(meta)
            
            # 冲突处理 (zhang2024llm -> zhang2024llm2)
            if sem_key in key_counts:
                key_counts[sem_key] += 1
                sem_key = f"{sem_key}{key_counts[sem_key]}"
            else:
                key_counts[sem_key] = 1

            # 存储映射
            self.ref_to_semantic[ref_id] = sem_key
            
            # 重建 Bib 条目 (替换原本的 refX 为 semantic key)
            new_entry = f"@article{{{sem_key},\n{body}\n}}" # 简单重建，保留 body
            self.semantic_to_entry[sem_key] = new_entry

    def replace_in_text(self, text):
        """将 text 中的 \cite{ref1} 替换为 \cite{yang2024code}"""
        def replacer(match):
            refs = match.group(1).split(',')
            new_refs = []
            for r in refs:
                r = r.strip()
                if r in self.ref_to_semantic:
                    new_refs.append(self.ref_to_semantic[r])
                else:
                    new_refs.append(r) # 没找到就保持原样
            return f"\\cite{{{','.join(new_refs)}}}"

        return re.sub(r'\\cite\{([^\}]+)\}', replacer, text)

    def generate_final_bib(self, final_text):
        """根据最终文本中实际用到的引用，生成 clean bib"""
        used_keys = set()
        matches = re.findall(r'\\cite\{([^\}]+)\}', final_text)
        for m in matches:
            for k in m.split(','):
                used_keys.add(k.strip())
        
        final_entries = []
        for k in used_keys:
            if k in self.semantic_to_entry:
                final_entries.append(self.semantic_to_entry[k])
        
        return "\n\n".join(final_entries)

# ==============================================================================
# 🤖 LLM 调用函数
# ==============================================================================
def call_llm(client, prompt, system_prompt="You are a helpful assistant.", max_tokens=4096):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ LLM Call Failed: {e}")
        return None

# ==============================================================================
# 🚀 核心流程
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help="Result directory")
    parser.add_argument('--api-base', type=str, default=DEFAULT_API_BASE)
    args = parser.parse_args()

    # 初始化 OpenAI Client
    client = OpenAI(base_url=args.api_base, api_key=DEFAULT_API_KEY)
    
    # 文件路径
    draft_tex = os.path.join(args.dir, "draft_paper.tex")
    draft_bib = os.path.join(args.dir, "draft_refs.bib")
    
    if not os.path.exists(draft_tex):
        print("❌ draft_paper.tex not found. Run storm_to_ieee.py first.")
        return

    # 1. 读取 Draft 正文
    # 我们需要提取 \begin{document} 之后，\bibliographystyle 之前的内容
    with open(draft_tex, 'r', encoding='utf-8') as f:
        full_latex = f.read()

    # 简单的提取 Body 逻辑
    body_match = re.search(r'\\end\{abstract\}(.*?)\\bibliographystyle', full_latex, re.DOTALL)
    if not body_match:
        # 如果没有 abstract 块 (draft可能没有)，尝试从 maketitle 后提取
        body_match = re.search(r'\\maketitle(.*?)\\bibliographystyle', full_latex, re.DOTALL)
    
    if not body_match:
        print("❌ Could not extract body from draft_paper.tex")
        return
    
    raw_body = body_match.group(1).strip()

    # 2. 语义化引用 (Pre-processing)
    print("🔄 [Step 1/4] Semanticizing citations...")
    cit_manager = CitationManager(draft_bib)
    semantic_body = cit_manager.replace_in_text(raw_body)

    # 3. LLM 润色 (Polishing)
    print("✍️ [Step 2/4] Polishing body text with Qwen...")
    polish_prompt = f"""
You are an expert academic editor for IEEE Transactions.
Please refine the following LaTeX content.

**Requirements:**
1. Improve the flow, clarity, and academic tone.
2. Connect paragraphs logically.
3. **CRITICAL**: Do NOT remove or modify citation keys (e.g., \\cite{{yang2024code}}). Keep them exactly where they are.
4. **CRITICAL**: Maintain the LaTeX structure (\\section, \\textbf, etc.).
5. Output ONLY the refined LaTeX body code. No markdown code blocks, no intro text.

**Content to Polish:**
{semantic_body[:25000]} 
""" # 截断以防超长
    
    polished_body = call_llm(client, polish_prompt, system_prompt="You are a strict LaTeX editor.")
    if not polished_body:
        print("⚠️ Polishing failed, using semantic draft.")
        polished_body = semantic_body
    
    # 清理一下 LLM 可能输出的 ```latex ... ```
    polished_body = re.sub(r'^```latex', '', polished_body).replace('```', '').strip()

    # 4. LLM 生成标题和摘要 (Metadata Gen)
    print("🧠 [Step 3/4] Generating Title & Abstract from polished text...")
    meta_prompt = f"""
Based on the following academic paper content, generate a high-quality Title and Abstract.

**Content:**
{polished_body[:10000]}... (truncated)

**Output Format:**
Return a JSON object strictly:
{{
  "title": "Your Generated Title Here",
  "abstract": "Your generated abstract here (approx 150-250 words)."
}}
"""
    meta_response = call_llm(client, meta_prompt, system_prompt="You are a JSON generator.")
    
    try:
        # 尝试提取 JSON
        json_match = re.search(r'\{.*\}', meta_response, re.DOTALL)
        if json_match:
            meta_data = json.loads(json_match.group(0))
        else:
            raise ValueError("No JSON found")
            
        title = meta_data.get('title', 'AI Generated Survey')
        abstract = meta_data.get('abstract', 'Summary generation failed.')
    except Exception as e:
        print(f"⚠️ Metadata generation failed: {e}. Using placeholders.")
        title = "AI Generated Survey (Polished)"
        abstract = "Abstract generation failed. Please review the body text."

    # 5. 组装最终 LaTeX (Assembly)
    print("📝 [Step 4/4] Assembling final_paper.tex...")
    
    final_latex = f"""\\documentclass[conference]{{IEEEtran}}
\\usepackage{{cite}}
\\usepackage{{amsmath,amssymb,amsfonts}}
\\usepackage{{algorithmic}}
\\usepackage{{graphicx}}
\\usepackage{{textcomp}}
\\usepackage{{xcolor}}
\\usepackage{{url}}
\\def\\BibTeX{{{{\\rm B\\kern-.05em{{\\sc i\\kern-.025em b}}\\kern-.08em
    T\\kern-.1667em\\lower.7ex\\hbox{{E}}\\kern-.125emX}}}}

\\begin{{document}}

\\title{{{title}}}
\\author{{\\IEEEauthorblockN{{Generated by STORM & Qwen}}}}

\\maketitle

\\begin{{abstract}}
{abstract}
\\end{{abstract}}

\\begin{{IEEEkeywords}}
Large Language Models, Code Intelligence, Survey, Artificial Intelligence
\\end{{IEEEkeywords}}

{polished_body}

\\bibliographystyle{{IEEEtran}}
\\bibliography{{final_refs}}

\\end{{document}}
"""

    # 生成 Clean Bib
    final_bib_content = cit_manager.generate_final_bib(polished_body)

    # 写入文件
    out_tex = os.path.join(args.dir, "final_paper.tex")
    out_bib = os.path.join(args.dir, "final_refs.bib")
    
    with open(out_tex, 'w', encoding='utf-8') as f: f.write(final_latex)
    with open(out_bib, 'w', encoding='utf-8') as f: f.write(final_bib_content)

    print("-" * 40)
    print("🎉 Polishing Complete!")
    print(f"   Final TeX: {out_tex}")
    print(f"   Final Bib: {out_bib}")

if __name__ == "__main__":
    main()