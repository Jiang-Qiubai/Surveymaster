import os
import json
import re
import argparse
import arxiv  # pip install arxiv
from openai import OpenAI

# ==============================================================================
# ⚙️ 配置部分
# ==============================================================================
DEFAULT_API_BASE = "http://172.17.0.1:8091/v1" 
DEFAULT_API_KEY = "EMPTY"
MODEL_NAME = "qwen3-32b" 

# ==============================================================================
# 🧹 1. 文本清洗与结构处理
# ==============================================================================
def clean_think_block(text):
    """清理 <think> 块及常见的模型废话"""
    if not text: return ""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    garbage_prefixes = [
        r"^Here is the (section|lead|abstract|summary|report).*?:\n",
        r"^Sure, I can help.*?\n",
        r"^Okay, let's.*?\n"
    ]
    for pattern in garbage_prefixes:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    return text

def is_garbage_line(line):
    """判断垃圾行"""
    line = line.strip()
    if not line: return False
    if re.match(r'^(Okay|So|But|And|First|Second), (let|maybe|I need|I should)', line, re.IGNORECASE): return True
    if re.match(r'^Source \d+:', line, re.IGNORECASE): return True
    return False

def escape_latex(text):
    """强制转义 LaTeX 特殊字符 (保留部分用于后续处理的字符暂不转义，如 `)"""
    if not text: return ""
    text = re.sub(r'\b must\b', ' ', text, flags=re.IGNORECASE)
    text = text.replace(' - ', ' -- ') 
    
    replacements = {
        '\\': r'\textbackslash{}',
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}'
        # 注意：不转义 * 和 `，留给后续 Markdown 处理
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text

def remove_summary_section(text):
    """移除 '# summary' 章节"""
    pattern = r'(^|\n)#\s*summary\s*\n(.*?)(?=\n#|\Z)'
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        print(f"✂️ Removed original '# summary' section ({len(match.group(0))} chars).")
        return text.replace(match.group(0), "", 1).strip()
    return text

# ==============================================================================
# 📡 2. 数据获取
# ==============================================================================
def fetch_arxiv_metadata(urls):
    client = arxiv.Client()
    ids = []
    url_to_id = {}
    for url in urls:
        match = re.search(r'arxiv\.org/abs/([\d\.]+)', url)
        if match:
            aid = match.group(1)
            ids.append(aid)
            url_to_id[url] = aid
    
    if not ids: return {}
    print(f"📡 Fetching metadata for {len(ids)} papers...")
    results = {}
    try:
        search = arxiv.Search(id_list=ids)
        for r in client.results(search):
            sid = r.get_short_id().split('v')[0]
            results[sid] = {
                'title': escape_latex(r.title.replace('\n', ' ')), 
                'authors': [escape_latex(a.name) for a in r.authors], 
                'year': r.published.year, 
                'journal': f"arXiv:{sid}"
            }
    except Exception as e:
        print(f"⚠️ Arxiv fetch warning: {e}")
    
    url_to_meta = {}
    for url, aid in url_to_id.items():
        base_id = aid.split('v')[0]
        if base_id in results: url_to_meta[url] = results[base_id]
    return url_to_meta

def rebuild_index(text, url_info):
    matches = re.findall(r'\[([\d,\s]+)\]', text)
    seen_ids = set()
    appearance_order = [] 
    for m in matches:
        parts = [p.strip() for p in m.split(',')]
        for p in parts:
            if p.isdigit():
                pid = int(p)
                if pid not in seen_ids:
                    seen_ids.add(pid)
                    appearance_order.append(pid)
    
    unified_index = url_info.get('url_to_unified_index', {})
    try:
        old_id_to_url = {int(v): k for k, v in unified_index.items()}
    except ValueError:
        old_id_to_url = {}

    if not old_id_to_url and 'url_to_info' in url_info:
        for idx, url in enumerate(url_info['url_to_info'].keys(), start=1):
            old_id_to_url[idx] = url
            
    sorted_urls = []
    id_map = {} 
    current_idx = 1
    for old_id in appearance_order:
        if old_id in old_id_to_url:
            sorted_urls.append(old_id_to_url[old_id])
            id_map[old_id] = current_idx
            current_idx += 1
    return sorted_urls, id_map

# ==============================================================================
# 🧠 3. LLM 生成模块
# ==============================================================================
def generate_title_abstract(text, api_base):
    print("🧠 Generating Title and Abstract using LLM...")
    client = OpenAI(base_url=api_base, api_key=DEFAULT_API_KEY)
    context = text[:25000]
    prompt = f"""
Read the following academic survey content (body text) and generate a high-quality IEEE-style Title and Abstract.

**Input Content:**
{context}...

**Requirements:**
1. **Title**: Concise, professional, starting with keywords like "A Survey on...", "A Comprehensive Review of...".
2. **Abstract**: 200-300 words. Summarize background, methods, challenges, and future directions.
3. **Constraint**: Do NOT include citations in the abstract.
4. **Output Format**: Strictly JSON.

**JSON Schema:**
{{
  "title": "string",
  "abstract": "string"
}}
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.6
        )
        data = json.loads(response.choices[0].message.content)
        return {
            "title": escape_latex(data.get("title", "Generated Survey")),
            "abstract": escape_latex(data.get("abstract", "Abstract generation failed."))
        }
    except Exception as e:
        print(f"⚠️ Metadata generation failed: {e}")
        return {"title": "AI Generated Survey", "abstract": "Abstract generation failed."}

# ==============================================================================
# 📝 4. 格式转换 (Markdown -> LaTeX) [核心修改]
# ==============================================================================
def convert_body_to_latex(text, id_map):
    lines = text.split('\n')
    latex_lines = []
    
    in_itemize = False # 列表状态追踪

    for line in lines:
        line = line.strip()
        if not line:
            if in_itemize:
                latex_lines.append("\\end{itemize}")
                in_itemize = False
            latex_lines.append("")
            continue
        
        if is_garbage_line(line): continue
            
        # 1. 识别并处理结构 (Header / List)
        is_header = False
        is_list_item = False
        level = 0
        content = line
        
        if line.startswith('# '): 
            level = 1; content = line[2:].strip(); is_header = True
        elif line.startswith('## '): 
            level = 2; content = line[3:].strip(); is_header = True
        elif line.startswith('### '): 
            level = 3; content = line[4:].strip(); is_header = True
        elif line.startswith('- ') or line.startswith('* '):
            is_list_item = True
            content = line[2:].strip()

        # 处理列表环境的开闭
        if is_list_item:
            if not in_itemize:
                latex_lines.append("\\begin{itemize}")
                in_itemize = True
        else:
            if in_itemize:
                latex_lines.append("\\end{itemize}")
                in_itemize = False

        # 2. 转义内容 (LaTeX 特殊字符)
        content = escape_latex(content)
        
        # 3. 处理 Markdown 样式
        # 引用 [1] -> \cite{ref1}
        def replacer(match):
            try:
                parts = [int(p) for p in match.group(1).split(',') if p.strip().isdigit()]
                new_ids = [f"ref{id_map[p]}" for p in parts if p in id_map]
                if new_ids: return f"\\cite{{{', '.join(new_ids)}}}"
            except: pass
            return match.group(0)
        content = re.sub(r'\[([\d,\s]+)\]', replacer, content)

        # 加粗 **text** -> \textbf{text}
        content = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', content)
        
        # 行内代码 `text` -> \texttt{text}
        content = re.sub(r'`(.+?)`', r'\\texttt{\1}', content)

        # 4. 【核心修复】清洗残留的 Markdown 符号
        # 移除未匹配成对的 ** (避免 user 提到的 "保留了 **")
        content = content.replace('**', '') 
        # 移除未匹配成对的 `
        content = content.replace('`', '')
        
        # 5. 生成 LaTeX 行
        if is_header:
            if level == 1: latex_lines.append(f"\\section{{{content}}}")
            elif level == 2: latex_lines.append(f"\\subsection{{{content}}}")
            else: latex_lines.append(f"\\textbf{{{content}}}")
        elif is_list_item:
            latex_lines.append(f"\\item {content}")
        else:
            latex_lines.append(content)
    
    if in_itemize:
        latex_lines.append("\\end{itemize}")
            
    return "\n".join(latex_lines)

# ==============================================================================
# 🚀 主程序
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help="STORM output dir")
    parser.add_argument('--llm-url', type=str, default=DEFAULT_API_BASE)
    args = parser.parse_args()
    
    input_path = os.path.join(args.dir, "storm_gen_article_polished.txt")
    if not os.path.exists(input_path):
        input_path = os.path.join(args.dir, "storm_gen_article.txt")
    if not os.path.exists(input_path):
        print(f"❌ Error: No article file found in {args.dir}")
        return

    json_path = os.path.join(args.dir, "url_to_info.json")
    if not os.path.exists(json_path):
        print(f"❌ Error: url_to_info.json not found")
        return
    
    with open(input_path, 'r', encoding='utf-8') as f: raw_text = f.read()
    with open(json_path, 'r', encoding='utf-8') as f: url_info = json.load(f)

    print(f"📄 Processing: {input_path}")

    # 流程
    pre_cleaned_text = clean_think_block(raw_text)
    body_text = remove_summary_section(pre_cleaned_text)
    sorted_urls, id_map = rebuild_index(body_text, url_info)
    url_to_meta = fetch_arxiv_metadata(sorted_urls)
    meta_data = generate_title_abstract(body_text, args.llm_url)
    latex_body = convert_body_to_latex(body_text, id_map)

    # 组装 LaTeX
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

\\title{{{meta_data['title']}}}
\\author{{\\IEEEauthorblockN{{Generated by STORM Pipeline}}}}

\\maketitle

\\begin{{abstract}}
{meta_data['abstract']}
\\end{{abstract}}

\\begin{{IEEEkeywords}}
Large Language Models, Survey, Artificial Intelligence
\\end{{IEEEkeywords}}

{latex_body}

\\bibliographystyle{{IEEEtran}}
\\bibliography{{references}}

\\end{{document}}
"""

    # 生成 BibTeX
    bib_entries = []
    for i, url in enumerate(sorted_urls, 1):
        key = f"ref{i}"
        if url in url_to_meta:
            m = url_to_meta[url]
            entry = f"@article{{{key}, title={{{m['title']}}}, author={{{' and '.join(m['authors'])}}}, journal={{{m['journal']}}}, year={{{m['year']}}}, url={{{url}}}}}"
        else:
            info = url_info.get('url_to_info', {}).get(url, {})
            title = escape_latex(info.get('title', 'Unknown Source'))
            author = escape_latex(info.get('author', 'Unknown Author'))
            entry = f"@misc{{{key}, title={{{title}}}, author={{{author}}}, howpublished={{\\url{{{url}}}}}}}"
        bib_entries.append(entry)

    # 保存
    tex_path = os.path.join(args.dir, "main.tex")
    bib_path = os.path.join(args.dir, "references.bib")
    with open(tex_path, 'w', encoding='utf-8') as f: f.write(final_latex)
    with open(bib_path, 'w', encoding='utf-8') as f: f.write("\n".join(bib_entries))
    
    print("-" * 40)
    print("✅ Post-processing complete!")
    print(f"   Title: {meta_data['title']}")
    print(f"   Output: {tex_path}")
    print("-" * 40)

if __name__ == "__main__":
    main()