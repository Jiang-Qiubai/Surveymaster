# 项目概述

这是一个基于STORM（Synthesis of Topic-Oriented Research Materials）框架的学术综述生成器，能够根据指定主题自动检索、分析并生成高质量的文献综述报告。

配置文件详解

文件位置：run_storm_arxiv.py

硬编码配置区域

# LLM服务配置
LLM_API_URL = "http://172.17.0.1:8091/v1"  # 本地LLM API地址
LLM_MODEL_NAME = "openai/qwen3-32b"         # 使用的大语言模型

# 检索参数配置
SEARCH_TOP_K = 10       # 每一轮搜索更多的论文（原为5）
RETRIEVE_TOP_K = 10     # 每一段写作参考更多的片段（原为5）
MAX_CONV_TURNS = 5      # 增加对话轮数以覆盖更多子话题（原为3）
MAX_TOKEN_WRITE = 16384 # 写作时允许的最大输出长度
<img width="1514" height="861" alt="image" src="https://github.com/user-attachments/assets/636af80f-f894-4a04-af2f-549ca4fde66d" />

# 本地嵌入模型配置
LOCAL_EMBEDDING_PATH = "~/models/paraphrase-MiniLM-L6-v2"


启动命令

bash start.sh "Recent advances in LLM-based code completion" cs.AI "code completion"


参数说明：

1. 文章主题："Recent advances in LLM-based code completion"
   • 要生成综述的学术主题

2. arXiv论文分区：cs.AI
   • 计算机科学-人工智能分类

   • 其他常用分类：cs.CL（计算语言学）、cs.LG（机器学习）、cs.SE（软件工程）等

3. 关键字："code completion"
   • 用于文献检索的关键词

功能特点

1. 智能检索：自动从arXiv获取相关论文
2. 深度分析：通过多轮对话深入理解文献
3. 结构化生成：自动组织生成逻辑清晰的综述报告
4. 本地化支持：支持本地部署的嵌入模型和LLM

使用前提

1. 确保本地已启动LLM API服务（端口8091）
2. 已下载并配置好本地嵌入模型
3. 安装所需的Python依赖包

输出结果

程序运行后将生成包含以下内容的综述报告：
• 研究背景与意义

• 关键方法和技术综述

• 当前研究挑战

• 未来发展方向

• 参考文献列表

注：配置中的参数已针对生成长篇综述进行优化，如需调整检索深度或输出长度，可相应修改SEARCH_TOP_K、RETRIEVE_TOP_K和MAX_TOKEN_WRITE参数。
