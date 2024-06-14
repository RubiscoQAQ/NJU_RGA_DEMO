# 实现过程

[构建检索增强生成 （RAG） 应用程序 |🦜️🔗 LangChain的](https://python.langchain.com/v0.2/docs/tutorials/rag/)

[LCEL 起步 | 🦜️🔗 Langchain](https://python.langchain.com.cn/docs/expression_language/get_started)

## 1. 环境安装和准备

### 1.1 安装必要的环境

下载python3.9

- pip3 install openai langchain : langchain的基本环境
- pip3 install langchain-community
- pip3 install langchain-openai
- pip3 install pymilvus==2.2.6 ：向量数据库
  - 版本不能太新，例如：python 3.12就会报错

```python
from langchain_openai import OpenAI, ChatOpenAI
import os
os.environ["OPENAI_API_KEY"] = "None"
os.environ["OPENAI_API_BASE"] = "http://10.58.0.2:8000/v1"
llm_completion = OpenAI(model_name="Qwen1.5-14B")
llm_chat = ChatOpenAI(model_name="Qwen1.5-14B")

print(llm_completion.invoke("hello"))
```

示例demo

## 2. 实现Loader

参考阅读：[How to load JSON | 🦜️🔗 LangChain](https://python.langchain.com/v0.2/docs/how_to/document_loader_json/)

### 2.1 引入依赖(windows参考2.2 自行实现)

- pip install jq

  - windows下100%报错，我的建议是：winget install jqlang.jq
    - 可以是可以，但是无法import。
  - windows参考：[Building wheels for collected package: jq failed in Windows · Issue #4396 · langchain-ai/langchain (github.com)](https://github.com/langchain-ai/langchain/issues/4396)自行实现一个JsonLoader

### 2.2 自行实现一个简易的Loader（其实没用）

```python
import json
from pathlib import Path
from typing import Optional, Union, List, Callable

from langchain.docstore.document import Document
from langchain_community.document_loaders.base import BaseLoader


class MyJSONLoader(BaseLoader):
    """
    Windows环境下，无法通过pip安装jq。
    作为一种JSONLoader的替代方案
    """

    def __init__(
            self,
            file_path: Union[str, Path],
            content_key: Optional[str] = None,
            metadata_func: Optional[Callable[[dict,int, dict], dict]] = None,
            json_lines: bool = False
    ):
        self.file_path = Path(file_path).resolve()
        self._content_key = content_key
        self._metadata_func = metadata_func
        self._json_lines = json_lines

    def create_documents(self, processed_data):
        documents = []
        for item in processed_data:
            content = item.get('content', '')
            metadata = item.get('metadata', {})
            document = Document(page_content=content, metadata=metadata)
            documents.append(document)
        return documents

    def process_json(self, data):
        if isinstance(data, list):
            processed_data = []
            for item_cnt, item in enumerate(data,start=1):
                content = item.get(self._content_key, '') if self._content_key else ''
                metadata = {}
                if self._metadata_func and isinstance(item, dict):
                    metadata = self._metadata_func(item,item_cnt, {})
                processed_data.append({'content': content, 'metadata': metadata})
            return processed_data
        else:
            return []

    def load(self) -> List[Document]:
        # 加载，并返回Json格式解析出的Document
        docs = []
        with open(self.file_path, mode='r', encoding='utf-8') as json_file:
            if self._json_lines:
                # jsonl格式
                for line_ptr, line in enumerate(json_file, start=1):
                    try:
                        data = json.loads(line)
                        processed_json = self.process_json(data)
                        docs.extend(self.create_documents(processed_json))
                    except json.JSONDecodeError:
                        print(f"Error: Invalid JSON format at line {line_ptr}.")
            else:
                try:
                    data = json.load(json_file)
                    processed_json = self.process_json(data)
                    docs = self.create_documents(processed_json)
                except json.JSONDecodeError:
                    print("Error: Invalid JSON format in the file")
        return docs

```

## 3. 文本拆分（也没啥用）

为了解决超大文本的读取问题，往往提供文档拆分的功能。

参考：[操作指南 |🦜️🔗 LangChain的](https://python.langchain.com/v0.2/docs/how_to/#text-splitters)

我们可以使用最简单的RecursiveCharacterTextSplitter，来将比较大的document进行拆分

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,chunk_overlap=200,add_start_index=True
)
```

其中，设置超过1000个字符的块进行拆分，块与块之间有200字符的重叠

## 4. 向量存储（没啥用）

将Document存储为向量文本，主要包括两个主要工作：

1. 通过文本嵌入模型，将文本转换为嵌入
   [文本嵌入模型 |🦜️🔗 LangChain的](https://python.langchain.com/v0.2/docs/how_to/embed_text/)
2. 通过向量数据库，将嵌入存储和查询。
   [如何创建和查询向量存储 |🦜️🔗 LangChain的](https://python.langchain.com/v0.2/docs/how_to/vectorstores/)

### Milvus向量数据库

[Milvus向量数据库安装、使用全中文文档教程 – Milvus向量库中文文档 (milvus-io.com)](https://www.milvus-io.com/overview)

## 5. 根据论文生成回答（重要的是prompt的编写）

[How to get a RAG application to add citations | 🦜️🔗 LangChain](https://python.langchain.com/v0.2/docs/how_to/qa_citations/)

```
'''
    你现在担任一名从事科普的专业人员。负责按照指定流程，依据相关论文，专业的解答提出的问题。
    你的工作必须遵守以下规则：
    
    1. 解答问题时应该遵守流程如下：
    
    (1): 遍历给定的Document列表
    (2): 根据这些Document的page_content，生成目标问题的回答。如果这些page_content都和问题无关，则回答：我不知道。
    (3): 你只能根据这些论文的page_content生成回答，不能根据已有知识回答或者编造回答。
    (4): 你的回答一定要遵守以下格式：回答写在[回答]部分，引用的Document写在[参考文献]部分
        [回答]: [你生成的回答]
        [参考文献]: [你选择的参考文献]

    你的回答应该包含引用的论文，并按照Document格式写在[参考文献]部分
    2. 选择参考文献的流程如下：
    (1): 遍历给定的Document列表
    (2): 选择与回答相关的Document，作为参考文献。你不需要保证参考文献Document和回答完全对应。

    3. 最后，你必须以如下格式输出。其中，[回答]部分填写根据这些Document的page_content生成的回答，引用的Document写在[参考文献]部分
        [回答]: [你生成的回答]
        [参考文献]: [你选择的参考文献]
    
    4. 提供给你Document格式如下：
    Document(
        page_content： 论文abstract,
        metadata(
            access_id：论文的唯一id,
            authors：论文的作者,
            title：论文的题目,
            comments：论文的评论，一般为作者的补充信息,
            journal_ref：论文的发布信息,
            doi：电子发行doi,
            categories：论文的分类
        )
    )
    5. 其他额外的约束：
    - 对于每一次用户提问，你应该且只应该输出一次回答。
    - 你的回答应该保持通俗易懂并且严谨的科普文风格，且必须使用中文回答。
    - 如果你对问题给出了回答，则必须选择相关Document作为参考文献。只有回答：“我不知道”时，才能让参考文献为空。
    - 注意，必须保证输出的格式正确，即满足：
        [回答]: [你生成的回答]
        [参考文献]: [你选择的参考文献]
    6. 按照以上要求，示例如下：
    - 用户提问：软件工程领域如何适应不同领域？
    - 用户给出参考Document列表：[Document(1),Document(2),Document(3),Document(4)]
    - 你遍历这几篇论文，分别判断它们是否和问题相关
    - 你发现只有Document(1)和Document(4)涉及了软件工程领域，因此，你根据这两篇论文的page_content，生成答案。并将这两篇论文以目标格式填入参考文献中。
        [回答]: [你生成的回答]
        [参考文献]: [你选择的参考文献]

    - 如果没有论文相关，则直接回答不知道。
        [回答]: 我不知道
      
    '''
```



## 6. 最后的格式解析

LangChain提供了Json格式的解析，但是必须要求模型生成正确的Json格式。容易出现异常。

因此，在调用链中新增一个大模型处理，用于将输出格式化为json格式

```
'''
    现在要你帮忙将一段用户输入的文本转换为Json格式用于保存。
    规则：
    (1). 用户输入的文本会包含 [回答] 和 [参考文献] 两个标签。分别对应json中的content和cite两个key。你需要将[回答]标签后面的内容作为content的value，[参考文献]标签后面的内容作为cite的value
    (2). content对应的值是一个字符串，存储了用户输入的回答内容部分
    (3). cite对应的值是一个字符串列表，存储了用户输入参考文献内容部分。你需要正确的将用户输入的参考文献文本拆分成列表
    (4). 输出格式规定为Json格式，例如：{{"content":"content_string","cite": ["cite1_string", "cite2_string"]}}
    现在有两个角色：
    - Json转换器，精通自然语言处理。用于将用户输入按照规则转换为Json格式。
    - Json校验器，擅长检测Json语法。用于检查Json格式是否合法。
    你需要按照以下步骤完成工作，且每一步都要遵守以上规则：
    Step 1: 扮演Json转换器，按照规则，将用户输入转换为目标Json格式，例如：{{"content":"content_string","cite": ["cite1_string", "cite2_string"]}}，然后执行Step 2
    Step 2: 扮演Json校验器，按照Json语法，校验上一步的输出是否满足目标格式：{{"content":"content_string","cite": ["cite1_string", "cite2_string"]}}。如果不满足，则返回Step 1，如果满足，则进入Step 3
    Step 3: 输出经过Step 2校验过的Json格式文件
    说明：content_string、cite1_string、cite2_string、回答内容、参考文献内容 为占位符，你不能直接输出
    约束：你最终只能输出一个Json格式的字符串。不要有其他任何无关输出
    '''
```

## 7. 使用大模型对输入进行润色

发现用英文输入有利于文献的查找。因此，通过prompt训练大模型对输入进行润色

```
system_prompt = (
    '''
    现在要你帮忙将一段用户输入的问题转换为更加利于向量数据库查询的形式。
    规则：
        - 翻译时要准确传达学术论文的事实和背景，同时风格上保持为通俗易懂并且严谨的科普文风格。
        - 保留特定的英文术语、数字或名字，并在其前后加上空格，例如："中 UN 文"，"不超过 10 秒"。
        - 即使上意译也要保留术语，例如 FLAC，JPEG 等。保留公司缩写，例如 Microsoft, Amazon 等。
        - 全角括号换成半角括号，并在左括号前面加半角空格，右括号后面加半角空格。
    现在有三个角色：
    - 英语老师，精通英文，能精确的理解英文并用中文表达
    - 中文老师，精通中文，能精确的理解中文并用英文表达
    - 校长，精通中文和英文，擅长校对审查

    和步骤来翻译这篇文章，每一步都必须遵守以上规则，打印每一步的输出结果：
    Step 1：现在你是中文老师，精通中文，对原文按照字面意思直译，务必遵守原意，翻译时保持原始中文的段落结构，不要合并分段
    Step 2：扮演英文老师，精通英文，擅长写通俗易懂的科普文章，对中文老师翻译的内容重新意译，遵守原意的前提下让内容更通俗易懂，符合英文表达习惯，但不要增加和删减内容，保持原始分段
    Step 3：扮演校长，精通中文和英文，校对回译稿和原稿中的区别，重点检查两点：翻译稿和原文有出入的位置；不符合英文表达习惯的位置；
    Step 5：英文老师基于校长的修改意见，修改初稿，作为最终稿
    Step 6：输出英文老师的最终稿
    
    你只需要输出英文老师提供的最终稿。
    '''
)
```

# 反思

1. 对于迭代式解答问题暂时没有太多思路。
2. 生成答案时，即使prompt已经明确给出格式要求，模型仍然存在不按要求书写参考文档的问题。考虑如下：
   1. 升级能力更强的大模型
   2. 第一步先生成答案，再训练一个prompt，从答案中选择对应的参考论文

# 参考阅读：

1. [LangChain 中文入门教程 - General - LangChain中文社区](https://www.langchain.cn/t/topic/35)
2. [LangChain 中文入门教程-实例 - General - LangChain中文社区](https://www.langchain.cn/t/topic/626)
3. [How to load JSON | 🦜️🔗 LangChain](https://python.langchain.com/v0.2/docs/how_to/document_loader_json/)

3. [Building wheels for collected package: jq failed in Windows · Issue #4396 · langchain-ai/langchain (github.com)](https://github.com/langchain-ai/langchain/issues/4396)
4. [构建检索增强生成 （RAG） 应用程序 |🦜️🔗 LangChain的](https://python.langchain.com/v0.2/docs/tutorials/rag/)
5. [操作指南 |🦜️🔗 LangChain的](https://python.langchain.com/v0.2/docs/how_to/#text-splitters)
6. [如何创建和查询向量存储 |🦜️🔗 LangChain的](https://python.langchain.com/v0.2/docs/how_to/vectorstores/)
7. [文本嵌入模型 |🦜️🔗 LangChain的](https://python.langchain.com/v0.2/docs/how_to/embed_text/)
8. [Milvus向量数据库安装、使用全中文文档教程 – Milvus向量库中文文档 (milvus-io.com)](https://www.milvus-io.com/overview)
9. [How to get a RAG application to add citations | 🦜️🔗 LangChain](https://python.langchain.com/v0.2/docs/how_to/qa_citations/)