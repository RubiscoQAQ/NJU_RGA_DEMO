import json
import os
from pathlib import Path
from pprint import pprint

from jsonlines import jsonlines
from langchain_community.vectorstores import Milvus
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import PyConfig
from utils import Utils
from documentLoader.InputLoader import MyJSONLoader
from prompts.QueryPrompt import queryPrompt
from prompts.FormatPrompt import formatPrompt, parser
#
# # 1. 从Json中读取数据
loader = MyJSONLoader(
    file_path=PyConfig.JSON_INPUT_PATH,
    content_key="question",
    metadata_func=Utils.item_metadata_func,
)
datas = loader.load()
pprint('-----------1. 从Json中读取数据-------------')
pprint(datas)
#
# # 2. 文本切分
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000, chunk_overlap=200, add_start_index=True
# )
# all_splits = text_splitter.split_documents(data)
# pprint('-----------2. 文本切分-------------')
# pprint(all_splits)
# pprint('-----------3. 查向量数据库，构建知识库-------------')
# # 3. 嵌入模型
# embedding = HuggingFaceEmbeddings(model_name=PyConfig.EMBEDDING_MODEL_NAME)
# # 4. 向量数据库
# db = Milvus(embedding_function=embedding, collection_name="arXiv",
#             connection_args={"host": PyConfig.MILVUS_HOST, "port": PyConfig.MILVUS_PORT})
# # res = db.search("重复的数据会对In-content Learning产生什么影响？", search_type="similarity")
#
#
# # pprint(res)
# # 5. 生成prompt
# pprint('-----------4. 生成prompt-------------')
# os.environ["OPENAI_API_KEY"] = PyConfig.OPENAI_KEY
# os.environ["OPENAI_API_BASE"] = PyConfig.OPENAI_BASE
# llm_chat = ChatOpenAI(model_name=PyConfig.MODEL_NAME)
# # pmt = queryPrompt.format_messages(context=res, input="重复的数据会对In-content Learning产生什么影响？")
# # pprint(pmt)
# # res = llm_chat.invoke(pmt)
# # pprint(res)
#
# # 6.构造执行链
# pprint('-----------5. 构造执行链-------------')
#
# chain_from_json = (
#         RunnablePassthrough.assign(
#             context=(lambda x: Utils.format_docs(x))) | queryPrompt | llm_chat | StrOutputParser()
# )
#
# retrieve_docs = (lambda x: x['input']) | db.as_retriever()
#
# chain = RunnablePassthrough.assign(context=retrieve_docs).assign(
#     answer=chain_from_json
# )
# # 7. 调用执行链
# pprint('-----------6. 执行执行链-------------')
# res = chain.invoke({"input": "代码评审的目标是什么？"})
# pprint(res)
# pprint('-----------7. 输出格式化-------------')
# format_chat = ChatOpenAI(model_name=PyConfig.MODEL_NAME)
# chain_to_json = (
#         {"unformat_out": chain} | formatPrompt | format_chat | parser
# )
# pprint('-----------8. QA运行-------------')
# while True:
#     res = chain_to_json.invoke({"input": input()})
#     pprint(res)

pprint("----------jsonl模式-----------")

for line_ptr,data in enumerate(datas,start=1):
    res = {}
    try:
            question = data.page_content
            pprint("question:" + question)
            res['question']=question
            res['answer'] = 'self.ask(question)'
            with jsonlines.open(PyConfig.JSON_OUTPUT_PATH, mode="a") as file_jsonl:
                file_jsonl.write(res)
    except:
        print(f"Error: Invalid JSON format at line {line_ptr}.")