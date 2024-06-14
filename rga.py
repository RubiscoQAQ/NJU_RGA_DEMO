import json
from pathlib import Path

import jsonlines
import os
from pprint import pprint
from langchain.globals import set_debug, set_verbose
from langchain_community.vectorstores import Milvus
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from config import PyConfig
from utils import Utils
from documentLoader.InputLoader import MyJSONLoader
from prompts.QueryPrompt import queryPrompt
from prompts.FormatPrompt import formatPrompt, parser
from prompts.QuestionPromopt import questionPrompt


class MyRGA:
    """
    基于LangChain的RGA问答系统
    """

    def init_milvus(self):
        pprint("---------初始化Milvus向量数据库--------")
        embedding = HuggingFaceEmbeddings(model_name=PyConfig.EMBEDDING_MODEL_NAME)
        self.db = Milvus(embedding_function=embedding, collection_name="arXiv",
                         connection_args={"host": PyConfig.MILVUS_HOST, "port": PyConfig.MILVUS_PORT})

    def init_chat_llm(self):
        pprint("---------初始化对话用大模型--------")
        self.llm_chat = ChatOpenAI(model_name=PyConfig.MODEL_NAME)

    def init_format_llm(self):
        pprint("---------初始化格式化用大模型--------")
        self.llm_format = ChatOpenAI(model_name=PyConfig.MODEL_NAME)

    def init_pre_process_llm(self):
        pprint("---------初始化输入润色用大模型--------")
        self.llm_pre_process = ChatOpenAI(model_name=PyConfig.MODEL_NAME)

    def get_chain(self):
        # 用于向对话用大模型获取回答
        chain_from_json = (
                RunnablePassthrough.assign(
                    context=(lambda x: Utils.format_docs(x))) | queryPrompt | self.llm_chat | StrOutputParser()
        )
        # 用于向输入润色用大模型获取向量数据库的搜索结果
        retrieve_docs = (
                {
                    "input": RunnablePassthrough()} | questionPrompt | self.llm_pre_process | StrOutputParser() | self.db.as_retriever()
        )
        # 用于向格式化用大模型获取格式化后的答案
        chain_to_json = (
                {"unformat_out": RunnablePassthrough.assign(context=retrieve_docs).assign(answer=chain_from_json)}
                | formatPrompt | self.llm_format | parser
        )
        return chain_to_json

    def init_json_document(self, url):
        loader = MyJSONLoader(
            file_path=url,
            content_key="question",
            metadata_func=Utils.item_metadata_func,
        )
        return loader.load()

    def ask(self, question):
        pprint("----------用户提问-----------")
        pprint(question)
        answer = self.get_chain().invoke({'input': question})
        pprint(answer)
        return answer['cite'], answer['content']

    def user_ask(self):
        self.ask(input())

    def jsonl(self, url=PyConfig.JSON_INPUT_PATH):
        pprint("----------jsonl模式-----------")
        for line_ptr, data in enumerate(self.init_json_document(url), start=1):
            res = {}
            try:
                question = data.page_content
                pprint("question:" + question)
                res['question'] = question
                res['answer'] = {}
                answer = self.ask(question)
                res['answer']['cite'] = answer[0]
                res['answer']['content'] = answer[1]
                pprint('answer:' + str(res['answer']))
                with jsonlines.open(PyConfig.JSON_OUTPUT_PATH, mode="w") as file_jsonl:
                    file_jsonl.write(res)
            except:
                print(f"Error: Invalid JSON format at line {line_ptr}.")

    def __init__(self, debug=None):
        os.environ["OPENAI_API_KEY"] = PyConfig.OPENAI_KEY
        os.environ["OPENAI_API_BASE"] = PyConfig.OPENAI_BASE
        self.llm_pre_process = None
        self.db = None
        self.llm_chat = None
        self.llm_format = None
        self.debug = debug
        self.out_url = PyConfig.JSON_OUTPUT_PATH
        self.init_milvus()
        self.init_chat_llm()
        self.init_format_llm()
        self.init_pre_process_llm()
        self._set_debug()

    def _set_debug(self):
        if self.debug is None:
            pass
        elif self.debug == 'debug':
            set_debug(True)
        elif self.debug == 'verbose':
            set_verbose(True)
