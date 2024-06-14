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
    ):
        self.file_path = Path(file_path).resolve()
        self._content_key = content_key
        self._metadata_func = metadata_func

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
                try:
                    data = json.load(json_file)
                    processed_json = self.process_json(data)
                    docs = self.create_documents(processed_json)
                except json.JSONDecodeError:
                    print("Error: Invalid JSON format in the file")
        return docs
