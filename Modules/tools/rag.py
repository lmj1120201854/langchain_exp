# 1.Load 导入Document Loaders
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from typing import Dict, List, Any
from langchain.embeddings.base import Embeddings
# from langchain.pydantic_v1 import BaseModel
from pydantic import BaseModel
from volcenginesdkarkruntime import Ark

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant

import logging  # 导入Logging工具
from langchain_openai import ChatOpenAI  # ChatOpenAI模型
from langchain.retrievers.multi_query import (
    MultiQueryRetriever,
)  # MultiQueryRetriever工具
from langchain.chains import RetrievalQA  # RetrievalQA链

class DoubaoEmbeddings(BaseModel, Embeddings):
    client: Ark = None
    api_key: str = ""
    model: str

    def __init__(self, **data: Any):
        super().__init__(**data)
        if self.api_key == "":
            self.api_key = os.environ["EMBEDDING_API_KEY"]
        self.client = Ark(
            base_url=os.environ["EMBEDDING_BASE_URL"],
            api_key=self.api_key
        )

    def embed_query(self, text: str) -> List[float]:
        """
        生成输入文本的 embedding.
        Args:
            texts (str): 要生成 embedding 的文本.
        Return:
            embeddings (List[float]): 输入文本的 embedding，一个浮点数值列表.
        """
        embeddings = self.client.embeddings.create(model=self.model, input=text)
        return embeddings.data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    class Config:
        arbitrary_types_allowed = True

class RAG_Memeory:
    def __init__(self, config: Dict):
        self.config = config
        self.documents = []
        self.chunked_documents = []
        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=config["chunk_size"], 
                                                            chunk_overlap=config["chunk_overlap"])
        self.embeddings = DoubaoEmbeddings(model=os.environ["EMBEDDING_MODEL"])

    def __call__(self, text: str):
        return self.embeddings.embed_query(text)
    
    def load_documents(self):
        for file in os.listdir(self.config["base_dir"]):
            # 构建完整的文件路径
            file_path = os.path.join(self.config["base_dir"], file)
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                self.documents.extend(loader.load())
            elif file.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
                self.documents.extend(loader.load())
            elif file.endswith(".txt"):
                loader = TextLoader(file_path)
                self.documents.extend(loader.load())

    def split_documents(self):
        self.chunked_documents.extend(self.text_splitter.split_documents(self.documents))

    def store_documents(self):
        self.vectorstore = Qdrant.from_documents(
            documents=self.chunked_documents, 
            embedding=self.embeddings,
            location=":memory:",  # in-memory 存储
            collection_name="my_documents",
            )  # 指定collection_name
        
    def get_retirever(self, llm):
        retriever_from_llm = MultiQueryRetriever.from_llm(retriever=self.vectorstore.as_retriever(), llm=llm)
        return retriever_from_llm
    
    def init(self):
        self.load_documents()
        self.split_documents()
        self.store_documents()