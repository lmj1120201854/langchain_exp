import sys
import os
import langchain

from modules.inputs.input import *
from modules.tools.rag import RAG_Memeory
from modules.tools.load_config import load_module_config, load_env_config
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

import warnings
warnings.filterwarnings("ignore")

def main():
    load_env_config('langchain_exp\configs\env.yaml')
    config = load_module_config('langchain_exp\configs\module.yaml')
    # print(config)

    # LLM
    llm = ChatOpenAI(model=os.environ["OPENAI_MODEL"], temperature=config["LLM"]["temperature"])

    # RAG_Memeory
    rag = RAG_Memeory(config["RAG"])
    rag.init()
    retirever = rag.get_retirever(llm=llm)

    # RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retirever)

    result = qa_chain({"query": '易速鲜花董事长致辞中提到了什么？'})
    print(result)

if __name__ == "__main__":
    main()
