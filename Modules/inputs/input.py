import langchain
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import FewShotPromptTemplate
from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import (
    ChatMessagePromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

class Input:
    def __init__(self, ):
        self.sys_message = SystemMessagePromptTemplate()
        self.user_message = HumanMessagePromptTemplate()
        self.output_message = AIMessagePromptTemplate()

    def __call__(self, text: str):
        return self.prompt(text)

