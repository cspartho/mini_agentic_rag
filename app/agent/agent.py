from llm.azure_configs import get_azure_llm
from .prompts import SYSTEM_PROMPT, GROUNDING_PROMPT

class RAGAgent:
    def __init__(self, tools):
        self.llm = get_azure_llm()
        self.tools = tools

    def run(self, query, search_tool):
        context = search_tool.invoke(query)
        
        formatted_prompt = GROUNDING_PROMPT.format(context=context, question=query)
        
        messages = [
            ("system", SYSTEM_PROMPT),
            ("human", formatted_prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content