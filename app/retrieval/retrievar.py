class RetrieverLogic:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def retrieve(self, query: str):
        docs = self.vectorstore.search(query, search_type="mmr", k=3)
        return "\n".join([d.page_content for d in docs])