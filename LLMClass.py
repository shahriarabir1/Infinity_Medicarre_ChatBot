from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv


class MedicalChatbot:
    def __init__(self, db_path: str, huggingface_repo_id: str, hf_token: str):
        load_dotenv()
        self.db_path = db_path
        self.repo_id = huggingface_repo_id
        self.hf_token = hf_token
        self.llm = self._load_llm()
        self.embedding_model = self._load_embedding_model()
        self.vector_db = self._load_vectorstore()
        self.qa_chain = self._build_qa_chain()

    def _load_llm(self):
        return HuggingFaceEndpoint(
            repo_id=self.repo_id,
            temperature=0.5,
            model_kwargs={
                "token": self.hf_token,
                "max_length": "512"
            }
        )

    def _load_embedding_model(self):
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def _load_vectorstore(self):
        return FAISS.load_local(
            self.db_path,
            self.embedding_model,
            allow_dangerous_deserialization=True
        )

    def _get_prompt_template(self):
        return PromptTemplate(
            input_variables=["question", "context"],
            template="""Use the pieces of information provided in the context to answer user's question.
If you don’t know the answer, just say that you don’t know. Don’t try to make up an answer.
Don’t provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please."""
        )

    def _build_qa_chain(self):
        prompt = self._get_prompt_template()
        retriever = self.vector_db.as_retriever(search_kwargs={"k": 3})
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

    def ask(self, question: str):
        result = self.qa_chain.invoke({"query": question})
        return result["result"], result["source_documents"]
