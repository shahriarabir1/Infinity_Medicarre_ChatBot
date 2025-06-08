from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    temperature=0.5,
    max_new_tokens=256
)


DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="You are a Medical Professional who have all knowledge about medicine and health.Analyze the question and Use the pieces of information provided in the context to answer user's question.Dont provide wrong answer.And Never answer question which are not related to health and medical. You should match and analyze answer with the context. \n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:",
)

retriever = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)


user_question = input("What is issue tell the doctor: ")
response = retriever.invoke({"query": user_question})
print("RESULT: ", response["result"])
