
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


finalPrompt = PromptTemplate(
    input_variables=["question", "context"],
    template="You are a Medical Professional who have all knowledge about medicine and health.Analyze the question and Use the pieces of information provided in the context to answer user's question.Dont provide wrong answer.And Never answer question which are not related to health and medical. \n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:",
)


def load_llm():
    load_dotenv()
    llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    temperature=0.5,
    max_new_tokens=256
    )
    return llm


def main():
    st.set_page_config(page_title="Infinity Medicare", page_icon=":hospital:", layout="wide")
    st.title("Ask Infinite MediCare")
    st.write("This is a medical chatbot that can answer your questions based on the provided context. Please enter your question below.")
    

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt=st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})

        

        try: 
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain=RetrievalQA.from_chain_type(
                llm=load_llm(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt':finalPrompt}
            )

            response = qa_chain.invoke({'query': prompt})
            result = response["result"]
            source_documents = response["source_documents"]

            # Prepare formatted source documents
            # sources_text = "#### 📚 Source Documents:\n"
            # for i, doc in enumerate(source_documents):
            #     sources_text += f"**Source {i+1}:**\n```\n{doc.page_content.strip()}\n```\n\n"

            # Final result to show
            result_to_show = f"### 🩺 Answer:\n{result}"

            # Display in Streamlit chat
            st.chat_message('assistant').markdown(result_to_show)

            # Save message to session state
            st.session_state.messages.append({
                'role': 'assistant',
                'content': result_to_show
            })


        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()