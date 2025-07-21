import streamlit as st
from helper import get_pdf_text, get_text_chunks, get_vector_store
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Initialize Ollama Gemma 3 model
def get_conversational_chain(vector_store):  # Changed back to original name
    llm = Ollama(
        model="gemma3:4b",
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repeat_penalty=1.1
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={'k': 3}),
        memory=memory,
        return_source_documents=True,
        get_chat_history=lambda h: h
    )
    return conversation_chain

def user_input(user_question):
    if st.session_state.conversation:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chatHistory = response['chat_history']
        for i, message in enumerate(st.session_state.chatHistory):
            if i % 2 == 0:
                st.write(f"**User:** {message.content}")
            else:
                st.write(f"**Gemma 3:** {message.content}")
    else:
        st.warning("Please process PDF files first")

def main():
    st.set_page_config("Information Retrieval with Gemma 3")
    st.header("Information Retrieval System with Gemma 3💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = None
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", 
            accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                try:
                    # Get PDF text
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text:
                        st.error("No text could be extracted from the PDFs")
                        return
                    
                    # Get the text chunks
                    text_chunks = get_text_chunks(raw_text)
                    if not text_chunks:
                        st.error("No valid text chunks could be created")
                        return
                    
                    # Create vector store
                    vector_store = get_vector_store(text_chunks)
                    
                    # Create conversation chain with Gemma 3
                    st.session_state.conversation = get_conversational_chain(vector_store)
                    st.success("Processing complete! You can now ask questions.")
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")


if __name__ == "__main__":
    main()