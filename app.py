from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import os
import random
import time

# Get the response from the data
def get_response(knowledge_base, prompt):
  docs = knowledge_base.similarity_search(prompt)
  llm = OpenAI()
  chain = load_qa_chain(llm, chain_type="stuff")
  response = chain.run(input_documents=docs, question=prompt)
  
  with get_openai_callback() as cb:
    print(cb)
    print(response)
    
  for word in response.split():
      yield word + " "
      time.sleep(0.05)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with your doc")
    st.header("DevObi Document Chat.")
    st.subheader("Interact with your document 💬",divider=True)
    st.text("Just upload a file, then chat with it.")
    st.text("It runs on the browser, so it's completely secure.")
    

    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # Clear the messages when a new PDF is uploaded
    if pdf is None:
      st.session_state.messages = []

    # extract the text
    if pdf is not None:
      pdf_reader = PdfReader(pdf)
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()
        
      # split into chunks
      text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
      chunks = text_splitter.split_text(text)
      
      # create embeddings
      embeddings = OpenAIEmbeddings()
      knowledge_base = FAISS.from_texts(chunks, embeddings)
      
      # *** CHATGPT STUFF ***
      # Initialize chat history
      if "messages" not in st.session_state:
        st.session_state.messages = []
         
      # Display chat messages from history on app rerun
      for message in st.session_state.messages:
        with st.chat_message(message["role"]):
          st.markdown(message["content"])
      
      # Accept user input
      if prompt := st.chat_input("Ask a question about the document"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
          st.markdown(prompt)
          
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
          response = st.write_stream(get_response(knowledge_base, prompt))
          st.session_state.messages.append({"role": "assistant", "content": response})


  # *** END OF CHATGPT STUFF

      # show user input
      # user_question = st.text_input("Ask a question about your PDF:", key="question")
      
      # if user_question:
      #  st.write(user_question + '\nthinking...')
      #  docs = knowledge_base.similarity_search(user_question)
        
      #  llm = OpenAI()
      #  chain = load_qa_chain(llm, chain_type="stuff")
      #  response_thread = ""
      #  with get_openai_callback() as cb:
      #    response = chain.run(input_documents=docs, question=user_question)
      #    response_thread = response + "\n" + response_thread
      #    print(cb)
      #  st.write(response_thread)
           


if __name__ == '__main__':
    main()
