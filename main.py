import streamlit as st
from langchain_helper import get_llm_code


st.title("AtliQ T-Shirts: Database Q&A 👕")

st.write("Ask any question about the AtliQ T-Shirts database, and get answers instantly!")

question = st.text_input("Enter your question:")

response_placeholder = st.empty()

if question:
    with st.spinner("Fetching answer..."):
        try:
            
            chain = get_llm_code()
        
            response = chain.run(question)
            
    
            st.header("Answer")
            response_placeholder.write(response)
        except Exception as e:
            
            st.error(f"An error occurred: {e}")


st.markdown("---")
st.caption("Powered by LangChain and Streamlit")
