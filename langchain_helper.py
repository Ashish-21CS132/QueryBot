from langchain_google_genai import GoogleGenerativeAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from langchain.prompts.prompt import PromptTemplate
from few_shots import few_shots

import os
from dotenv import load_dotenv

load_dotenv()


def get_llm_code():
    
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_name = os.getenv("DB_NAME")
    
    # connect database
    db = SQLDatabase.from_uri(
        f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",
        sample_rows_in_table_info=3,
    )
    
    #connect to the model
    llm = GoogleGenerativeAI(model="models/text-bison-001")

    #embeddings of the few_shots
    embeddings = HuggingFaceEmbeddings()
    to_vectorize = [" ".join(example.values()) for example in few_shots]
    vectordb = FAISS.from_texts(
        texts=to_vectorize, embedding=embeddings, metadatas=few_shots
    )
    
    #example selector
    example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectordb,
    k=2,
    )
    
    #promt template
    example_prompt = PromptTemplate(
    input_variables=["Question", "SQLQuery", "SQLResult","Answer",],
    template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
    )
    
    #yaha par saara karyakram hai to chain me jaega
    few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix=_mysql_prompt,
    suffix=PROMPT_SUFFIX,
    input_variables=["input", "table_info", "top_k"], #These variables are used in the prefix and suffix
    )
    
    #final chain
    
    new_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=few_shot_prompt)
    
    return new_chain




