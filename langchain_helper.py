from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain.agents import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
import os

import os
from dotenv import load_dotenv
load_dotenv()



def get_llm_code():
 db_user = os.getenv("DB_USER")
 db_password = os.getenv("DB_PASSWORD")
 db_host = os.getenv("DB_HOST")
 db_name = os.getenv("DB_NAME")

 db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",sample_rows_in_table_info=3)

#  print(db.table_info)
 os.getenv("GOOGLE_API_KEY")
 llm = ChatGoogleGenerativeAI(model="gemini-pro")

 toolkit = SQLDatabaseToolkit(db=db, llm=llm)
 agent = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)

 return agent
 

