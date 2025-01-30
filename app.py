from typing import TypedDict
import streamlit as st
import os
import re
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain import hub
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Initialize the SQL Database
db = SQLDatabase.from_uri("sqlite:///Chinook.db")

# Initialize Groq Model
model_name = "llama-3.2-90b-vision-preview"
llm = ChatGroq(model=model_name, api_key=st.secrets["GROQ_API_KEY"])

# Convert Question to SQL Query
query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")
query_prompt_template.messages[0].prompt.template += """
Respond ONLY with the SQL query. Do not include explanations, markdown formatting, or any other text.
"""

def generate_sql(question: str):
    """Generate SQL query from natural language question"""
    try:
        prompt = query_prompt_template.invoke({
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": question,
        })
        
        # Get raw response
        response = llm.invoke(prompt)
        content = response.content
        
        # Clean SQL from markdown formatting
        if "```sql" in content:
            clean_query = content.split("```sql")[-1].split("```")[0].strip()
        else:
            clean_query = content.strip()
            
        return clean_query
        
    except Exception as e:
        st.error(f"SQL Generation Error: {str(e)}")
        return None

def execute_sql(query: str):
    """Execute SQL query and return results"""
    try:
        execute_query_tool = QuerySQLDatabaseTool(db=db)
        return execute_query_tool.invoke(query)
    except Exception as e:
        st.error(f"Execution Error: {str(e)}")
        return None
    
def generate_answer(question: str, query: str, result: str) -> str:
    """Generate natural language answer from query results"""
    try:
        prompt = f"""
        Explain the following database query results in natural language:
        
        Question: {question}
        SQL Query: {query}
        Results: {result}
        
        Provide a concise answer using the results. Avoid technical terms.
        """
        response = llm.invoke(prompt)
        
        # Clean up any XML/thinking tags
        answer = re.sub(r'<\/?think>', '', response.content)
        return answer.strip()
    except Exception as e:
        st.error(f"Answer generation failed: {str(e)}")
        return None

# Streamlit interface
st.title("🤖 SQL Query Agent")

# Sidebar with styled content
with st.sidebar:
    st.markdown("""
    <div style="border-bottom: 1px solid #ddd; padding-bottom: 10px; margin-bottom: 20px;">
        <h1 style="color: #2e86c1; font-size: 1.5em;">🔍 Database Explorer</h1>
    </div>
    """, unsafe_allow_html=True)

    # Database Information Section
    st.markdown("### 📊 Database Connection")
    with st.container(border=True):
        st.markdown("""
        <div style="font-size: 0.9em;">
            <div style="margin-bottom: 8px;">
                <span style="color: #666;">Database:</span> 
                <span style="font-family: monospace;">Chinook.db</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <div>
                    <span style="color: #666;">Dialect:</span> 
                    <code>{dialect}</code>
                </div>
                <div>
                    <span style="color: #666;">Tables:</span> 
                    <code>{table_count}</code>
                </div>
            </div>
            <div>
                <div style="color: #666; margin-bottom: 4px;">Tables List:</div>
                {tables}
            </div>
        </div>
        """.format(
            dialect=db.dialect,
            table_count=len(db.get_usable_table_names()),
            tables="\n".join([f"• <code>{table}</code>" for table in db.get_usable_table_names()])
        ), unsafe_allow_html=True)

    # Model Information Section
    st.markdown("### 🤖 AI Configuration")
    with st.container(border=True):
        st.markdown(f"""
        <div style="font-size: 0.9em;">
            <div style="margin-bottom: 4px;">
                <span style="color: #666;">Model:</span> 
                <span style="font-family: monospace;">{model_name}</span>
            </div>
            <div style="margin-bottom: 4px;">
                <span style="color: #666;">Provider:</span> 
                <span>Groq</span>
            </div>
            <div>
                <span style="color: #666;">Status:</span> 
                # Change the status check in the sidebar section to:
                <span style="color: {'#28a745' if st.secrets.get("GROQ_API_KEY") else '#dc3545'}">
                    {'Connected' if st.secrets.get("GROQ_API_KEY") else 'Disconnected'}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Main content
question = st.text_input("Ask a question about the database:", "")

if question:
    # Generate SQL
    with st.spinner("Generating SQL query..."):
        sql_query = generate_sql(question)
    
    if sql_query:
        st.subheader("Generated SQL Query")
        st.code(sql_query, language="sql")
        
        # Execute and show results
        with st.spinner("Executing query..."):
            result = execute_sql(sql_query)
        
        if result:
            st.subheader("Query Results")
            st.write(result)
            
            # Generate and show natural language answer
            with st.spinner("Generating natural language explanation..."):
                answer = generate_answer(question, sql_query, result)
            
            if answer:
                st.subheader("Natural Language Answer")
                st.markdown(f"**{answer}**")
