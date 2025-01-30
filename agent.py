# Step 1: Install necessary packages
# Run this in your terminal or notebook
# !pip install langchain-community langchainhub langgraph

# Step 2: Import required libraries
import os
from typing_extensions import TypedDict, Annotated
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain import hub
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

# Step 3: Initialize the SQL Database
# Replace 'sqlite:///Chinook.db' with your database URI
db = SQLDatabase.from_uri("sqlite:///Chinook.db")
print(f"Database dialect: {db.dialect}")
print(f"Available tables: {db.get_usable_table_names()}")

# Step 4: Initialize the Groq Model
import getpass

if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

from langchain_groq import ChatGroq

llm = ChatGroq(model="deepseek-r1-distill-llama-70b")

# Step 5: Define the Application State
class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

# Step 6: Define the Query Output Structure
class QueryOutput(TypedDict):
    query: Annotated[str, "Syntactically valid SQL query."]

# Step 7: Convert Question to SQL Query
query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")
query_prompt_template.messages[0].prompt.template += """
Respond ONLY with the SQL query. Do not include explanations, markdown formatting, or any other text.
"""

def write_query(state: State):
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )
    
    try:
        # Try structured output first
        structured_llm = llm.with_structured_output(QueryOutput)
        result = structured_llm.invoke(prompt)
        return {"query": result["query"]}
    except Exception as e:
        print(f"Structured output failed, falling back to text parsing: {e}")
        # Fallback to text parsing
        response = llm.invoke(prompt)
        # Extract SQL query from markdown formatting if present
        if "```sql" in response.content:
            clean_query = response.content.split("```sql")[-1].split("```")[0].strip()
        else:
            clean_query = response.content.strip()
        return {"query": clean_query}

# Step 8: Execute the SQL Query
def execute_query(state: State):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    try:
        result = execute_query_tool.invoke(state["query"])
        return {"result": result}
    except Exception as e:
        return {"result": f"Error executing query: {str(e)}"}

# Step 9: Generate the Answer
def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    try:
        response = llm.invoke(prompt)
        return {"answer": response.content}
    except Exception as e:
        return {"answer": f"Error generating answer: {str(e)}"}

# Step 10: Orchestrate with LangGraph
graph_builder = StateGraph(State)
graph_builder.add_node("write_query", write_query)
graph_builder.add_node("execute_query", execute_query)
graph_builder.add_node("generate_answer", generate_answer)

graph_builder.set_entry_point("write_query")
graph_builder.add_edge("write_query", "execute_query")
graph_builder.add_edge("execute_query", "generate_answer")

graph = graph_builder.compile()

# Step 11: Test the Application
def run_query(question: str):
    print(f"\nProcessing question: {question}")
    for step in graph.stream(
        {"question": question}, 
        stream_mode="updates"
    ):
        for key, value in step.items():
            print(f"{key}: {value}")

# Basic test
run_query("How many employees are there?")

# Step 12: Enhanced Human-in-the-Loop
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory, interrupt_before=["execute_query"])

def human_in_the_loop(question: str):
    print(f"\nProcessing with human approval: {question}")
    config = {"configurable": {"thread_id": "1"}}
    
    try:
        # Initial execution up to query generation
        for step in graph.stream(
            {"question": question},
            config,
            stream_mode="updates",
        ):
            print(step)
        
        # Get generated query
        thread_state = memory.get(config)
        generated_query = thread_state["values"]["write_query"]["query"]
        
        print(f"\nGenerated SQL Query:\n{generated_query}")
        user_approval = input("Do you want to execute this query? (yes/no): ").lower()
        
        if user_approval == "yes":
            # Continue execution
            for step in graph.stream(
                None,
                config,
                stream_mode="updates",
            ):
                print(step)
            # Get final answer
            thread_state = memory.get(config)
            print("\nFinal Answer:", thread_state["values"]["generate_answer"]["answer"])
        else:
            print("Execution cancelled by user.")
            
    except KeyboardInterrupt:
        print("\nOperation aborted by user.")
    except Exception as e:
        print(f"Error in processing: {str(e)}")

# Run with human approval flow
human_in_the_loop("How many customers are based in Germany?")