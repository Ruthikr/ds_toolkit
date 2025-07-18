# test_tools.py
import os
import json
from langchain.agents import create_react_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from database_toolkit import DatabaseToolkit
from financial_toolkit import FinancialToolkit
# Load environment variables
load_dotenv()
google_api_key = os.getenv("GEMINI_API_KEY")
if not google_api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

# Initialize toolkit
database_toolkit = DatabaseToolkit()
database_tools = database_toolkit.get_tools()

tool_names = [tool.name for tool in database_tools]

# The agent prompt
custom_prompt = PromptTemplate.from_template(
    """You are a data engineer agent. Your task is to fulfill data requests using the available tools.

Available Tools:
{tools}

Use the following format:

Question: {input}
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
)

# Set up the agent
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)
agent = create_react_agent(llm, database_tools, custom_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=database_tools,
    verbose=True,
    max_iterations=15, # Limit iterations for this test
    return_intermediate_steps=True,
    handle_parsing_errors=True
)

# The input prompt for the test

new_prompt = {
    "input": f"""
    connect to sqlite db ,the db is in current directory the filename is movies.sqlite.
  
Connect to the movies.sqlite database.

Perform the following:
1. List all movies released between 1990 and 2000 with rating > 7.5 and revenue over $10 million.
2. For each of the top 3 directors (by movie count), list their 2 highest-rated movies.
3. Find the average budget and revenue per year (group by release year).
4. For all movies where revenue is NULL or 0, suggest they may have flopped or were unreleased — list their titles.
5. Export the results of question 1 and 3 to CSV.
"""
}

# Create the database for the test


# Run the agent
try:
    print("Starting agent execution with complex database tasks...")
    res = agent_executor.invoke(new_prompt)
    print("\nAgent execution finished.")
    print(json.dumps(res, indent=2))
        
except Exception as e:
    print(f"\nAgent execution failed: {str(e)}")
