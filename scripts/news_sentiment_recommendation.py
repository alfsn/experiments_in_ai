#!/usr/bin/env python
# coding: utf-8

# The following work will implement the following:
# 1) given a stock ticker, we will retrieve latest news of the company (news agent)
# 2) given the news, the sentiment agent will evaluate the linked sentiment to the news
# 3) given the sentiment, a final agent will emit a recommendation to buy, sell or hold the ticker.

# Discussion with claude at
# https://claude.ai/chat/4a1f5da0-e13e-441d-9597-af8626152811

# In[1]:


from langchain.agents import Tool, AgentExecutor
from langchain.prompts import PromptTemplate 
from langchain import LLMChain
from langchain_openai import OpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.schema import AgentAction, AgentFinish
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents import create_react_agent
from langchain.prompts import ChatPromptTemplate


# In[2]:


from typing import List
import os
from dotenv import load_dotenv


# In[3]:


load_dotenv()


# In[4]:


type(os.environ["OPENAI_API_KEY"])


# In[5]:


def test_openai_connection():
    try:
        llm = OpenAI(temperature=0)
        
        # Create a simple prompt
        prompt = PromptTemplate(
            input_variables=["question"],
            template="Question: {question}\nAnswer:"
        )
        
        # Create a chain
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Try to run the chain
        response = chain.run("Is this connection working?")
        
        # If we get here, the API call was successful
        print("API connection successful!")
        print("Response:", response.strip())
        return True
    except Exception as e:
        print("API connection failed.")
        print("Error:", str(e))
        return False


# In[6]:


#test_openai_connection()


# # News agent

# In[7]:


search = DuckDuckGoSearchRun()


# In[8]:


tools = [
    Tool(
        name="Search",
        func=search.run,
        description="this will search for stock ticker news"
    )
]


# In[9]:


prompt = ChatPromptTemplate.from_template("""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
{agent_scratchpad}""")


# In[10]:


# Define the LLM
llm = OpenAI(temperature=0)


# # Sentiment agent

# The following class implements a way through which the sentiment agent
# 1) waits for the news agent to finish
# 2) parses the response the news agent gives, incorporating "Final Answer:" as the key place, as defined by our prompt

# Agent definition:

# In[11]:


agent = create_react_agent(llm, tools, prompt)


# In[12]:


agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, 
    tools=tools, 
    verbose=True,
    handle_parsing_errors=True
)


# In[13]:


def analyze_stock(ticker):
    query = f"Latest news about {ticker} stock"
    result = agent_executor.invoke({"input": query})
    return result["output"]


# In[14]:


if __name__ == "__main__":
    ticker = input("Enter a stock ticker symbol: ")
    try:
        analysis = analyze_stock(ticker)
        print(analysis)
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check your API key and network connection.")

