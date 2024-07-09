#!/usr/bin/env python
# coding: utf-8

# The following work will implement the following:
# 1) given a stock ticker, we will retrieve latest news of the company (news agent)
# 2) given the news, the sentiment agent will evaluate the linked sentiment to the news
# 3) given the sentiment, a final agent will emit a recommendation to buy, sell or hold the ticker.

# Discussion with claude at
# https://claude.ai/chat/4a1f5da0-e13e-441d-9597-af8626152811

# In[51]:


from langchain.agents import Tool, AgentExecutor
from langchain.prompts import PromptTemplate 
from langchain import LLMChain
from langchain.prompts import ChatPromptTemplate
import langgraph
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


# In[67]:


from typing import List
import os
from dotenv import load_dotenv
import random
from datetime import datetime, timedelta
import numpy as np
import logging


# In[68]:


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# In[53]:


load_dotenv()


# In[54]:


type(os.environ["OPENAI_API_KEY"])


# In[55]:


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


# In[56]:


#test_openai_connection()


# # Base classes

# In[57]:


class StockAgent:
    def __init__(self, role, llm):
        self.role = role
        self.llm = llm
        self.memory = []

    def act(self, state):
        # ABC to initiate actions
        pass


# # Stock market

# In[58]:


class Stock:
    def __init__(self, symbol, initial_price):
        self.symbol = symbol
        self.price = initial_price
        self.volume = 0
        self.history = []

    def update_price(self, change_percent):
        self.price *= (1 + change_percent)
        self.price = round(self.price, 2)

class StockMarket:
    def __init__(self):
        self.stocks = {
            "ABC": Stock("ABC", 150),
            "DEF": Stock("DEF", 2800),
            "GHI": Stock("GHI", 3300),
        }
        self.current_time = datetime.now()
        self.events = []

    def update(self):
        self.current_time += timedelta(minutes=5)  # Simulate 5-minute intervals
        self._update_stock_prices()
        self._generate_events()
        self._record_history()

    def _update_stock_prices(self):
        for stock in self.stocks.values():
            # Simulate price movement with some randomness
            change_percent = random.uniform(-0.02, 0.02)  # -2% to +2% change
            stock.update_price(change_percent)
            stock.volume = random.randint(1000, 100000)  # Simulate trading volume

    def _generate_events(self):
        # Randomly generate market events
        if random.random() < 0.05:  # 5% chance of an event occurring
            event_types = [
                "Earnings Report",
                "Economic Indicator Release",
                "Geopolitical Event",
                "Industry News",
                "Regulatory Announcement"
            ]
            event = random.choice(event_types)
            affected_stock = random.choice(list(self.stocks.keys()))
            self.events.append(f"{event} affecting {affected_stock}")

    def _record_history(self):
        for stock in self.stocks.values():
            stock.history.append({
                "time": self.current_time,
                "price": stock.price,
                "volume": stock.volume
            })


    def get_state(self):
        return {
            "time": self.current_time,
            "stocks": {symbol: {"price": stock.price, "volume": stock.volume} 
                       for symbol, stock in self.stocks.items()},
            "events": self.events
        }

    def get_stock_history(self, symbol):
        if symbol not in self.stocks:
            return None
        return self.stocks[symbol].history
    def get_state(self):
        return {
            "time": self.current_time,
            "stocks": {symbol: {"price": stock.price, "volume": stock.volume} 
                       for symbol, stock in self.stocks.items()},
            "events": self.events
        }

    def place_order(self, symbol, order_type, quantity):
        if symbol not in self.stocks:
            return False, "Invalid stock symbol"
        
        stock = self.stocks[symbol]
        if order_type == "buy":
            # Simulate a buy order (in a real system, this would involve more complex logic)
            stock.volume += quantity
            return True, f"Bought {quantity} shares of {symbol} at {stock.price}"
        elif order_type == "sell":
            # Simulate a sell order
            stock.volume -= quantity
            return True, f"Sold {quantity} shares of {symbol} at {stock.price}"
        else:
            return False, "Invalid order type"


# ## Traders

# In[75]:


class Trader(StockAgent):
    def __init__(self, name, initial_balance, llm, market_outlook):
        super().__init__("Trader", llm)
        self.name = name
        self.balance = initial_balance
        self.portfolio = {}
        self.llm = llm
        self.market_outlook=market_outlook
        self.logger = logging.getLogger(name)

    def act(self, state):
        market = state['market']
        market_state = market.get_state()
        
        for symbol, stock_data in market_state['stocks'].items():
            analysis = self.analyze_stock(symbol, stock_data, market)
            if analysis['action'] != 'hold':
                self.execute_trade(symbol, analysis['action'], analysis['quantity'], stock_data['price'], market)

        return state

    def analyze_stock(self, symbol, stock_data, market):
        # Get stock history
        history = market.get_stock_history(symbol)
        if not history or len(history) < 20:
            return {'action': 'hold', 'quantity': 0}

        # Calculate simple moving averages
        prices = [entry['price'] for entry in history]
        sma_5 = sum(prices[-5:]) / 5
        sma_20 = sum(prices[-20:]) / 20

        current_price = stock_data['price']

        # Prepare prompt for LLM
        prompt = ChatPromptTemplate.from_template("""
        You are a stock trader assistant. Analyze the following data and suggest an action:
        Stock: {symbol}
        Current Price: {current_price}
        5-day SMA: {sma_5}
        20-day SMA: {sma_20}
        Recent events: {events}

        Your market outlook is {market_outlook}

        Suggest an action (buy, sell, or hold) and a quantity (1-100) based on this data.
        Respond in the format: ACTION,QUANTITY
        """)

        self.logger.info(f"Sending prompt to LLM for {symbol}")
        response = self.llm(prompt.format_messages(
            symbol=symbol,
            current_price=current_price,
            sma_5=sma_5,
            sma_20=sma_20,
            events=", ".join(market.get_state()['events']),
            market_outlook=self.market_outlook
        ))

        self.logger.info(f"LLM response for {symbol}: {response.content}")

        action, quantity = response.content.strip().split(',')
        return {'action': action.lower(), 'quantity': int(quantity)}

    def execute_trade(self, symbol, action, quantity, price, market):
        if action == 'buy':
            cost = quantity * price
            if self.balance >= cost:
                success, message = market.place_order(symbol, 'buy', quantity)
                if success:
                    self.balance -= cost
                    self.portfolio[symbol] = self.portfolio.get(symbol, 0) + quantity
                    self.logger.info(f"{self.name} bought {quantity} shares of {symbol} at {price}")
                else:
                    self.logger.warning(f"Trade failed: {message}")
            else:
                self.logger.warning(f"{self.name} doesn't have enough balance to buy {quantity} shares of {symbol}")
        elif action == 'sell':
            if symbol in self.portfolio and self.portfolio[symbol] >= quantity:
                success, message = market.place_order(symbol, 'sell', quantity)
                if success:
                    self.balance += quantity * price
                    self.portfolio[symbol] -= quantity
                    self.logger.info(f"{self.name} sold {quantity} shares of {symbol} at {price}")
                else:
                    self.logger.warning(f"Trade failed: {message}")
            else:
                self.logger.warning(f"{self.name} doesn't have enough shares of {symbol} to sell")

    def get_portfolio_value(self, market):
        market_state = market.get_state()
        total_value = self.balance
        for symbol, quantity in self.portfolio.items():
            if symbol in market_state['stocks']:
                total_value += quantity * market_state['stocks'][symbol]['price']
        return total_value


# ## Analysts
# 

# In[76]:


# TBD


# In[77]:


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


# In[78]:


market = StockMarket()
trader1 = Trader("bull", initial_balance=1000000, llm=ChatOpenAI(), market_outlook="bullish")
trader2 = Trader("bear", initial_balance=1000000, llm=ChatOpenAI(), market_outlook="bearish")
trader3 = Trader("neutral", initial_balance=1000000, llm=ChatOpenAI(), market_outlook="neutral")


# In[79]:


# Simulate market updates
for i in range(10):
    market.update()
    state = {"market": market}
    trader1.act(state)
    trader2.act(state)
    trader3.act(state)
    
    print(f"\nAfter iteration {i+1}:")
    print(f"Trader 1 portfolio value: ${trader1.get_portfolio_value(market):.2f}")
    print(f"Trader 2 portfolio value: ${trader2.get_portfolio_value(market):.2f}")
    print(f"Trader 3 portfolio value: ${trader3.get_portfolio_value(market):.2f}")


# In[87]:


state["market"].


# In[65]:


print(f"Trader 1 final portfolio value: ${trader1.get_portfolio_value(market):.2f}")
print(f"Trader 2 final portfolio value: ${trader2.get_portfolio_value(market):.2f}")
print(f"Trader 3 final portfolio value: ${trader3.get_portfolio_value(market):.2f}")


# # All together

# In[10]:


def create_stock_market_graph():
    market = StockMarket()

    trader1 = Trader("Trader1", ChatOpenAI())
    trader2 = Trader("Trader2", ChatOpenAI())
    #analyst1 = Analyst("Analyst1", ChatOpenAI())
    #analyst2 = Analyst("Analyst2", ChatOpenAI())

    workflow = langgraph.Graph()

    # Define nodes
    workflow.add_node("market_update", market.update)
    workflow.add_node("trader1", trader1.act)
    workflow.add_node("trader2", trader2.act)
    #workflow.add_node("analyst1", analyst1.act)
    #workflow.add_node("analyst2", analyst2.act)

    # Define edges (information flow)
    workflow.add_edge("market_update", "trader1")
    workflow.add_edge("market_update", "trader2")
    #workflow.add_edge("market_update", "analyst1")
    #workflow.add_edge("market_update", "analyst2")

    #workflow.add_edge("analyst1", "trader1")
    #workflow.add_edge("analyst1", "trader2")
    #workflow.add_edge("analyst2", "trader1")
    #workflow.add_edge("analyst2", "trader2")

    return workflow


# In[ ]:


def run_simulation(num_steps):
    graph = create_stock_market_graph()
    state = {"market": StockMarket().get_state()}

    for _ in range(num_steps):
        market.update()
        state = {"market": market}
        trader.act(state)
        # Log results, update visualizations, etc.

run_simulation(100)  # Run for 100 time steps

