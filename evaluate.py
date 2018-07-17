import keras
from keras.models import load_model
from agent.agent import Agent
from functions import *
import sys
# we check inputs
if len(sys.argv) != 3:
	print("Usage: python evaluate.py [stock] [model]")


# we save inputs
stock_name, model_name = sys.argv[1], sys.argv[2]
model = load_model("models/" + model_name)
window_size = model.layers[0].input.shape.as_list()[1]
# get agent and data
agent = Agent(window_size, True, model_name)
data = getStockDataVec(stock_name)
data_len = len(data) -1
batch_size = 32
state = getState(data, 0,window_size+1)
total_profit = 0
# inventory of stock
agent.inventory = []

for t in range(data_len):
    # we get action form agent
    action = agent.act(state)
    # sit action
    next_state = getState(data, t+1, window_size+1)
    reward = 0
    # buy action
    if action == 1:
        agent.inventory.append(data[t])
        print("buy: " + formatPrice(data[t]))
    # sell action and if we have more than 0 stock
    elif action == 2 and len(agent.inventory) > 0:
        bought_price = agent.inventory.pop(0)
        reward = max(data[t]-bought_price, 0)
        total_profit += data[t] - bought_price
        print("Sell: "+formatPrice(data[t])+" | profit: "+formatPrice(data[t]-bought_price))
    # we save if we almost end loop
    if t == data_len - 1:
        done = True
    else:
        done = False
    # we send dtata to agent
    agent.memory.append((state, action, reward, next_state, done))
    state = next_state
    # we print how much we earn
    if done:
        print(stock_name + " Total Profit: " + formatPrice(total_profit))
