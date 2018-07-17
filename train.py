from agent.agent import Agent
from functions import *
import sys
# we check inputs
if len(sys.argv) != 4:
	print("Usage: python train.py [stock] [window] [episodes]")
# we save inputs
stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
# get agent
agent = Agent(window_size)
# get data of stock
data = getStockDataVec(stock_name)
data_len = len(data) - 1
batch_size = 32
# now we iterate over
for episode in range(episode_count + 1):
    print("episode " + str(episode) + "/" + str(episode_count))
    state = getState(data, 0, window_size+1)
    # starting value for agent
    total_profit = 0
    agent.inventory = []
    for t in range(data_len):
        # get action form agent
        action = agent.act(state)
        # sit action
        next_state=getState(data,t+1,window_size+1)
        reward = 0
        # buy action
        if action ==1:
            agent.inventory.append(data[t])
            print("buy: " + formatPrice(data[t]))
        # sell action and if we have more than 0 stock
        elif action == 2 and len(agent.inventory) > 0 :
            bought_price = agent.inventory.pop(0)
            reward = max(data[t]-bought_price, 0)
            total_profit += data[t] - bought_price
            print("sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))
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
        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)
    # we save model every 10 episodes
    if episode % 10 ==0:
        agent.model.save("models/model_ep" + str(episode))
