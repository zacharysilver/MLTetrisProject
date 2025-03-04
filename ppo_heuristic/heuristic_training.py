from tetris_gamestate import TetrisGameState
from heuristic_agent import HeuristicTetrisAgent
state = TetrisGameState()
agent = HeuristicTetrisAgent()
state.reset() 
for i in range(1000):
    reward, done = state.step(agent.choose_action(state))
    print(reward)
    if done:
        break
state.step(0)
print(state.__dict__)
