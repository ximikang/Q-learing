#-*- utf-8 -*-
import numpy as np
import random
class Cliff(object):
    def __init__(self):
        self.reward = self._reward_init()
        print(self.reward)
        self.row = 4
        self.col = 12
        self.gamma = 0.7
        self.q_matrix = np.zeros((4,12,5))
        self.main()

    def _reward_init(self):
        re = np.ones((4,12))*-5
        # 奖励
        re[1][5:8] = np.ones((3))*-1
        # 悬崖
        re[3][1:11] = np.ones((10))*-100
        return re

    def valid_action(self, current_state):
        itemrow, itemcol = current_state
        valid = [0]
        if(itemrow-1 >= 0): valid.append(1)
        if(itemrow+1 <= self.row-1):valid.append(2)
        if(itemcol-1 >= 0): valid.append(3)
        if(itemcol+1 <= self.col-1): valid.append(4)
        return valid

    def transition(self, current_state, action):
        itemrow, itemcol = current_state
        if (action is 0):   next_state = current_state
        if (action is 1):   next_state = (itemrow-1, itemcol)
        if (action is 2):   next_state = (itemrow+1, itemcol)
        if (action is 3):   next_state = (itemrow, itemcol-1)
        if (action is 4):   next_state = (itemrow, itemcol+1)
        return(next_state)
    def _indextoPosition(self,index):
        index += 1
        itemrow = int(np.floor(index/self.col))
        itemcol = index%self.col
        return(itemrow, itemcol)

    def _positiontoIndex(self,itemrow,itemcol):
        itemindex = (itemrow)*self.col+itemcol-1
        return itemindex
    
    def getreward(self, current_state, action):
        next_state = self.transition(current_state, action)
        next_row, next_col = next_state
        r = self.reward[next_row, next_col]
        return r
    def main(self):
        for i in range(1000):
            start_state = (3, 0)
            current_state = start_state
            while current_state != (3, 11):
                action = random.choice(self.valid_action(current_state))
                next_state = self.transition(current_state, action)
                future_rewards = []
                for action_next in self.valid_action(next_state):
                    next_row, next_col = next_state
                    future_rewards.append(self.q_matrix[next_row][next_col][action_next])
                q_state = self.getreward(current_state, action) + self.gamma*max(future_rewards)
                current_row, current_col = current_state
                self.q_matrix[current_row][current_col][action] = q_state
                current_state = next_state
                #print(self.q_matrix)
        print(self.q_matrix)
if __name__ == "__main__":
    Cliff()


    
