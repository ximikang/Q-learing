#-*- utf-8 -*-
# qvkang
import numpy as np
import random
import turtle as t
class Cliff(object):
    def __init__(self):
        self.reward = self._reward_init()
        print(self.reward)
        self.row = 4
        self.col = 12
        self.gamma = 0.7
        self.start_state = (3, 0)
        self.end_state = (3, 11)
        self.q_matrix = np.zeros((4,12,5))
        self.main()

    def _reward_init(self):
        re = np.ones((4,12))*-5
        # 奖励
        re[1][5:8] = np.ones((3))*-1
        # 悬崖
        re[3][1:11] = np.ones((10))*-100
        #目标
        re[3][11] = 100
        return re

    def valid_action(self, current_state):
        # 判断当前状态下可以走的方向
        itemrow, itemcol = current_state
        valid = [0]
        if(itemrow-1 >= 0): valid.append(1)
        if(itemrow+1 <= self.row-1):valid.append(2)
        if(itemcol-1 >= 0): valid.append(3)
        if(itemcol+1 <= self.col-1): valid.append(4)
        return valid

    def transition(self, current_state, action):
        # 从当前状态转移到下一个状态
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
        # 得到下一步的奖励
        next_state = self.transition(current_state, action)
        next_row, next_col = next_state
        r = self.reward[next_row, next_col]
        return r
    def path(self):
        #绘图path  使用turtle的绘图库
        t.speed(10)
        t.begin_fill()
        paths = []
        current_state = self.start_state
        t.pensize(5)
        t.penup()
        t.goto(current_state)
        t.pendown()
        #移动到初始位置
        paths.append(current_state)
        while current_state != self.end_state:
            current_row, current_col = current_state
            valid_action = self.valid_action(current_state)
            valid_value = [self.q_matrix[current_row][current_col][x] for x in valid_action]
            max_value = max(valid_value)
            action = np.where(self.q_matrix[current_row][current_col] == max_value)
            print(current_state,'-------------',action)
            next_state = self.transition(current_state,int(random.choice(action[0])))
            paths.append(next_state)
            next_row,next_col = next_state
            t.goto(next_col*20, 60-next_row*20)
            current_state = next_state

    def main(self):
        #主要循环迭代
        for i in range(1000):
            current_state = self.start_state
            while current_state != self.end_state:
                action = random.choice(self.valid_action(current_state))
                next_state = self.transition(current_state, action)
                future_rewards = []
                for action_next in self.valid_action(next_state):
                    next_row, next_col = next_state
                    future_rewards.append(self.q_matrix[next_row][next_col][action_next])
                #core trasmite rule
                q_state = self.getreward(current_state, action) + self.gamma*max(future_rewards)
                current_row, current_col = current_state
                self.q_matrix[current_row][current_col][action] = q_state
                current_state = next_state
                #print(self.q_matrix)
        #绘图1000次
        for i in range(1000):
            self.path()
        print(self.q_matrix)

if __name__ == "__main__":
    Cliff()


    
