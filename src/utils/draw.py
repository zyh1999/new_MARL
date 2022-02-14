from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import  time
# 定义Summary_Writer
writer = []
value = []
value_q_mix=None
value_return=None
value_grad = []
step = []
path='Draw/'+str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
print(path)
def draw_value():
    for id in range(value[0].shape[-1]):
        tmp = SummaryWriter('./'+path+'/' + "agent_value"+str(id))
        writer.append(tmp)
        for i in range(len(step)):
            writer[-1].add_scalar('agent_q_value', torch.tensor(value[i][id]), global_step=step[i])
        writer[-1].close()

def draw_grad():
    for id in range(value[0].shape[-1]):
        tmp = SummaryWriter('./'+path+'/' +"agent_grad"+ str(id))
        writer.append(tmp)
        for i in range(len(step)):
            writer[-1].add_scalar('agent_q_grad', torch.tensor(value_grad[i][id]), global_step=step[i])
        writer[-1].close()

def draw_estimate():
    tmp = SummaryWriter('./'+path+'/' + 'Q_mix')
    writer.append(tmp)
    for i in range(len(step) - 1):
        writer[-1].add_scalar('estimate', torch.tensor(value_q_mix[0][i][0]), global_step=step[i])
    writer[-1].close()

    tmp = SummaryWriter('./'+path+'/' + 'return')
    writer.append(tmp)
    sum = value_return.sum()
    for i in range(len(step) - 1):
        writer[-1].add_scalar('estimate', torch.tensor(sum), global_step=step[i])
        sum = sum - value_return[0][i][0]
    writer[-1].close()

def main():

    draw_value()
    draw_estimate()
    draw_grad()

if __name__ == '__main__':
    main()