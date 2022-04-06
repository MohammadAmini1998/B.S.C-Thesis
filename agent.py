import random
from collections import namedtuple,deque
from torch.optim import adam,RMSprop
from torch import nn
import torch
import numpy as np
from network import Network
import gc
from GPUtil import showUtilization as gpu_usage

class Agent():
    def __init__(self,max_memory,batch_size,
                 save_dir,eps_start,eps_end,
                 eps_decay,action_dim,state_dim,gamma,algorithm,
                 learning_rate,min_exp,sync_every,learn_every,save_every):
        self.max_memory=max_memory
        self.batch_size=batch_size
        self.memory=deque(maxlen=self.max_memory)
        self.save_dir=save_dir


        self.memory_initalizing_done=False




        #Epsilon:
        self.eps_start=eps_start
        self.eps_end=eps_end
        self.eps_decay=eps_decay
        self.exploration_rate=0


        self.action_dim=action_dim
        self.state_dim=state_dim


        self.gamma=gamma
        self.learning_rate=learning_rate

        self.min_exp=min_exp


        self.sync_every=sync_every


        self.learn_every=learn_every
        self.save_every=save_every

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.use_cuda = torch.cuda.is_available()


        self.net = Network(self.state_dim, self.action_dim).to(self.device)



        self.current_step=0

        self.algorithm=algorithm


        #optimizer and loss :
        #self.optimizer=adam.Adam(self.net.parameters(),lr=self.learning_rate,d)
        self.optimizer=adam.Adam(self.net.parameters(),lr=self.learning_rate)
        self.loss=nn.SmoothL1Loss()






    def select_action(self,state):
        rand = random.random()
        if rand < epsilon:
            action= random.randrange(self.number_of_actions)
        else:
            state = (torch.FloatTensor(np.float32(state)).unsqueeze(0)).to(self.device)
            q_value = self.forward(state)
            action= q_value.max(1)[1].data[0]
        self.current_step+=1
        self.exploration_rate=np.interp(self.current_step, [0, self.eps_decay], [self.eps_start, self.eps_end])
        return action


    def remember(self,state,next_state,reward,done,action):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.memory.append((state, action, reward, next_state, done))


    def sample(self):
        state, action, reward, next_state, done = zip(*random.sample(self.memory, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def compute_td_loss(batch_size):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(np.float32(state))
        next_state = torch.FloatTensor(np.float32(next_state))
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(done)

        q_values = current_model(state)
        next_q_values = current_model(next_state)
        next_q_state_values = target_model(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = reward + gamma * next_q_value * (1 - done)

        loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return los



    def sync_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())





    def learn(self):

        if self.current_step % self.sync_every==0:
            self.sync_target()


        if self.current_step<self.min_exp:
            return None


        if self.current_step % self.save_every==0:
            self.save()

        if self.current_step % self.learn_every != 0:
            return None

        if self.current_step>self.min_exp:
            loss = compute_td_loss(self.batch_size)
            return loss












    def save(self):
            save_path = self.save_dir / f"{self.algorithm}____{int(self.current_step) }____{self.exploration_rate}.chkpt"
            torch.save(
                dict(
                    model=self.net.state_dict(),
                    exploration_rate=self.exploration_rate
                ),
                save_path
            )
            print(f"Game saved to {save_path} at step {self.current_step}")
    def load(self, load_path):
        # if not load_path.isdir():
        #     raise ValueError(f"{load_path} does not exits")
            ckp = torch.load(load_path, map_location=('cuda' if self.use_cuda else 'cpu'))
            exploration_rate = ckp.get("exploration_rate")
            state_dict = ckp.get("model")
            print(f"Loading model at {load_path} with exploration rate {exploration_rate}")
            self.exploration_rate = exploration_rate
            self.net.load_state_dict(state_dict)




























