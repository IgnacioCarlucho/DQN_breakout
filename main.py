import tensorflow as tf
import numpy as np
from replay_buffer_v1 import ReplayBuffer
from q_network import Network
import gym
import wrappers as wp
from collections import deque
import argparse
from utils import str2bool

np.set_printoptions(threshold=np.nan)
# Base learning rate 




parser = argparse.ArgumentParser('DQN')
parser.add_argument('--gpu', type=str, choices=['gpu', 'cpu'], default='gpu')
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--sim', type=int, default=150)
parser.add_argument('--epsilon', type=float, default=1.)
parser.add_argument("--train", type=str2bool, nargs='?', const=True, default=True, help="Activate train mode.")
parser.add_argument("--save", type=str2bool, nargs='?', const=True, default=True, help="Activate save models")
parser.add_argument("--load", type=str2bool, nargs='?', const=True, default=False, help="Activate save models")
parser.add_argument('--epsilon_decay', type=float, default=0.0002)
args = parser.parse_args()

LEARNING_RATE = 0.0001
RANDOM_SEED = 1234
N_ACTIONS = 4
SIZE_FRAME = 84

DEVICE ='/' + args.gpu + ':0' # '/cpu:0'



def trainer(MINIBATCH_SIZE=32, GAMMA = 0.99,load=True ,save=True, epsilon=1.0, min_epsilon=0.1, BUFFER_SIZE=500000, train_indicator=True, render = True):
    with tf.Session() as sess:

        # configuring the random processes
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)
        # set evironment
        # robot = gym_environment('FrozenLakeNonskid4x4-v3', False, False, False) 
        # breakout
        # env = gym.make('BreakoutDeterministic-v4')
        env = wp.wrap_dqn(gym.make('BreakoutDeterministic-v4'))
        # Pong-v0
        # env= wp.wrap_dqn(gym.make('PongDeterministic-v4'))
        agent = Network(sess,SIZE_FRAME,N_ACTIONS,LEARNING_RATE,DEVICE)
        
        
        
        # TENSORFLOW init seession
        sess.run(tf.global_variables_initializer())
               
        # Initialize target network weights
        agent.update_target_network()
        # Initialize replay memory
        replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)
        replay_buffer.load()
        print('buffer size is now',replay_buffer.count)
        # this is for loading the net  
        if load:
            try:
                agent.recover()
                print('********************************')
                print('models restored succesfully')
                print('********************************')
            except tf.errors.NotFoundError:
                print('********************************')
                print('Failed to restore models')
                print('********************************')
        
       
        
        total_frames_counter = 0 
        frames_number = 0
        frames_to_save = 0
        while total_frames_counter < 10000000:
            
            if frames_to_save > 10000:
                agent.save()
                frames_to_save = 0

            if frames_number > 10000: 
                agent.update_target_network()
                frames_number = 0
                print('update_target_network')
                # agent.save()
                # replay_buffer.save()
            
            state = env.reset()
            q0 = np.zeros(N_ACTIONS)
            ep_reward = 0.
            done = False
            step = 0
            total_loss = deque()
            loss = 0.
            while not done:
               
                frames_number = frames_number + 1
                frames_to_save = frames_to_save + 1 
                total_frames_counter = total_frames_counter + 1
                if total_frames_counter > 20000:
                    epsilon -= 0.00000085
                    epsilon = np.maximum(min_epsilon,epsilon)
                    train_indicator = True
                else:
                    train_indicator = False #True
                    
                
                # for visualization
                # numpy_horizontal = np.hstack((np.array(state)[:,:,0], np.array(state)[:,:,1], np.array(state)[:,:,2],np.array(state)[:,:,3]))
                # cv2.imshow('image', numpy_horizontal)
                # cv2.waitKey(1)
                # time.sleep(0.05)
                
                # 1. get action with e greedy
                if np.random.random_sample() < epsilon:
                    #Explore!
                    action = np.random.randint(0,N_ACTIONS)
                else:
                    # Just stick to what you know bro
                    q0, X = agent.predict(np.reshape(np.array(state).astype(np.uint8),[-1,SIZE_FRAME,SIZE_FRAME,4])) 
                    action = np.argmax(q0)
                
                next_state, reward, done, info = env.step(action)#env.step(action)
                # env.render()
                # state = observation

                
                if train_indicator:
                    # Keep adding experience to the memory until
                    # there are at least minibatch size samples
                    if replay_buffer.size() > MINIBATCH_SIZE:

                        # 4. sample random minibatch of transitions: 
                        s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(MINIBATCH_SIZE)
                        
                        q_eval = agent.predict_target(s2_batch)
                        q_target = np.zeros(MINIBATCH_SIZE)

                        for k in range(MINIBATCH_SIZE):
                            if t_batch[k]:
                                q_target[k] = r_batch[k]
                            else:
                                q_target[k] = r_batch[k] + GAMMA * np.max(q_eval[k])
                        
                        #5.3 Train agent! 
                        loss, _ = agent.train(np.reshape(a_batch,(MINIBATCH_SIZE,1)),np.reshape(q_target,(MINIBATCH_SIZE,1)), s_batch )
                        # in case you want to understand the innner workings of this
                        # target_final, q_acted, delta, loss, optimize = agent.train_v2(np.reshape(a_batch,(MINIBATCH_SIZE,1)),np.reshape(q_target,(MINIBATCH_SIZE,1)), s_batch )
                        # print('target_final', target_final, 'q_acted', q_acted, 'delta', delta, 'loss', loss)
                
                # 3. Save in replay buffer:
                replay_buffer.add(state,action,reward,done,next_state) 
                state = next_state
                ep_reward = ep_reward + reward
                step +=1
                total_loss.append(loss)
            
            print('th',total_frames_counter+1,'Step', step,'Reward:',ep_reward,'epsilon', round(epsilon,3), np.mean(total_loss))

            # print('the reward at the end of the episode,', reward)
            
                        

        print('*************************')
        print('now we save the model')
        agent.save()
        #replay_buffer.save()
        print('model saved succesfuly')
        print('*************************')
        
        


if __name__ == '__main__':
    trainer(epsilon=args.epsilon , train_indicator = args.train, load=args.load) 
