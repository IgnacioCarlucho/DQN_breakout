
import tensorflow as tf
import numpy as np
import time
from replay_buffer import ReplayBuffer
from q_network import Network
from image import imageGrabber
import gym

DEVICE = '/gpu:0'

# Base learning rate 
LEARNING_RATE = 0.001
# Soft target update param
TAU = 0.001
RANDOM_SEED = 11543521#1234
EXPLORE = 400000


N_ACTIONS = 4
SIZE_FRAME = 84

def trainer(epochs=1000, MINIBATCH_SIZE=16, GAMMA = 0.99,save=1, save_image=1, epsilon=1.0, min_epsilon=0.001, BUFFER_SIZE=200000, train_indicator=True, render = True):
    with tf.Session() as sess:

        # configuring the random processes
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)
        # set evironment
        env = gym.make('Breakout-v0')
          
        agent = Network(sess,SIZE_FRAME,N_ACTIONS,LEARNING_RATE,TAU,DEVICE)
        IG=imageGrabber()
        
        
        # TENSORFLOW init seession
        sess.run(tf.global_variables_initializer())
               
        # Initialize target network weights
        agent.update_target_network()
        # Initialize replay memory
        replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)
        replay_buffer.load()
        print('buffer size is now',replay_buffer.count)
        # this is for loading the net  
        if save:
            try:
                agent.recover()
                print('********************************')
                print('models restored succesfully')
                print('********************************')
            except:
                print('********************************')
                print('Failed to restore models')
                print('********************************')
        
       
        


        for i in range(epochs):

            if (i%500 == 0) and (i != 0): 
                print('*************************')
                print('now we save the model')
                agent.save()
                replay_buffer.save()
                agent.update_target_network()
                print('model saved succesfuly')
                print('*************************')

         
            IG.reset()
            observation = env.reset()
            state = IG.setInitState(observation) 
            print(state,state.size,state.shape, 'state')
            q0 = np.zeros(4)
            ep_reward = 0.
            done = False
            step = 0
            counter = 0 # this counter is for the reward
            loop_time  = 0.

            while not done:
                
                
                
                epsilon -= (epsilon/EXPLORE)
                epsilon = np.maximum(min_epsilon,epsilon)
                
        
                # 1. get action with e greedy
                
                if np.random.random_sample() < epsilon:
                    #Explore!
                    action = np.random.randint(0,3)
                else:
                    # Just stick to what you know bro
                    q0 = agent.predict(np.reshape(state,(1,SIZE_FRAME,SIZE_FRAME,4)) ) 
                    action = np.argmax(q0)        
            	
                observation, reward, done, info = env.step(action)
                #if render: env.render()
                next_state = IG.get_state(observation)
               	
                
                if train_indicator:
                    
                   # Keep adding experience to the memory until
                    # there are at least minibatch size samples
                    if replay_buffer.size() > MINIBATCH_SIZE:

                        # 4. sample random minibatch of transitions: 
                        s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(MINIBATCH_SIZE)

                        
                        q_eval = agent.predict_target(np.reshape(s2_batch,(-1,SIZE_FRAME,SIZE_FRAME,4)))

                        q_target = q_eval.copy()
                        for k in range(MINIBATCH_SIZE):
                            if t_batch[k]:
                                q_target[k][a_batch[k]] = r_batch[k]
                            else:
                                # TODO check that we are grabbing the max of the triplet
                                q_target[k][a_batch[k]] = r_batch[k] + GAMMA * np.max(q_eval[k])

                       
                        #5.3 Train agent! 
                        #train(inputs, prediction)
                        agent.train(np.reshape(s_batch,(-1,SIZE_FRAME,SIZE_FRAME,4)),np.reshape(q_target,(MINIBATCH_SIZE, 4)) )
                        
                        
                        
                # Get next state and reward
                
                                

                # 3. Save in replay buffer:
                replay_buffer.add(state,action,reward,done,next_state) 
                
                # prepare for next state
                state = IG.update() 

                ep_reward = ep_reward + reward
                step +=1
                
                
                #end2 = time.time()
                #print(step, action, q0, round(epsilon,3), round(reward,3))#, round(loop_time,3), nseconds)#'epsilon',epsilon_to_print )
               	#print(end-start, end2 - start)
                 
            
            print('reseting noise')
            print('th',i+1,'Step', step,'Reward:',ep_reward,'epsilon', epsilon )
            print('the reward at the end of the episode,', reward)
          
                        

            

            #time.sleep(15)

        print('*************************')
        print('now we save the model')
        agent.save()
        replay_buffer.save()
        print('model saved succesfuly')
        print('*************************')
        
        


if __name__ == '__main__':
    trainer(epochs=1000 ,save_image = False, epsilon= .5, train_indicator = True)
