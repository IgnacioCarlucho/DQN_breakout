
import tensorflow as tf
import numpy as np
import time
from replay_buffer import ReplayBuffer
from q_network import Network
import gym
import wrappers as wp

DEVICE = '/gpu:0'

# Base learning rate 
LEARNING_RATE = 4*1e-4
# Soft target update param
RANDOM_SEED = 1234

N_ACTIONS = 3
SIZE_FRAME = 84

def trainer(epochs=1000, MINIBATCH_SIZE=32, GAMMA = 0.99,save=1, save_image=1, epsilon=1.0, min_epsilon=0.1, BUFFER_SIZE=250000, train_indicator=True, render = True):
    with tf.Session() as sess:

        # configuring the random processes
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)
        # set evironment
        # robot = gym_environment('FrozenLakeNonskid4x4-v3', False, False, False) 
        #env = gym.make('BreakoutDeterministic-v4')
        env = wp.wrap_dqn(gym.make('BreakoutDeterministic-v4'))
        # Pong-v0
        #env= wp.wrap_dqn(gym.make('PongDeterministic-v4'))
        agent = Network(sess,SIZE_FRAME,N_ACTIONS,LEARNING_RATE,DEVICE)
        #IG=imageGrabber()
        
        
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
        
       
        algo = 0
        total_frames_counter = 0 
        frames_number = 0
        #for i in range(epochs):
        while total_frames_counter < 2000000:
            
            if (total_frames_counter%500 == 0) : 
                print('*************************')
                print('now we save the model')
                agent.save()
                #replay_buffer.save()
                print('model saved succesfuly')
                print('*************************')
                
            if frames_number > 5000: 
                 agent.update_target_network()
                 frames_number = 0
                 print('update_target_network')

         
            #IG.reset()
            state = env.reset()
            
            #obs_1 = state.__array__(dtype=np.int32)
            #obs_2 = np.array(state).astype(np.float32) / 255.0
            
            


            #print(state,state.size,state.shape, 'state')
            q0 = np.zeros(N_ACTIONS)
            ep_reward = 0.
            done = False
            step = 0
            
            
            
            while not done:
               

                frames_number = frames_number + 1
                total_frames_counter = total_frames_counter + 1
                if total_frames_counter > 20000:
                    epsilon -= 0.00001 # 0.000001
                    epsilon = np.maximum(min_epsilon,epsilon)
                    train_indicator = True
                else:
                    train_indicator = False
                    
                
        
                # 1. get action with e greedy
                
                if np.random.random_sample() < epsilon:
                    #Explore!
                    action = np.random.randint(0,N_ACTIONS)
                else:
                    # Just stick to what you know bro
                    #print(np.array(state).astype(np.float32) / 255.0)
                    #print(np.array(state).astype(np.float32))
                    #time.sleep(5)
                    q0 = agent.predict(np.reshape(np.array(state).astype(np.float32) / 255.0,(1,SIZE_FRAME,SIZE_FRAME,4)) ) 
                    action = np.argmax(q0)
                    #print(q0, action)        
                
                #print(env.unwrapped.ale.lives())
                
                next_state, reward, done, info = env.step(1+action)#env.step(action)
                # state = observation

                
                #next_state = np.array(observation).astype(np.float32) / 255.0
                #cv2.imshow('image', state[0])
                #cv2.waitKey(0)
                #print(state)
                # time.sleep(2)
                
                if train_indicator:
                    
                   # Keep adding experience to the memory until
                    # there are at least minibatch size samples
                    if replay_buffer.size() > MINIBATCH_SIZE:

                        
                        # 4. sample random minibatch of transitions: 
                        s_batch, a_batch, r_batch, t_batch, s2_batch, algo = replay_buffer.sample_batch(MINIBATCH_SIZE)

                        
                        q_eval = agent.predict_target(np.reshape(s2_batch,(-1,SIZE_FRAME,SIZE_FRAME,4)))
                        #print('q_eval',q_eval)
                        #q_target = np.zeros(MINIBATCH_SIZE)
                        q_target = q_eval.copy()
                        for k in range(MINIBATCH_SIZE):
                            if t_batch[k]:
                                #print(q_target[k])
                                q_target[k][a_batch[k]] = r_batch[k]
                            else:
                                # TODO check that we are grabbing the max of the triplet
                                q_target[k][a_batch[k]] = r_batch[k] + GAMMA * np.max(q_eval[k])
                        #5.3 Train agent! 
                        agent.train(np.reshape(a_batch,(MINIBATCH_SIZE,1)),np.reshape(q_target,(MINIBATCH_SIZE,N_ACTIONS)), np.reshape(s_batch,(-1,SIZE_FRAME,SIZE_FRAME,4)) )
                        
                        
                # Get next state and reward
                
                                

                # 3. Save in replay buffer:
                replay_buffer.add(state,action,reward,done,next_state) 
                
                # prepare for next state
                #state = IG.update() 
                state = next_state
                ep_reward = ep_reward + reward
                step +=1
                
                
                #end2 = time.time()
                #print(step, action, q0, round(epsilon,3), round(reward,3))#, round(loop_time,3), nseconds)#'epsilon',epsilon_to_print )
                   #print(end-start, end2 - start)
                 
            
            
            print('th',total_frames_counter+1,'Step', step,'Reward:',ep_reward,'epsilon', round(epsilon,3), algo)
            #print('the reward at the end of the episode,', reward)
          
                        

            

            #time.sleep(15)

        print('*************************')
        print('now we save the model')
        agent.save()
        #replay_buffer.save()
        print('model saved succesfuly')
        print('*************************')
        
        


if __name__ == '__main__':
    trainer(epochs=20000 ,save_image = False, epsilon= 1., train_indicator = True) # 0.8
