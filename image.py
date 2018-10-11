
import tensorflow as tf
import numpy as np
import cv2

SIZE_FRAME = 84

class imageGrabber:
    
    ##Init function, create subscriber and required vars.
    def __init__(self):
       
        self.state = []
        self.observation = []
        self.new_state = []

    def get_state(self,observation):
        self.preprocesses_cv2_v2(observation)
        #TODO change to numpy append
        self.new_state = np.stack((self.observation,self.state[0],self.state[1],self.state[2]))
        #self.state = np.reshape(self.state,(1,SIZE_FRAME,SIZE_FRAME,4)) 
        return self.new_state

    def setInitState(self,observation):
        self.preprocesses_cv2_v2(observation)
        self.state = np.stack((self.observation, self.observation, self.observation, self.observation), axis = 0)
        #cv2.imwrite('1.png',self.currentState[0])
        
        return self.state

    def update(self):
        self.state = self.new_state
        return self.state

    def cv2_pre(self,observation):
        # I grey it, cut the score and then I resize
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        observation = observation[40:200,0:160].copy()
        self.observation = cv2.resize(observation, (84, 84))
        #self.observation = np.resize(observation,(84*84))

    def preprocesses_cv2_v2(self,observation):
        # In here I grey out the image and then resize it directly as they used in the paper
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        self.observation = cv2.resize(observation, (84, 84))
   
    def preprocesses_tf(self,observation):
       
        output = tf.image.rgb_to_grayscale(observation)
        output = tf.image.crop_to_bounding_box(output, 34, 0, 160, 160)
        output = tf.image.resize_images(output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        self.observation = tf.squeeze(output)

        return self.observation

    def reset(self):
        self.state = []

    

    
