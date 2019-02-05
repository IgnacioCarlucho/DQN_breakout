import gym
import numpy as np 
import cv2
import tensorflow as tf
import time 
import wrappers as wp

tf.enable_eager_execution()

env = gym.make('Breakout-v0')
state = env.reset()
# state.shape = (210, 160, 3)
start = time.time()
state_grey = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
state_grey = state_grey[40:200,0:160].copy()
state_grey = cv2.resize(state_grey, (84, 84))

#state_grey = np.reshape(state_grey, (84*84))
state = np.stack((state_grey, state_grey, state_grey,state_grey), axis = 0)
end = time.time()
print(end-start)

obse = np.resize(state_grey,(84*84))
stacked = np.stack((state_grey,state_grey, state_grey, state_grey), axis = 0)
print(stacked, stacked.shape)
#print(state,state.shape)
#cv2.imshow('image', state_grey)
#cv2.waitKey(0)

'''

'''
output = tf.image.rgb_to_grayscale(state)

output = tf.image.crop_to_bounding_box(output, 34, 0, 160, 160)
output = tf.image.resize_images(output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
output_2 = tf.squeeze(output)
print(output_2.numpy(), output.numpy().shape)

cv2.imshow('image', output.numpy())

cv2.waitKey(0)
'''

env = wp.wrap_dqn(gym.make('BreakoutDeterministic-v4'))
state = env.reset()
done = False
for _ in range(50):
	if not done: 
		next_state, reward, done, info = env.step(1)
		print('state', next_state, 'done', done)
		time.sleep(0.1)