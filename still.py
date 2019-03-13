IS_RENDER = 1
IS_TRAIN = 1
IS_START = 0

import matplotlib.pyplot as plt 
import tensorflow as tf
import numpy as np
from tqdm import trange
from mujoco_py import load_model_from_xml, load_model_from_path, MjSim, MjViewer
import math
import os






np.random.seed(1)
tf.set_random_seed(1)

MAX_EP_STEPS = 600
LR_A = 0.001 # learning rate for actor
LR_C = 0.002  # learning rate for critic
GAMMA = 0.99     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 120000
BATCH_SIZE = 64
TRAIN_REPEAT = 3


from mujoco_py.modder import TextureModder

class manipulator():

    def __init__(self):
        
        self.DoF = DoF = 6
        self.model = load_model_from_path("model-ur10/xyz.xml")
        self.sim = MjSim(self.model)
        
        self.sim.data.qpos[1] = -np.pi/2
      #  self.sim.data.qpos[2] = -np.pi/2         
        self.sim.data.qpos[3] = -np.pi/2
        self.sim.step()

        if IS_RENDER: self.viewer = MjViewer(self.sim)
        self.sim_state = self.sim.get_state()
        
        self.state_dim = 19
        self.action_dim = 5
        
    def reset(self):
        self.sim.set_state(self.sim_state)
       # self.qvel=np.zeros(5)
        
  
        self.rx=np.random.uniform(0.4,0.85)
        self.ry=np.random.uniform(0.5,0.9)
        self.rz=np.random.uniform(0.0,0.4)
        a=np.random.random()
        b=np.random.random()




     

        




        self.sim.data.qpos[6]=self.rx  
        self.sim.data.qpos[7]=self.ry 
        self.sim.data.qpos[8]=self.rz     



        self.sim.data.sensordata[0]=0
        self.sim.data.sensordata[1]=0
        self.sim.data.sensordata[2]=0.1273
        self.sim.data.sensordata[3]=0
        self.sim.data.sensordata[4]=0.220941
        self.sim.data.sensordata[5]=0.1273
        self.sim.data.sensordata[6]=0
        self.sim.data.sensordata[7]=0.049041
        self.sim.data.sensordata[8]=0.7393
        self.sim.data.sensordata[9]=0
        self.sim.data.sensordata[10]=0.049041
        self.sim.data.sensordata[11]=1.3116
        self.sim.data.sensordata[12]=0
        self.sim.data.sensordata[13]=0.163941
        self.sim.data.sensordata[14]=1.3116
        self.sim.data.sensordata[15]=0
        self.sim.data.sensordata[16]=0.245841
        self.sim.data.sensordata[17]=1.4273

        s = []

       # for i in range(4):
        #    s.append( self.sim.data.qvel[i] )  #real vel 
        #for i in range(4):
         #   s.append( self.sim.data.qpos[i] )

        s.append( self.rx)
        s.append( self.ry)
        s.append( self.rz)
        s.append( self.sim.data.sensordata[3])
        s.append( self.sim.data.sensordata[4])
        s.append( self.sim.data.sensordata[5])

        s.append( self.sim.data.sensordata[6])
        s.append( self.sim.data.sensordata[7])
        s.append( self.sim.data.sensordata[8])

        s.append( self.sim.data.sensordata[9])
        s.append( self.sim.data.sensordata[10])
        s.append( self.sim.data.sensordata[11])
        s.append( self.sim.data.sensordata[12])
        s.append( self.sim.data.sensordata[13])
        s.append( self.sim.data.sensordata[14])

        s.append( self.sim.data.sensordata[15])
        s.append( self.sim.data.sensordata[16])
        s.append( self.sim.data.sensordata[17])
        s.append(0)




      #  s.append( self.sim.data.sensordata[0])
       # s.append( self.sim.data.sensordata[1])
       # s.append( self.sim.data.sensordata[2])
       # s.append( self.sim.data.sensordata[3])
       # s.append( self.sim.data.sensordata[4])
       # s.append( self.sim.data.sensordata[5])
       # s.append( self.sim.data.sensordata[6])
       # s.append( self.sim.data.sensordata[7])
       # s.append( self.sim.data.sensordata[8])
       # s.append( self.sim.data.sensordata[9])
       # s.append( self.sim.data.sensordata[10])
       # s.append( self.sim.data.sensordata[11])
       # s.append( self.sim.data.sensordata[12])
       # s.append( self.sim.data.sensordata[13])
       # s.append( self.sim.data.sensordata[14])
       # s.append( self.sim.data.sensordata[15])
       # s.append( self.sim.data.sensordata[16])
       # s.append( self.sim.data.sensordata[17])



        s = np.array(s)
        
        return s
    def step(self, a):
                
        for i in range(5):
            self.sim.data.qpos[i]+= a[i]
       #    self.sim.data.qvel[i]= self.qvel[i]
    #    self.sim.data.qvel[3]=0
       # self.sim.data.qvel[4]=0
        self.sim.data.qvel[5]=0
     #   self.sim.data.qpos[3]=-np.pi/2
       # self.sim.data.qpos[4]=0
        self.sim.data.qpos[5]=0 
   
            # The limitation of velocity is important
        if self.sim.data.qvel[i]>1.0:
            self.sim.data.qvel[i] = 1.0
        if self.sim.data.qvel[i]<-1.0:
            self.sim.data.qvel[i] = -1.0
                
    #        self.sim.data.qvel[i] = self.qvel[i] 
                      
        self.sim.data.qpos[6]=self.rx  
        self.sim.data.qpos[7]=self.ry 
        self.sim.data.qpos[8]=self.rz  
        
        self.sim.step()
        if IS_RENDER: self.viewer.render()
        
        dis = np.linalg.norm(self.sim.data.sensordata[15:18]-[self.rx, self.ry,self.rz])

        d=0

        r = -dis*0.1
        if dis < 0.3:
            r+=0.05
        if dis < 0.2:
            r+=0.1
        if dis < 0.1:
            r+=0.2
            d=1
        if dis < 0.05:
            r+=0.5
            d=1
        if dis < 0.02:
            r+=0.5
            d=1
        if dis < 0.01:
            r+=0.5
            d=1
        
        q=[]
        q.append(self.sim.data.qpos[0])
        q.append(self.sim.data.qpos[1])
        q.append(self.sim.data.qpos[2])
        q.append(self.sim.data.qpos[3])
        q.append(self.sim.data.qpos[4])
        
        s = []

       # for i in range(4):
        #    s.append( self.sim.data.qvel[i] )  #real vel 
       # for i in range(4):
        #    s.append( self.sim.data.qpos[i] )
        s.append( self.rx)
        s.append( self.ry)
        s.append( self.rz)
        s.append( self.sim.data.sensordata[3])
        s.append( self.sim.data.sensordata[4])
        s.append( self.sim.data.sensordata[5])
        s.append( self.sim.data.sensordata[6])
        s.append( self.sim.data.sensordata[7])
        s.append( self.sim.data.sensordata[8])

   
        s.append( self.sim.data.sensordata[9])
        s.append( self.sim.data.sensordata[10])
        s.append( self.sim.data.sensordata[11])
        s.append( self.sim.data.sensordata[12])
        s.append( self.sim.data.sensordata[13])
        s.append( self.sim.data.sensordata[14])

        s.append( self.sim.data.sensordata[15])
        s.append( self.sim.data.sensordata[16])
        s.append( self.sim.data.sensordata[17])
        s.append(d)





       # s.append( self.sim.data.sensordata[0])
       # s.append( self.sim.data.sensordata[1])
       # s.append( self.sim.data.sensordata[2])
       # s.append( self.sim.data.sensordata[3])
       # s.append( self.sim.data.sensordata[4])
       # s.append( self.sim.data.sensordata[5])
       # s.append( self.sim.data.sensordata[6])
       # s.append( self.sim.data.sensordata[7])
       # s.append( self.sim.data.sensordata[8])
       # s.append( self.sim.data.sensordata[9])
       # s.append( self.sim.data.sensordata[10])
       # s.append( self.sim.data.sensordata[11])
       # s.append( self.sim.data.sensordata[12])
       # s.append( self.sim.data.sensordata[13])
       # s.append( self.sim.data.sensordata[14])
       #  s.append( self.sim.data.sensordata[15])
       # s.append( self.sim.data.sensordata[16])
       # s.append( self.sim.data.sensordata[17])


        s = np.array(s)

        info = dis
        return s, r, d, info,q

env=manipulator()

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound, env):
        self.env = env
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()
                
        self.var = 3.0      
            
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self.build_a_nn(self.S, scope='eval', trainable=True)
            a_ = self.build_a_nn(self.S_, scope='target', trainable=True)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self.build_c_nn(self.S, self.a, scope='eval', trainable=True)
            q_ = self.build_c_nn(self.S_, a_, scope='target', trainable=True)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(at, (1-TAU)*at+TAU*ae), tf.assign(ct, (1-TAU)*ct+TAU*ce)]
            for at, ae, ct, ce in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=(self.R + GAMMA * q_), predictions=q)
       # reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-4), tf.trainable_variables())

        
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, name="adam-ink", var_list = self.ce_params)

        a_loss = - tf.reduce_mean(q) # maximize the q
        #reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-4), tf.trainable_variables())

        
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

     #   tf.summary.FileWriter("logs/", self.sess.graph)
       # self.sess.run(tf.global_variables_initializer())
        
        self.saver = tf.train.Saver()
        
        if IS_START:
            self.sess.run(tf.global_variables_initializer())
        else:
            self.saver.restore(self.sess, "save/12.ckpt")
  

    def choose_action(self, s):
        a = self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]
        
        a= np.clip(a,-0.3,0.3)      
 
        
        
        return a

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
        self.sess.run(self.atrain, {self.S: bs, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
    
        trans = np.hstack((s,a,[r],s_))
        
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = trans
        self.pointer += 1       
        if self.pointer > MEMORY_CAPACITY:
     
            self.learn()
    def build_a_nn(self, s, scope, trainable):
        # Actor DPG
        with tf.variable_scope(scope):
           # s_norm = tf.contrib.layers.layer_norm(s,center=True, scale=True, begin_norm_axis=0 )
            l1 = tf.layers.dense(s, 520, activation = tf.nn.tanh, name = 'l1', trainable = trainable)
            l1 = tf.contrib.layers.layer_norm(l1,center=True, scale=True)
            l2 = tf.layers.dense(l1,520,activation = tf.nn.tanh, name = 'l2', trainable = trainable)
            l2 = tf.contrib.layers.layer_norm(l2,center=True, scale=True)            
          #  k_init = tf.random_uniform_initializer(minval=-0.002, maxval=0.002)
            a = tf.layers.dense(l2, self.a_dim, activation = tf.nn.tanh, name = 'a', trainable = trainable)    
            return tf.multiply(a, self.a_bound, name = "scaled_a")  
    def build_c_nn(self, s, a, scope, trainable):
        # Critic Q-leaning
        with tf.variable_scope(scope):
            n_l1 = 600
              
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable = trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable = trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable = trainable)
            
           # s_norm = tf.contrib.layers.layer_norm(s,center=True, scale=True, begin_norm_axis=0 )
           # a_norm = tf.contrib.layers.layer_norm(a,center=True, scale=True, begin_norm_axis=0 )
            
            linear_output = tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1      
            l1 = tf.nn.relu( linear_output )
            l1 = tf.contrib.layers.layer_norm(l1, center=True, scale=True)
            l2 = tf.layers.dense(l1, 600, activation=tf.nn.relu,trainable = trainable)
            l2 = tf.contrib.layers.layer_norm(l2, center=True, scale=True)
           # k_init = tf.random_uniform_initializer(minval=-0.002, maxval=0.002)
            q = tf.layers.dense(l2, 1, trainable = trainable)
            return q
  
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = 0.005
MAX_EPISODES = 2001

ddpg = DDPG(a_dim, s_dim, a_bound, env)
r_save = []
d_save = []
o_save = []
q_save = []
'''
while True:
    s=env.reset()
    while True:	
        a = [0,0,0,0,0]
       # a = ddpg.choose_action(s)
        s_, r, d, info = env.step(a)
       # print(d)
        s=s_

'''
k=0
if IS_TRAIN:
    for i in trange(100000):
        
        s = env.reset() 
        dsave=[]  
        q_save = []   
        ep_reward = 0
        for j in range(MAX_EP_STEPS):
        
            # Add exploration noise
            a = ddpg.choose_action(s)
          #  print(a)
                
            s_, r, done, info,q = env.step(a)
            q_save.append(q)
            

            ddpg.store_transition(s, a, r, s_)

            s = s_
    
            ep_reward += r
            dsave.append(info)

       # print(len(q_save))
       # print(q_save)

        if min(dsave)<0.05:
            k+=1

        d_save.append(min(dsave))
        r_save.append(ep_reward)
        o_save.append(dsave[-1])
        
        if i % 100 == 0:
            ddpg.saver.save(ddpg.sess, 'save/1.ckpt')

        print(i, ':',ep_reward,min(dsave))
    
    print(k)

    xypoints = r_save[:]

    plt.plot(np.array(xypoints[:]).reshape(-1), 'g--', label='reward')
    plt.title('Reward')
    plt.xlabel('Learning Cycles')
    plt.ylabel('Reward')
    plt.legend(loc='upper left')
    plt.savefig('v1.jpg')
    plt.show()

    xypoints = d_save[:]

    plt.plot(np.array(xypoints[:]).reshape(-1), 'g--', label='dis')
    plt.title('min_dis')
    plt.xlabel('Learning Cycles')
    plt.ylabel('dis')
    plt.legend(loc='upper left')
    plt.savefig('v2.jpg')
    plt.show()


    xypoints = o_save[:]

    plt.plot(np.array(xypoints[:]).reshape(-1), 'g--', label='dis')
    plt.title('last_dis')
    plt.xlabel('Learning Cycles')
    plt.ylabel('dis')
    plt.legend(loc='upper left')
    plt.savefig('v3.jpg')
    plt.show()



















else:
    
    EPISODES = 1
    ddpg.saver.restore(ddpg.sess, 'model/v8.ckpt')

    s = env.reset()
       # print(s)
    _save = []
    ep_reward = 0
    for i in range(600):
            
        a = ddpg.choose_action(s)
            
        s_, r, done, info = env.step(a)
        s=s_
        ep_reward += r 
        if info<0.05:
            break 

        
        dis_save.append(info[0])


    print(ep_reward)
    xypoints = dis_save

    plt.plot(np.array(xypoints).reshape(-1), 'r--', label='distance')
    plt.title('Distance')
    plt.xlabel('Time Steps')
    plt.ylabel('Distance')
    plt.legend(loc='upper right')
    plt.show()
    print(min(dis_save))
    print(dis_save[-1]) 

