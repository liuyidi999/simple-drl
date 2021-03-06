import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import math

import time
#________________________________________________________________create data___________________________________#

##________________________________________________generate financial prices series________________________#
def gen_financial_data(size=700):
	alpha=0.3
	k=0
	X=np.zeros(size)
	p=np.zeros(size)
	beta=np.zeros(size)
	p[0]=1
	beta[0]=1
	np.random.seed(seed=7)
	for i in range(size-1):
		beta[i+1]=alpha*beta[i]+np.random.randn()
		p[i+1]=p[i]+beta[i]+k*np.random.randn()
	R=max(p)-min(p)		
	z=[math.exp(pr/R) for pr in p]
	return np.array(z)
##________________________________________________generate financial prices series________________________#

##________________________________________________generate training batches________________________#

def gen_batch(num_steps,state_size,p):
	epoch_size=(len(p)-1-state_size-num_steps)#epochsize means how many times trade or training happens, we make first 100 data points useless for the sake of make space for training time stacks and state length.
	pr1=np.zeros([1,num_steps],dtype=np.float32)#pr1 and pr0 means prices series of realtime and one step lag.pr0 is lagged.
	pr0=np.zeros([1,num_steps],dtype=np.float32)
	X=np.zeros([state_size,num_steps])
	for i in range(epoch_size):
		#data_pr1=p[i*epoch_size+1+state_size:i*epoch_size+1+num_steps+state_size]
		pr1=[p[1+state_size+i:1+state_size+i+num_steps]]
		last_p=[p[1+i+num_steps]]
		pr0=[p[i+state_size:i+state_size+num_steps]]
		second_last_p=[p[i+num_steps]]
		for j in range(num_steps):
			X[:,j]=p[1+j+i:1+state_size+j+i]
		#data_pr0=p[i*epoch_size+state_size:i*epoch_size+num_steps+state_size]
		#data_X=p[i*epoch_size+1:i*epoch_size+state_size+1]
		yield (np.array(pr0),np.array(pr1),np.array(X),np.array(last_p),np.array(second_last_p))
##________________________________________________generate training batches________________________#
#plt.plot(gen_financial_data())
#plt.show()

#with tf.variable_scope('rnn_cell'):
#	W=tf.get_variable('W',[state_size,state_size])
#	b=tf.get_variable('b',[state_size],initializer=tf.constant_initializer(0.0))
#action: the last action
#state_size: state representation size.
#________________________________________________________________create data___________________________________#





















record=[]
k=1
tf.set_random_seed(6)	

p=gen_financial_data()#p contains all prices series


	
#__________________________________________________________define network___________________________________________________________#	
num_steps=10#time stack						
action_size=1 #output size of action which used as another input of last node later
state_size=50 #using last 10 prices.and also present size so state_size is ac
miu=100#investment scale
hidden_layers=15#hidden layers
learning_rate=0.01
theta=0#transaction cost percentage
#each cell has a output of a tanh and thus an action
#each cellis inputs include the last action(action_size) and the current state representation(state_size)


		

#define place holder all tensors
p_pr0=tf.placeholder(tf.float32,[action_size,num_steps])
last_p=tf.placeholder(tf.float32,[1])
second_last_p=tf.placeholder(tf.float32,[1])
p_pr1=tf.placeholder(tf.float32,[action_size,num_steps])
x=tf.placeholder(tf.float32, [state_size,num_steps], name='input_placeholder')
#y=tf.placeholder(tf.int32, [action_size,num_steps], name='onput_placeholder')
#each placeholde is supposed to hace a be num_steps long. in each training, the data will load on for a time window of num_steps. it is time stack.

# define initial action
init_action=tf.expand_dims(tf.zeros([action_size]),axis=1)#shape([1,1])

rnn_inputs=tf.unstack(x,axis=1)

action=init_action#shape([1,1])

rnn_outputs=[]
constant_last_actions=[]#record last action
#_________________________________________define the network weights and thresholds + RECURRENT structure___________________#
with tf.variable_scope('action_choose'):
		W1=tf.get_variable('W1',[hidden_layers+action_size,action_size])#hidden layers outputs + action of last step-->next action
		b1=tf.get_variable('b1',[action_size],initializer=tf.constant_initializer(0.0))#next action inputs + treshold(s)
		W=tf.get_variable('W',[state_size,hidden_layers])#inputs-->hiddellayers
		b=tf.get_variable('b',[hidden_layers],initializer=tf.constant_initializer(0.0))#hidden layers inputs+hidden layers thresholds


def action_choose(rnn_input,action,hidden_layers,action_size):
	with tf.variable_scope('action_choose',reuse=True):
		W1=tf.get_variable('W1',[hidden_layers+action_size,action_size])# same as above
		b1=tf.get_variable('b1',[action_size],initializer=tf.constant_initializer(0.0))#
		W=tf.get_variable('W',[state_size,hidden_layers])
		b=tf.get_variable('b',[hidden_layers],initializer=tf.constant_initializer(0.0))
		hidden=tf.tanh(tf.matmul(tf.expand_dims(rnn_input,0),W)+b)# make rnn inputs shapr of [1,state_size]
		#cp_action=tf.expand_dims(action,0)
		#print(action)
		#print(hidden)
		new_hidden_input=tf.concat([hidden,action],axis=1)
		act=tf.tanh(tf.matmul(new_hidden_input,W1)+b1)
	return act


for rnn_input in rnn_inputs:# unfold according to num_steps
	constant_last_actions.append(tf.squeeze(action))#record all action of last step
	action=action_choose(rnn_input,action,hidden_layers,action_size)#create network structure one by one
	rnn_outputs.append(action)
	

#_________________________________________define the network weights and thresholds + RECURRENT structure___________________#
#define  action_choose

#cell=tf.contrib.rnn.BasicRNNCell(hidden_layers)
#rnn_outputs,final_state=tf.contrib.rnn.static_rnn(cell,rnn_inputs,initial_state=)


#constant_last_actions.append(rnn_outputs[-2-num_steps:-2])#record last actions with one step lag
#revenue=[miu*((pr1-pr0)*(constant_last_action)-theta*(tf.abs(rnn_output-constant_last_action))) for pr0, pr1,rnn_output,constant_last_action in [p_pr0,p_pr1,rnn_outputs,constant_last_actions]]
#________________________________________define network outputs_________________________________#
final_action=rnn_outputs[-1]
constant_last_action=tf.squeeze(constant_last_actions)

#neg_revenue=tf.reduce_mean(miu*(tf.reduce_sum(tf.abs(p_pr1-p_pr0)*theta)-tf.matmul(p_pr0,constant_last_action)))
#neg_revenue=miu*(tf.square(rnn_outputs[-1]-rnn_outputs[-2])*theta-tf.matmul([last_p-second_last_p],rnn_outputs[-2]))
#neg_revenue=[tf.reduce_sum(tf.where(tf.greater(rnn_output,constant_last_action),miu*((rnn_output-constant_last_action)*theta-(pr1-pr0)*rnn_output),miu*((constant_last_action-rnn_output)*theta-(pr1-pr0)*rnn_output))) for rnn_output,constant_last_action,pr1,pr0 in zip(rnn_outputs, constant_last_actions, p_pr1, p_pr0) ]
#neg_revenue=[tf.reduce_sum(tf.where(tf.greater(rnn_outputs,constant_last_actions),miu*(tf.reduce_sum((rnn_outputs-constant_last_actions)*theta)-tf.matmul((pr1-pr0),rnn_outputs)),miu*(tf.reduce_sum((constant_last_actions-rnn_outputs)*theta)-tf.matmul((pr1-pr0),rnn_outputs))))]
#neg_revenue=miu*(tf.subtract(tf.abs(tf.subtract(rnn_outputs,constant_last_action))*theta,tf.multiply((p_pr1-p_pr0),constant_last_action)))
neg_revenue=miu*((-1)*tf.multiply(tf.subtract(p_pr1,p_pr0),constant_last_actions))
#neg_revenue=tf.reduce_mean(miu*((-1)*tf.multiply(tf.subtract(p_pr1,p_pr0),constant_last_action)))
constant_action=tf.reduce_mean(final_action)

#neg_revenue=miu*(tf.subtract(tf.abs(tf.subtract(rnn_outputs[-1],constant_last_action[-1])*theta),tf.multiply((p_pr1[-1]-p_pr0[-1]),constant_last_action[-1])))


total_revenue=tf.reduce_sum(miu*((-1)*tf.multiply(tf.subtract(last_p,second_last_p),final_action)))
#neg=tf.reduce_mean(neg_revenue)
#use constant_last_action to store data generate by the agent. 
train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(neg_revenue)
#train_step=tf.train.AdagradOptimizer(learning_rate).minimize(neg_revenue)

#________________________________________define network outputs_________________________________#

#__________________________________________________________define network___________________________________________________________#	


#all_actions=np.zeros(len(p)-1-state_size-100)
#all_profits=np.zeros(len(p)-1-state_size-100)
all_profits=[]
all_actions=[]
current_average_profits=[]
#__________________________________________________________initialize network_________________________________________________________#
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	training_state=[np.zeros([action_size])]
	#print(training_state)
	#=tf.expand_dims(ts,axis=0)	
#print(training_state)
#__________________________________________________________initialize network_________________________________________________________#
#____________________________________________first round of running with random weights_______________________________________________#
	##calculate pr0 and pr1.
	for i,(pr0,pr1,X,Last_p,Second_last_p) in enumerate(gen_batch(num_steps,state_size,p)):
		#print('action',training_state)
		#print(all_actions)
	##calculate pr0 and pr1.
		Pr0,Pr1,lastp,secondp,rnn_out,profits,profit,training_state,constant_action_,_=sess.run([p_pr0,p_pr1,last_p,second_last_p,rnn_outputs,neg_revenue,total_revenue,final_action,constant_action,train_step],feed_dict ={x:X,init_action:training_state,p_pr0:pr0,p_pr1:pr1,last_p:Last_p,second_last_p:Second_last_p})
		#training_state=tf.squeeze(training_state,1)
		#profit=tf.squeeze(profit,1)
		#print('profit',profit)
		#all_profits[i]=tf.squeeze(profit)
		#all_actions[i]=tf.squeeze(action)
		all_profits.append(-1*profit)
		current_average_profits.append(np.sum(all_profits))
		#print(profit)
		all_actions.append(constant_action_)
		
	sess.close()
time_needed=time.clock()
print(time_needed)
fig=plt.figure(figsize=(16,9))
p1=plt.subplot(3,1,1)
p2=plt.subplot(3,1,2)
p3=plt.subplot(3,1,3)
p1.plot(p)
p1.set_ylabel('prices series',fontsize=16)
p1.set_title('stack=%d, state size=%d, nn=%d, trasaction cost=%s, learning rate=%s,\n total profits=%s, buy and hold=%s, computing time=%s' % (num_steps,state_size,hidden_layers,str(float('%0.2f' % theta)),str(float('%0.4f' % learning_rate)),str(float('%0.2f' % np.sum(all_profits))),str(float('%0.2f' % float((p[-1]-p[0])*100))),str(float('%0.2f' % time_needed))),fontsize=16)
#p1.set_title(['stack=',num_steps,' state size=',state_size,' nn=',hidden_layers,' trasaction cost=',theta,' learning rate=',learning_rate,' total profits=',np.sum(all_profits),' buy and hold=', (p[-1]-p[0])*100, ' computing time=',time_needed])
#plt.plot(p)#
p2.plot(all_actions)
p2.set_ylabel('actions',fontsize=16)
p3.plot(current_average_profits)
p3.set_ylabel('total profits',fontsize=16)
p3.set_xlabel('time',fontsize=16)

#print(len(all_profits))
#print(len(all_actions))
print(np.sum(all_profits))
record1=np.sum(all_profits)
print((p[-1]-p[0])*100)
record2=(p[-1]-p[0])*100
#plt.show()
plt.savefig('/home/yidi/rnn/experiments/figure%s.svg'%k)
plt.savefig('/home/yidi/rnn/experiments/figure%s.png'%k)

#____________________________________________first round of running with random weights_______________________________________________#






























