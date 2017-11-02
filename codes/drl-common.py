import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import math
import time
#_________________________________________________________create data___________________________________#
	##________________________________________________generate financial prices series________________________#
def gen_financial_data(size=7000):
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
#_______________________________________________________create data___________________________________#



#_______________________________________________________define loop framework___________________________________#
#hidden_layer2_nodes=30
hidden_layer3_nodes=20
hidden_layer4_nodes=20
nn_range=[15,20,25,30]
stack_range=[5,6,7,8]
state_range=[45,50,55,60]
learning_rate_range=[0.005,0.01,0.05,0.1]
num_experiments=len(nn_range)*len(stack_range)*len(state_range)
record=[]
record.append(['stack','state_size','hidden_layers','learning_rate','transaction_cost','total_profits','buy_and_hold'])
q=0
tf.set_random_seed(q)
p=gen_financial_data()#p contains all prices series
theta=0.01#transaction cost percentage
#________________________________________________________define loop framework___________________________________#


#__________________________________________________________define network___________________________________________________________#	
for num_steps in stack_range:#time stack:						
	action_size=1 #output size of action which used as another input of last node later
	for state_size in state_range: #using last 10 prices.and also present size so state_size is ac
		miu=100#investment scale
		for hidden_layers in nn_range:#hidden layers
			hidden_layer2_nodes=hidden_layers
			for learning_rate in learning_rate_range:
				
				q=q+1
				k=str(q)
				#each cell has a output of a tanh and thus an action
				#each cellis inputs include the last action(action_size) and the current state representation(state_size)
				
				#define place holder all tensors
				p_pr0=tf.placeholder(tf.float32,[action_size,num_steps],name='laged_prices%s'%q)
				last_p=tf.placeholder(tf.float32,[1])
				second_last_p=tf.placeholder(tf.float32,[1])
				p_pr1=tf.placeholder(tf.float32,[action_size,num_steps],name='prices%s'%q)
				with tf.name_scope('inputs%s'%q):
					x=tf.placeholder(tf.float32, [state_size,num_steps], name='input_placeholder%s'%q)
				#y=tf.placeholder(tf.int32, [action_size,num_steps], name='onput_placeholder')
				#each placeholde is supposed to hace a be num_steps long. in each training, the data will load on for a time window of num_steps. it is time stack.

				# define initial action
				init_action=tf.expand_dims(tf.zeros([action_size]),axis=1)#shape([1,1])

				rnn_inputs=tf.unstack(x,axis=1)

				action=init_action#shape([1,1])

				rnn_outputs=[]
				constant_last_actions=[]#record last action
				#_________________________________________define the network weights and thresholds + RECURRENT structure___________________#
				with tf.variable_scope(k) as scope:
						W1=tf.get_variable('W1',[hidden_layers+action_size,action_size])#hidden layers outputs + action of last step-->next action
						b1=tf.get_variable('b1',[action_size],initializer=tf.constant_initializer(0.0))#next action inputs + treshold(s)
						#d_W1=tf.get_variable('d_W1',[hidden_layers,hidden_layer2_nodes],initializer=tf.constant_initializer(0.0))#hidden layer2 weights
						#d_b1=tf.get_variable('d_b1',[hidden_layer2_nodes],initializer=tf.constant_initializer(0.0))#hidden layer2 threshold
						W=tf.get_variable('W',[state_size,hidden_layers])#inputs-->hiddellayers
						b=tf.get_variable('b',[hidden_layers],initializer=tf.constant_initializer(0.0))#hidden layers inputs+hidden layers thresholds
		

				def action_choose(rnn_input,action,hidden_layers,action_size):
					with tf.variable_scope(k,reuse=True):
						W1=tf.get_variable('W1',[hidden_layers+action_size,action_size])# same as above
						b1=tf.get_variable('b1',[action_size],initializer=tf.constant_initializer(0.0))#
						#d_W1=tf.get_variable('d_W1',[hidden_layers,hidden_layer2_nodes],initializer=tf.constant_initializer(0.0))#hidden layer2 weights
						#d_b1=tf.get_variable('d_b1',[hidden_layer2_nodes],initializer=tf.constant_initializer(0.0))#hidden layer2 threshold
						W=tf.get_variable('W',[state_size,hidden_layers])
						b=tf.get_variable('b',[hidden_layers],initializer=tf.constant_initializer(0.0))
						hidden_output1=tf.tanh(tf.matmul(tf.expand_dims(rnn_input,0),W)+b)
						#hidden_output2=tf.tanh(tf.matmul(hidden_output1,d_W1)+d_b1)
						#hidden=tf.tanh(tf.matmul(tf.expand_dims(rnn_input,0),W)+b)# make rnn inputs shapr of [1,state_size]
						#cp_action=tf.expand_dims(action,0)
						#print(action)
						#print(hidden)
						final_hidden_input=tf.concat([hidden_output1,action],axis=1)#the input of last tanh node
						act=tf.tanh(tf.matmul(final_hidden_input,W1)+b1)#action
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
				final_action=rnn_outputs[0]
				constant_last_action=tf.squeeze(constant_last_actions)
				neg_revenue=miu*(((-1)*tf.multiply(tf.subtract(p_pr1,p_pr0),constant_last_actions))+theta*tf.abs(tf.subtract(rnn_outputs,constant_last_actions)))
				#neg_revenue=tf.reduce_sum(miu*((-1)*tf.multiply(tf.subtract(p_pr1,p_pr0),constant_last_actions)))
				constant_action=tf.reduce_mean(final_action)
				total_revenue=tf.reduce_sum(miu*((-1)*tf.multiply(tf.subtract(last_p,second_last_p),final_action)))
				#use constant_last_action to store data generate by the agent. 
				train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(neg_revenue)
				#merged=tf.summary.merge_all()
				#train_step=tf.train.AdagradOptimizer(learning_rate).minimize(neg_revenue)

				#________________________________________define network outputs_________________________________#
#__________________________________________________________define network___________________________________________________________#	


				all_profits=[]#record all the profits
				all_actions=[]#record all actions
				current_average_profits=[]
				#__________________________________________________________initialize network_________________________________________________________#
				with tf.Session() as sess:
					sess.run(tf.global_variables_initializer())
					training_state=[np.zeros([action_size])]
					#merged=tf.summary.merge_all()
					
					
				#__________________________________________________________initialize network_________________________________________________________#


				#____________________________________________first round of running with random weights_______________________________________________#
					##calculate pr0 and pr1.
					for i,(pr0,pr1,X,Last_p,Second_last_p) in enumerate(gen_batch(num_steps,state_size,p)):
						Pr0,Pr1,rnn_out,profits,profit,training_state,constant_action_,_=sess.run([p_pr0,p_pr1,rnn_outputs,neg_revenue,total_revenue,final_action,constant_action,train_step],feed_dict ={x:X,init_action:training_state,p_pr0:pr0,p_pr1:pr1,last_p:Last_p,second_last_p:Second_last_p})
						#summary=sess.run(merged,feed_dict ={x:X,init_action:training_state,p_pr0:pr0,p_pr1:pr1,last_p:Last_p,second_last_p:Second_last_p})
						all_profits.append(-1*profit)
						current_average_profits.append(np.sum(all_profits))
						#print(profit)
						all_actions.append(constant_action_)
						#writer.add_summary(summary)
						#writer.flush()
					#writer=tf.summary.FileWriter('/home/yidi/simple-drl/codes',sess.graph)					
					#writer.close()
					#writer=tf.summary.FileWriter("/home/yidi/simple-drl/codes",sess.graph)
					#sess.close()
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
				#record[q,]=(str(float('%0.1f' % num_steps)),str(float('%0.1f' % state_size)),str(float('%0.1f' % hidden_layers)),str(float('%0.4f' % learning_rate)),str(float('%0.2f' % theta)),str(float('%0.2f' % np.sum(all_profits))),str(float('%0.2f' % float((p[-1]-p[0])*100))))
				record.append([num_steps,state_size,hidden_layers,learning_rate,theta,np.sum(all_profits),(p[-1]-p[0])*100])
				plt.close('all')
f=open('/home/yidi/rnn/experiments/record.csv','w')
for line in record:
	f.write(str(line))
	f.write('\n')
	
#____________________________________________first round of running with random weights_______________________________________________#

 




























