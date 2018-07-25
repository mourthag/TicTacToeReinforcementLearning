import TicTacToeRL as ttt
import ExpBuffer
import QNet
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import matplotlib.pyplot as plt


#Update Target Netw with primary values
def updateTargetGraph(tfVars, tau):
    totalVars = len(tfVars)
    opHolder = []
    for idx,var in enumerate(tfVars[0:totalVars//2]):
        opHolder.append(tfVars[idx+totalVars//2].assign((var.value()*tau) + ((1 - tau) * tfVars[idx+totalVars//2].value())))
    return opHolder

def updateTarget(opHolder, sess):
    for op in opHolder:
        sess.run(op)

learning_rate = 0.00025
batch_size = 64 #Number of Experiences per training step
update_frequency = 4 #Perform training every n steps
y = .99 #QValue Discount
startE = 1 #Starting chance for random actions
endE = 0.01 #Final chance for random actions
annealing_steps = 70000  #goto endE within n steps
num_episodes = 1000000 #number of games played
num_random_steps = 50000 
load_model = True
path = "./dqn" #saving path
h_size = 256 #number of hidden neurons
tau = 0.001 #factor for target net updates from primary net

#GO!
tf.reset_default_graph()
mainQn = QNet.QNetwork(h_size=h_size, a_size=ttt.action_size, lr=learning_rate)
targetQn = QNet.QNetwork(h_size=h_size, a_size=ttt.action_size, lr=learning_rate)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
trainables = tf.trainable_variables()

targetOps = updateTargetGraph(trainables, tau)

trainingBuffer = ExpBuffer.experience_buffer(5000000)

e = startE
stepDrop = (startE - endE) / annealing_steps


jList = []
rList = []
total_steps = 0
wins = 0

if not os.path.exists(path):
    os.makedirs(path)

with tf.Session() as sess:
    sess.run(init)
    print("Quak")
    if load_model == True:
        print("Loading Model...")
        checkpoint = tf.train.get_checkpoint_state(path)
        saver.restore(sess, checkpoint.model_checkpoint_path)
    for i in range(num_episodes):
        episode_buffer = ExpBuffer.experience_buffer(1000000)
        s = ttt.emptyboard()
        d = False
        rAll = 0
        j=0
        while not d:
            j+=1

            if np.random.rand(1) < e or total_steps < num_random_steps:
                a = np.random.randint(0, 9)
            else:
                a = sess.run(mainQn.predict,feed_dict={mainQn.input:[s.flatten()]})[0]
            s1,r,d = ttt.setMachine(a,s, True)
            total_steps += 1
            episode_buffer.add(np.reshape(np.array([s.flatten(),a,r,s1.flatten(),d]),[1,5]))

            if(total_steps > num_random_steps):
                if e > endE:
                    e -= stepDrop

                if total_steps % update_frequency == 0:
                    trainBatch = trainingBuffer.sample(batch_size)
                    Q1 = sess.run(mainQn.predict,feed_dict={mainQn.input:np.vstack(trainBatch[:,3])})
                    Q2 = sess.run(targetQn.QOut, feed_dict={targetQn.input:np.vstack(trainBatch[:,3])})

                    end_multiplier = -(trainBatch[:,4] - 1)
                    doubleQ = Q2[range(batch_size),Q1]
                    targetQ = trainBatch[:,2] + (y*doubleQ * end_multiplier)
                    #Update the network with our target values.
                    _ = sess.run(mainQn.updateModel, \
                        feed_dict={mainQn.input:np.vstack(trainBatch[:,0]),mainQn.targetQ:targetQ, mainQn.actions:trainBatch[:,1]})
                    
                    updateTarget(targetOps, sess)
            rAll += r
            s = s1

            if d == True:
                break

        trainingBuffer.add(episode_buffer.buffer)
        jList.append(j)
        rList.append(rAll)

        if i % 5000 == 0 and i > 0:
            saver.save(sess,path+'/model-'+str(i)+'.ckpt')
            print("Saved Model")
            
        if i % 50000 == 0 and i > 0:
            rmat = np.resize(np.array(rList),[len(rList)//100,100])
            rMean = np.average(rmat,1)
            plt.plot(rMean)
            plt.show()

        if len(rList) % 100 == 0:
            print(i,np.mean(rList[-100:]), e)
    saver.save(sess,path+'/model-'+str(i)+'.ckpt')
print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")

rmat = np.resize(np.array(rList),[len(rList)//100,100])
rMean = np.average(rmat,1)
plt.plot(rMean)
plt.show()
