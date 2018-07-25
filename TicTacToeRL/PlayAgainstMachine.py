import tensorflow as tf
import QNet
import TicTacToeRL as ttt

path = "./dqn"

tf.reset_default_graph()
mainQn = QNet.QNetwork(h_size=512, a_size=ttt.action_size, lr=0.0003)
saver = tf.train.Saver()

with tf.Session() as sess:
    checkpoint = tf.train.get_checkpoint_state(path)
    saver.restore(sess, checkpoint.model_checkpoint_path)
    
    board = ttt.emptyboard()
    d= False;
    while not d:
        a = sess.run(mainQn.predict, feed_dict={mainQn.input:[board.flatten()]})[0]

        board, reward, d = ttt.setMachine(a, board, False)
        print(board)
        
        if d:
            if reward > 0:
                print("AI wins!")
                break
            if reward < 0:
                print("You win!")
                break
            print("Draw")
            break
        
        a = int(input("Place O on? "))
        board, reward, d = ttt.setMachine(a, board, False)
        print(board)
        if d:
            if reward > 0:
                print("AI wins!")
                break
            if reward < 0:
                print("You win!")
                break
            print("Draw")
            break
        