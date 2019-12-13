
import tensorflow as tf
import numpy as np
import os
import time
import sys
from sklearn import metrics

from config import *
from utils import read_vocab,get_emb,process_file_trans,batch_iter

class transformer:
    def __init__(self,word_path,vec_path):
        self.words, self.word2id = read_vocab(word_path)
        self.vecs = get_emb(vec_path) 
        self.input_x = tf.placeholder(tf.int32,[None,SEQ_LENGTH],name='input_x')
        self.input_y = tf.placeholder(tf.float32,[None,NUM_CLASS],name='input_y')
        self.keep_prob = tf.placeholder(tf.float32,name='keep_prob')
        self.sess = tf.Session()
        self.model = self.build_model()

    def feed_data(self, x_batch, y_batch, keep_prob):
        feed_dict = {
                self.input_x: x_batch,
                self.input_y: y_batch,
                self.keep_prob: keep_prob
            }
        return feed_dict

    def build_model(self):
        """build transformer model"""
        with tf.name_scope("embedding"):
            embedding = tf.Variable(self.vecs)
            embedding_input = tf.nn.embedding_lookup(embedding,self.input_x)

        with tf.name_scope("full_layer"):
            layer_full = tf.layers.dense(embedding_input,HIDDEN_DIM,name="full")
            layer_full = tf.contrib.layers.dropout(layer_full,self.keep_prob)
            layer_full = tf.nn.relu(layer_full)
            # 更改下行代码，即可更改模型结构。
            model_layer = model_trans(layer_full)
            # 分类器
            self.logits = tf.reduce_mean(tf.layers.dense(model_layer, NUM_CLASS, name = "logits"),1)
            self.y_pred = tf.argmax(tf.nn.softmax(self.logits), 1) # 预测类别

        with tf.name_scope("optimize"):
            #损失函数，交叉熵
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(self.cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(self.loss)
        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def train(self,train_dir,val_dir,keep_prob=0.5):
        print("start training ...")

        # 载入训练集与验证集
        print("Loading training and validation data ...")
        start_time = time.time()
        x_train, y_train = process_file_trans(train_dir,self.word2id,self.words,SEQ_LENGTH)
        x_val, y_val = process_file_trans(val_dir,self.word2id, self.words, SEQ_LENGTH)
        time_dif = time.time() - start_time
        print("Time usage:",time_dif)
        
        print("Prepare saver and Session ...")
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)
        """
        # 测试代码
        feed_dict = {
            self.input_x: x_train[:BATCH_SIZE],
            self.input_y: y_train[:BATCH_SIZE],
            self.keep_prob: KEEP_PROB
            }
        logits = self.sess.run(self.optim, feed_dict=feed_dict) # 优化
        print(np.shape(logits))
        """
        print("Training and evaluating ...")
        start_time = time.time()
        total_batch = 0 # 总批次
        best_acc_val = 0.0
        last_improved = 0
        require_improvement = 1000 # 如果超过1000轮未提升，提前结束训练

        flag = False
        for epoch in range(EPOCHS):
            batch_train = batch_iter(x_train, y_train, BATCH_SIZE)
            for x_batch, y_batch in batch_train:
                feed_dict = self.feed_data(x_batch, y_batch, KEEP_PROB)
                if total_batch % PRINT_PER_BATCH == 0:
                    # 每多少轮输出在训练集和验证集上的性能
                    feed_dict[self.keep_prob] = 1.0
                    loss_train, acc_train = self.sess.run([self.loss,self.acc],feed_dict=feed_dict)
                    loss_val, acc_val = self.evaluate(x_val, y_val)

                    if acc_val > best_acc_val:
                        # 保存最好结果
                        best_acc_val = acc_val
                        last_improved = total_batch
                        saver.save(sess=self.sess, save_path=SAVE_PATH)
                        improved_str = '***'
                    else:
                        improved_str = ''

                    time_dif = time.time() - start_time
                    msg = 'Iter: {0:>6}, Train Loss:{1:>6.2}, Train Acc:{2:7.2%},'\
                        + 'Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                    print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif,improved_str))
                feed_dict[self.keep_prob] = KEEP_PROB
                #print(len(feed_dict[self.input_x]),len(feed_dict[self.input_y]))
                self.sess.run(self.optim, feed_dict=feed_dict) # 优化
                total_batch += 1

                if total_batch - last_improved > require_improvement:
                    # 早停策略，如果设置的阈值太小，可能会造成训练不充分
                    print("No optimization for a long time , auto-stopping ...")
                    flag = True
                    break # 跳出循环
            if flag:# 两层循环，因此需要break两次
                break

    def test(self,test_dir,keep_prob=1.0):
        print("Loading test data ...")
        start_time = time.time()
        x_test, y_test = process_file_trans(test_dir,self.word2id,self.words,SEQ_LENGTH)
        saver = tf.train.Saver()
        saver.restore(sess=self.sess, save_path=SAVE_PATH) # 读取保存的模型
        print("Testing ...")
        loss_test, acc_test = self.evaluate(x_test, y_test)
        msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
        print(msg.format(loss_test, acc_test))
        data_len = len(x_test)
        num_batch = int((data_len - 1) / BATCH_SIZE) + 1

        y_test_cls = np.argmax(y_test,1)
        y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32) # 保存预测结果
        for i in range(num_batch):  # 逐批次处理
            start_id = i * BATCH_SIZE
            end_id = min((i + 1) * BATCH_SIZE, data_len)
            feed_dict = {
                model.input_x: x_test[start_id:end_id],
                model.keep_prob: 1.0
            }
            y_pred_cls[start_id:end_id] = self.sess.run(self.y_pred, feed_dict=feed_dict)

        # 评估
        print("Precision, Recall and F1-Score...")
        print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=CATEGORY))
        
        # 混淆矩阵
        print("Confusion Matrix...")
        cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
        print(cm)

        time_dif = time.time() - start_time
        print("Time usage:",time_dif)
        pass

    def evaluate(self,x_,y_):
        # 评估在某一数据集上的准确率和损失
        data_len = len(x_)
        batch_eval = batch_iter(x_, y_, 128)
        total_loss = 0.0
        total_acc = 0.0
        for x_batch, y_batch in batch_eval:
            batch_len = len(x_batch)
            feed_dict = self.feed_data(x_batch, y_batch, 1.0)
            loss, acc = self.sess.run([self.loss, self.acc], feed_dict=feed_dict)
            total_loss += loss * batch_len
            total_acc += acc * batch_len

        return total_loss / data_len, total_acc / data_len

    def model_trans(self, inp):
        return inp


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
         raise ValueError("""usage: python transformer.py [train / test]""")
    print('Configuring model ...')
    model = transformer(WORD_PATH,VEC_PATH)
    if sys.argv[1] == 'train':
        model.train(TRAIN_DIR, VAL_DIR)
    elif sys.argv[1] == 'test':
        model.test(TEST_DIR)
