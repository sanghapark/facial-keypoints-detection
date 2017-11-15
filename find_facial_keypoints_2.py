import pandas as pd
import numpy as np
import tensorflow as tf
import utils.data as ud
import datetime as dt
import os
from sklearn.cross_validation import train_test_split
from utils.constants import *
from utils.cnnmodel_general import CnnModel
from utils.cnnmodel_simple import CnnModel as simple_cnn_model

CHECK_POINT_DIR = TB_SUMMARY_DIR = './output/tf'
SAVE_MODEL = False

tf.set_random_seed(777)  # reproducibility

X_total, Y_total = ud.load_data_with_image_in_1D()
X_train, X_valid, Y_train, Y_valid = train_test_split(X_total, Y_total, test_size=VALIDATION_DATA_RATIO)


sess = tf.Session()
# cnnmodel01 = CnnModel(sess, 'CnnModel01')
cnnmodel01 = simple_cnn_model(sess, 'SimpleCnnModel01')
cnnmodel01.summary = tf.summary.merge_all()

tf.summary.scalar("loss", cnnmodel01.cost)
last_epoch = tf.Variable(0, name='last_epoch')


sess.run(tf.global_variables_initializer())

# Create summary writer
writer = tf.summary.FileWriter(TB_SUMMARY_DIR)
writer.add_graph(sess.graph)
global_step = 0

# Saver and Restore
saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(CHECK_POINT_DIR)

if checkpoint and checkpoint.model_checkpoint_path:
    try:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    except:
        print("Error on loading old network weights")
else:
    print("Could not find old network weights")


start_from = sess.run(last_epoch)


# train my model
print('Start learning from:', start_from)

cost_batch_vals = []
cost_valid_vals = []

print("# of Training Images: {}, # of Validation Images: {} \nStart Learning...".format(X_train.shape[0], X_valid.shape[0]))
for epoch in range(start_from, N_EPOCH):
    print('EPOCH: {} of {}'.format(epoch+1, N_EPOCH))
    n_batches = int(np.ceil(X_train.shape[0]/BATCH_SIZE))
    print("Total # of batches: {}".format(n_batches))
    for batch_index in range(n_batches):
        X_batch, Y_batch = ud.fetch_batch(X_train, Y_train, batch_index*BATCH_SIZE, BATCH_SIZE)
        print(X_batch.shape, Y_batch.shape)
        cost_val, _ = cnnmodel01.train(X_batch, Y_batch, keep_prop=0.5)
        cost_batch_vals.append(cost_val)
        print('\t batch: {:04d} of {}, cost (SSE): {:.9f}'.format(batch_index, n_batches, cost_val))

        # s = cnnmodel01.summarize(X_batch, Y_batch, keep_prop=0.5)
        # writer.add_summary(s, global_step=global_step)
        # global_step += 1
        
    cost_valid_val = cnnmodel01.validate(X_valid, Y_valid)
    cost_valid_vals.append(cost_valid_val)

    print('validation cost (SSE): {:.9f}'.format(cost_valid_val))
    print('validation cost (RMSE): {:.9f}'.format(np.sqrt(cost_valid_val/float(X_valid.shape[0]))))
    
    print("Saving network...")
    sess.run(last_epoch.assign(epoch + 1))
    if not os.path.exists(CHECK_POINT_DIR):
        os.makedirs(CHECK_POINT_DIR)
    saver.save(sess, CHECK_POINT_DIR + "/model", global_step=batch_index)
    
    print('='*100)
    if epoch > N_PAST_COST_VALS and np.mean(cost_valid_vals[-(N_PAST_COST_VALS+1):-1]) < cost_valid_val:
        print("Eearly Stopped!! Hardly getting better performance")
        break
print('Learning Finished!')


if not os.path.exists('output'):
    os.makedirs('output')

datetime = dt.datetime.now().strftime("%Y%m%d_%H%M")

if not os.path.exists('output/{}'.format(datetime)):
    os.makedirs('output/{}'.format(datetime))

save_path = tf.train.Saver().save(sess, "./output/{}/cnn_model_by_tensorflow.ckpt".format(datetime))

with open('./output/{}/validation_error.csv'.format(datetime), 'w') as file:
    for err in cost_valid_vals:
        file.write("%s\n" % err)


X_test, _ = ud.load_data_with_image_in_1D(test=True)
print('Predicting {} Test Data...'.format(X_test.shape[0]))
total_output = pd.DataFrame()
for batch_index in range(1, int(np.ceil(X_test.shape[0]/BATCH_SIZE)+1)):
    start = (batch_index-1)*BATCH_SIZE
    end = batch_index*BATCH_SIZE
    Y_predicted = cnnmodel01.predict(X_test[start:end,], keep_prop=1.0)
    print("Predicting {} ~ {} test images".format(start, len(Y_predicted)))
    partial_output = ud.batch_output_for_kaggle_submission(Y_predicted, start, len(Y_predicted))
    total_output = pd.concat([total_output, partial_output])
total_output.to_csv("./output/{}/kaggle_submission_CNN_TF.csv".format(datetime), index=0, columns = ['RowId','Location'] )

sess.close()
print('Finished Predicting Test data! Checkout the output file for Kaggle submission.')
