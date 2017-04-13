import numpy as np
import objectives
import operator
import scipy.special
import tensorflow as tf

y_true_ph = tf.placeholder(tf.float32, shape=(4))
y_pred_ph = tf.placeholder(tf.float32, shape=(4))
huber_loss_tensor = objectives.huber_loss(y_true_ph, y_pred_ph)
mean_huber_loss_tensor = objectives.mean_huber_loss(y_true_ph, y_pred_ph)

sess = tf.Session()
with sess.as_default():
    y_true = [1, 2, 3, 4]
    y_pred = [1, 2.5, 4, 33]
    expected = scipy.special.huber(1, list(map(operator.sub, y_true, y_pred)))
    expected_mean = np.mean(expected)

    sess.run(tf.global_variables_initializer())
    feed_dict = {y_true_ph: y_true, y_pred_ph: y_pred}
    output = sess.run(huber_loss_tensor, feed_dict=feed_dict)
    #print(output)
    #print(expected)
    print(output == expected)

    output = sess.run(mean_huber_loss_tensor, feed_dict=feed_dict)
    #print(output)
    #print(expected_mean)
    print(output == expected_mean)
