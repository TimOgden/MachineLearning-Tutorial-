import tensorflow as tf

x1 = tf.constant(5)
x2 = tf.constant(6)

#This does work
#result = x1 * x2

#This is more efficient
result = tf.multiply(x1,x2)
print(result)

'''
sess = tf.Session()
print(sess.run(result))
sess.close()
'''

#Another way to do it, will automatically close sess
with tf.Session() as sess:
	print(tf.test.gpu_device_name)

with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

with tf.Session() as sess:
    print ('Mulitplication: ', sess.run(c))