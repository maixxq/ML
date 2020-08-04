import tensorflow as tf
import numpy as np

# define regression model as class object
class Model(tf.keras.Model):
    # initialize w = 2, b =0
    def __init__(self):
        self.w = tf.Variable(2.0)
        self.b = tf.Variable(0.0)
        self.learning_rate = 0.001
        self.optimizer = tf.optimizers.SGD(self.learning_rate)
    
    def __call__(self,x):
        return self.w * x + self.b
    
    # define loss function

    def loss(self,true_y, predicted_y):
        return tf.reduce_mean(tf.square(true_y - predicted_y))

    def train(self, x, y, learning_rate):
        with tf.GradientTape() as tape:
            loss_value = self.loss(y, self.__call__(x))

        gradients = tape.gradient(loss_value, [self.w,self.b])
    # update w(new) and b(new)
        self.optimizer.apply_gradients(zip(gradients,[self.w,self.b]))


# parameters
training_steps = 100000
display_step = 50

# obtain training data
x = np.array([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
              7.042,10.791,5.313,7.997,5.654,9.27,3.1])

y = np.array([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
              2.827,3.465,1.65,2.904,2.42,2.94,1.3])


model = Model()

for step in range(1, training_steps+1):

    if step % display_step == 0:
        predicted_y = model(x)
        loss_value = model.loss(y, model(x))
        model.train(x, y, model.learning_rate)

        print("step: %i,loss: %f, w: %f, b: %f" % (step,loss_value,model.w.numpy(),model.b.numpy()))

import matplotlib.pyplot as plt

# graphic display
plt.plot(x,y,'ro', label='original data')
plt.plot(x, np.array(model.w*x+model.b), label='fitted line')
plt.legend()
plt.show()