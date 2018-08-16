"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from scipy.spatial.distance import pdist, squareform
from matplotlib import pyplot as plt
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

import time
class LinfPGDAttack:
  def __init__(self, model, epsilon, k, a, random_start, loss_func):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point."""
    self.model = model
    self.epsilon = epsilon
    self.k = k
    self.a = a
    self.rand = random_start
    self.loss_func = loss_func

    if loss_func == 'xent':
      loss = model.y_xent
    elif loss_func == 'cw':
      label_mask = tf.one_hot(model.y_input,
                              10,
                              on_value=1.0,
                              off_value=0.0,
                              dtype=tf.float32)
      correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
      wrong_logits = (1-label_mask) * model.pre_softmax - label_mask*1e4
      wrong_logit = tf.reduce_max(wrong_logits, axis=1)

      loss = -tf.nn.relu(correct_logit - wrong_logit + 50) 
    else:
      print('Unknown loss function. Defaulting to cross-entropy')
      loss = model.y_xent
      
    self.grad = tf.gradients(loss, model.x_input)[0]

  def perturb(self, x_nat, x_adv, y, sess):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
     

    mu = 1.0
    grad_history = np.zeros(x_nat.shape).astype(float)    
    
    for epoch in range(self.k):

      
      grad = sess.run(self.grad, feed_dict={self.model.x_input: x_adv,
                                           self.model.y_input: y})
                                         
      grad = np.divide(grad, np.linalg.norm(grad, ord=1, axis=1).reshape([-1, 1]))
      grad_history = mu * grad_history + grad
      
      x_adv += self.a * np.sign(grad_history)
  
        
      x_adv = np.clip(x_adv, x_nat - self.epsilon, x_nat + self.epsilon) 
      x_adv = np.clip(x_adv, 0, 1) # ensure valid pixel range


    return x_adv

    

if __name__ == '__main__':
  import json
  import sys
  import math

  from tensorflow.examples.tutorials.mnist import input_data

  from model import Model

  with open('config.json') as config_file:
    config = json.load(config_file)

  model_file = tf.train.latest_checkpoint(config['model_dir'])
  if model_file is None:
    print('No model found')
    sys.exit()

  model = Model()
  attack = LinfPGDAttack(model,
                         config['epsilon'],
                         config['k'],
                         config['a'],
                         config['random_start'],
                         config['loss_func'])
  saver = tf.train.Saver()

  mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

  with tf.Session() as sess:
    # Restore the checkpoint
    saver.restore(sess, model_file)

    # Iterate over the samples batch-by-batch
    num_eval_examples = config['num_eval_examples']
    eval_batch_size = config['eval_batch_size']
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

    

    print('Iterating over {} batches'.format(num_batches))
    
    
    x_adv_final = np.copy(mnist.test.images)
    
    for restart in range(50):
      # Initialize permutation
      permutation = np.arange(num_eval_examples)
      idx = np.arange(num_eval_examples)
      # Initialize data
      x_test, y_test = np.copy(mnist.test.images), np.copy(mnist.test.labels)
      
      
      x_adv = x_test + np.random.uniform(-attack.epsilon, attack.epsilon, x_test.shape)
        
      
      # per round
      t0 = time.time()
      
      for ibatch in range(num_batches):
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)
        
        x_batch = x_test[bstart:bend, :]
        x_batch_adv = x_adv[bstart:bend, :]
        y_batch = y_test[bstart:bend]
      
        
        x_adv[bstart:bend, :] = attack.perturb(x_batch, x_batch_adv, y_batch, sess)
      
      print('round Linf: ', np.max(np.abs(x_adv - mnist.test.images)))
      print('round adv acc: ', sess.run(attack.model.accuracy, feed_dict={attack.model.x_input: x_adv,
                                            attack.model.y_input: mnist.test.labels}))
                                            
      prediction = sess.run(attack.model.correct_prediction, feed_dict={attack.model.x_input: x_adv,
                                            attack.model.y_input: mnist.test.labels})
      ## Replace with wrong sample
      for i in range(prediction.shape[0]):
        if not prediction[i]:
          x_adv_final[i] = x_adv[i]        
        
      t1 = time.time()
            
      print('restart: ', restart, '   time per batch: ', t1 - t0)
      
      
      print('L2: ', np.mean(np.square(x_adv_final - mnist.test.images)))
      print('Linf: ', np.max(np.abs(x_adv_final - mnist.test.images)))
      
      print('adv acc: ', sess.run(attack.model.accuracy, feed_dict={attack.model.x_input: x_adv_final,
                                            attack.model.y_input: mnist.test.labels}))



    print('Storing examples')
    path = config['store_adv_path']
    
    np.save(path, x_adv_final)
    print('Examples stored in {}'.format(path))
