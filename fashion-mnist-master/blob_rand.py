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
import os
import sys
import time
from scipy.spatial.distance import pdist, cdist, squareform
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import cifar10_input
c = float(sys.argv[1])
print('c: ', c)

class LinfPGDAttack:
  def __init__(self, model, epsilon, num_steps, step_size, random_start, loss_func):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point."""
    self.model = model
    self.epsilon = epsilon
    self.num_steps = num_steps
    self.step_size = step_size
    self.rand = random_start

    if loss_func == 'xent':
      loss = model.xent
    elif loss_func == 'cw':
      label_mask = tf.one_hot(model.y_input,
                              10,
                              on_value=1.0,
                              off_value=0.0,
                              dtype=tf.float32)
      correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
      wrong_logit = tf.reduce_max((1-label_mask) * model.pre_softmax - 1e4*label_mask, axis=1)
      loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
    else:
      print('Unknown loss function. Defaulting to cross-entropy')
      loss = model.xent

    self.grad = tf.gradients(loss, model.x_input)[0]

  def perturb(self, x_nat, x_adv, y, sess):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
    
    batch_size = x_nat.shape[0] 
    
    for i in range(10):
      grad = sess.run(self.grad, feed_dict={self.model.x_input: x_adv,
                                            self.model.y_input: y})
      
      kxy, dxkxy = self.svgd_kernel(x_adv)
      svgd = -(np.matmul(kxy, -255.*grad.reshape([-1, 32*32*3])).reshape([-1, 32, 32, 3]) + dxkxy)/batch_size
      x_adv = np.add(x_adv, self.step_size * np.sign(c*svgd + 255.*grad), out=x_adv, casting='unsafe')

      x_adv = np.clip(x_adv, x_nat - self.epsilon, x_nat + self.epsilon) 
      x_adv = np.clip(x_adv, 0, 255) # ensure valid pixel range
      
      
    return x_adv
    
  def svgd_kernel(self, Theta):
    theta = Theta.reshape([-1, 32*32*3])/255.
    
    sq_dist = pdist(theta)
    pairwise_dists = squareform(sq_dist)**2
    
    h = np.median(pairwise_dists)
    h = np.sqrt(0.5*h/np.log(theta.shape[0]))

    # compute the rbf kernel
    Kxy = np.exp(-pairwise_dists / h**2 / 2)

    dxkxy = -np.matmul(Kxy, theta)
    sumkxy = np.sum(Kxy, axis=1)
    
    for i in range(theta.shape[1]):
      dxkxy[:, i] = dxkxy[:,i] + np.multiply(theta[:,i],sumkxy)
    dxkxy = dxkxy / (h**2)
    
    dxkxy = np.reshape(dxkxy, [-1, 32, 32, 3])
    return (Kxy, dxkxy)


if __name__ == '__main__':
  import json
  import sys
  import math


  from model import Model

  with open('config.json') as config_file:
    config = json.load(config_file)

  model_file = tf.train.latest_checkpoint(config['model_dir'])
  if model_file is None:
    print('No model found')
    sys.exit()

  model = Model(mode='eval')
  attack = LinfPGDAttack(model,
                         config['epsilon'],
                         config['num_steps'],
                         config['step_size'],
                         config['random_start'],
                         config['loss_func'])
  saver = tf.train.Saver()

  data_path = config['data_path']
  cifar = cifar10_input.CIFAR10Data(data_path)

  with tf.Session() as sess:
    # Restore the checkpoint
    saver.restore(sess, model_file)

    # Iterate over the samples batch-by-batch
    num_eval_examples = config['num_eval_examples']
    eval_batch_size = config['eval_batch_size']
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
    x_adv_final = np.copy(cifar.eval_data.xs)


    print('Iterating over {} batches'.format(num_batches))
    
    for restart in range(50):
      # Initialize permutation
      permutation = np.arange(num_eval_examples)
      idx = np.arange(num_eval_examples)
      # Initialize data
      x_test, y_test = np.copy(cifar.eval_data.xs), np.copy(cifar.eval_data.ys)
      
      x_adv = x_test + np.random.uniform(-attack.epsilon, attack.epsilon, x_test.shape)
      #x_adv = np.copy(x_test)    
      # per round
      t0 = time.time()
      
      for epoch in range(int(attack.num_steps/10)):
        np.random.shuffle(idx)
        x_test, x_adv, y_test = x_test[idx], x_adv[idx], y_test[idx]
        permutation = permutation[idx]
      
        for ibatch in range(num_batches):
          bstart = ibatch * eval_batch_size
          bend = min(bstart + eval_batch_size, num_eval_examples)
        
          x_batch = x_test[bstart:bend, :]
          x_batch_adv = x_adv[bstart:bend, :]
          y_batch = y_test[bstart:bend]
      
        
          x_adv[bstart:bend, :] = attack.perturb(x_batch, x_batch_adv, y_batch, sess)
          
      
      inv_permutation = np.argsort(permutation)
      x_adv = x_adv[inv_permutation]    

      print('Round Linf: ', np.max(np.abs(x_adv - cifar.eval_data.xs.astype(float))))
    
      ## Check Accuracy and Store Adversarial Samples
      total_corr, pred = 0, []
      for ibatch in range(num_batches):
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)

        x_batch = x_adv[bstart:bend, :]
        y_batch = cifar.eval_data.ys[bstart:bend]

        dict_adv = {model.x_input: x_batch,
                     model.y_input: y_batch}
        cur_corr, pred_batch = sess.run([model.num_correct, model.correct_prediction],
                                        feed_dict=dict_adv)

        total_corr += cur_corr
        pred.append(pred_batch)

      accuracy = total_corr / num_eval_examples
      pred = np.array(pred).flatten()
      
      print('Round Accuracy: ', accuracy)
      ## Replace with wrong sample
      for i in range(pred.shape[0]):
        if not pred[i]:
          x_adv_final[i,:] = x_adv[i,:]        
        
      t1 = time.time()
      print('round time: ', t1 - t0)
      ## Check Accuracy and Store Adversarial Samples
      total_corr, pred = 0, []
      for ibatch in range(num_batches):
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)

        x_batch = x_adv_final[bstart:bend, :]
        y_batch = cifar.eval_data.ys[bstart:bend]

        dict_adv = {model.x_input: x_batch,
                     model.y_input: y_batch}
        cur_corr, pred_batch = sess.run([model.num_correct, model.correct_prediction],
                                        feed_dict=dict_adv)

        total_corr += cur_corr
        pred.append(pred_batch)

      accuracy = total_corr / num_eval_examples
      pred = np.array(pred).flatten()
      
      print('adv acc: ', accuracy)
