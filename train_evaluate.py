from __future__ import division, print_function, absolute_import

import os
import functools
import numpy as np
import tensorflow as tf
import tqdm
import datasets
from models import models_factory
from layers import *
import click
from datetime import datetime

def get_time_str():
    #https://www.programiz.com/python-programming/datetime/current-datetime
    # datetime object containing current date and time
    now = datetime.now()
    return now.strftime("%d_%m_%Y_%H_%M_%S")

def get_trainable_str(trainable_kernel):
    trainable_str = ""
    if trainable_kernel:
        trainable_str = "_trainable"  
        
    return trainable_str

def train_and_evaluate(datasetname,n_classes,batch_size,
         model_name, kernel, trainable_kernel, cp, dp, gamma,pooling_method,
         epochs,lr,keep_prob,weight_decay,
         log_dir):
    
    print('Using the ' + datasetname + ' dataset.')

    # dataset
    train_dataset, train_samples = datasets.get_dataset(datasetname, batch_size)
    test_dataset, _ = datasets.get_dataset(datasetname, batch_size, subset="test", shuffle=False)

    if trainable_kernel.lower()=='true':
        print('Using a trainable kernel.')
        
    #Network
    kernel_fn = get_kernel(kernel, cp=cp, dp=dp, gamma=gamma, trainable=trainable_kernel.lower()=='true')

    model = models_factory.get_model(model_name,
                      num_classes=n_classes,
                      keep_prob=keep_prob,
                      kernel_fn=kernel_fn,
                      pooling=pooling_method)

    #Train optimizer, loss
    nrof_steps_per_epoch = (train_samples//batch_size)
    boundries = [nrof_steps_per_epoch*10, nrof_steps_per_epoch*15]
    values = [lr, lr*0.1, lr*0.01]
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(\
                    boundries,
                    values)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    #metrics
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    #Train step
    @tf.function
    def train_step(x,labels):
        with tf.GradientTape() as t:
            logits = model(x, training=True)
            loss = loss_fn(labels, logits)

        gradients = t.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, logits

    #Run

    ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)
    
    dt_string = get_time_str()
    
    trainable_str = get_trainable_str(trainable_kernel)

    #Summary writers
    train_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir,
                                                                      'summaries',
                                                                      'train'))
    granular_test_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir,
                                                                 'summaries',
                                                                 'test',
                                                                  dt_string,
                                                                  kernel + trainable_str,
                                                                 'granular_test'))                                               

    best_test_accuracy = 0
    best_test_accuracy_to_return = 0
    
    for ep in tqdm.trange(epochs, desc='Epoch Loop'):
        if ep < ep_cnt:
            continue

            # update epoch counter
        ep_cnt.assign_add(1)
        
        with train_summary_writer.as_default():
            # train for an epoch
            for step, (x,y) in enumerate(train_dataset):
                if len(x.shape)==3:
                    x = tf.expand_dims(x,3)
                tf.summary.image("input_image", x, step=optimizer.iterations)
                loss, logits = train_step(x,y)
                train_acc_metric(y, logits)
                tf.summary.scalar("loss", loss, step=optimizer.iterations)
                
                # Log every 100 batch
                if (step + 1) % 100 == 0:
                    train_acc = train_acc_metric.result() 
                    print("Training loss {:1.2f}, accuracu {} at step {}".format(\
                            loss.numpy(),
                            float(train_acc),
                            step + 1))
                    
                    
                    with granular_test_summary_writer.as_default():
                        for x_batch, y_batch in test_dataset:
                            if len(x_batch.shape)==3:
                                x_batch = tf.expand_dims(x_batch, 3)
                            test_logits = model(x_batch, training=False)
                            # Update test metrics
                            test_acc_metric(y_batch, test_logits)

                        test_acc = test_acc_metric.result()
                        tf.summary.scalar("accuracy", test_acc, step=(step + 1) + ep * nrof_steps_per_epoch)
                        test_acc_metric.reset_states()
                        print('[Step {}] Test acc: {}'.format(step + 1, float(test_acc)))
                        if best_test_accuracy <= float(test_acc):
                            best_test_accuracy = float(test_acc)
                            best_test_accuracy_to_return = test_acc
                        print("best test accuracy so far: " + str(best_test_accuracy))


            # Display metrics at the end of each epoch.
            train_acc = train_acc_metric.result()
            tf.summary.scalar("accuracy", train_acc, step=ep)
            print('Training acc over epoch: %s' % (float(train_acc),))
            # Reset training metrics at the end of each epoch
            train_acc_metric.reset_states()
            
    return best_test_accuracy

#args
@click.command()
##Data args
@click.option("-d","--datasetname", default="mnist", type=click.Choice(['cifar10','mnist', 'fashion_mnist']))
@click.option("--n_classes", default=10)
##Training args
@click.option('--model_name', default='lenetKNN')
@click.option('--kernel', default='polynomial')
@click.option('--trainable_kernel', default='False')
@click.option('--pooling_method', default='max') 
@click.option('--cp', default=1.0)
@click.option('--dp', default=3.0)
@click.option('--gamma', default=1.0)
@click.option("--batch_size", default=50)
@click.option("--epochs", default=20)
@click.option("--lr", default=0.003)
@click.option("--keep_prob", default=1.0)
@click.option("--weight_decay", default=0.0)
@click.option("--lr_search", default='False') 
##logging args
@click.option("-o","--base_log_dir", default="logs")          
def main(datasetname,n_classes,batch_size,
         model_name, kernel, trainable_kernel, cp, dp, gamma,pooling_method,
         epochs,lr,keep_prob,weight_decay,lr_search,
         base_log_dir):
    
    
    log_dir = os.path.join(os.path.expanduser(base_log_dir),
                           "{}".format(datasetname))
    os.makedirs(log_dir, exist_ok=True)
    
    #Fix TF random seed
    tf.random.set_seed(1777)
    
    trainable_str = get_trainable_str(trainable_kernel)

    if lr_search.lower()=='true':
        lr_search_test_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir,
                                                                 'summaries',
                                                                 'learning_rate_searches',
                                                                  get_time_str(),
                                                                  kernel + trainable_str,
                                                                 'granular_test'))
        current_lr = 0.1
        epochs = 1
        current_lr_step = 0
        while current_lr >= 0.001:
            print("current_lr: " + str(current_lr))
            best_accuracy = train_and_evaluate(datasetname,n_classes,batch_size,
                                               model_name, kernel, trainable_kernel, cp, dp, gamma,pooling_method,
                                               epochs,current_lr,keep_prob,weight_decay,
                                               log_dir)
            print("best_accuracy: " + str(best_accuracy))
            with lr_search_test_summary_writer.as_default():
                tf.summary.scalar("accuracy", best_accuracy, step=current_lr_step)
            current_lr = current_lr / 2
            current_lr_step = current_lr_step + 1
    else:
        train_and_evaluate(datasetname,n_classes,batch_size,
                           model_name, kernel, trainable_kernel, cp, dp, gamma,pooling_method,
                           epochs,lr,keep_prob,weight_decay,
                           log_dir)


if __name__=="__main__":
    main()
