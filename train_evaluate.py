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
from time import process_time 

def get_time_str():
    #https://www.programiz.com/python-programming/datetime/current-datetime
    # datetime object containing current date and time
    now = datetime.now()
    return now.strftime("%d_%m_%Y_%H_%M_%S")

def get_trainable_str(trainable_kernel):
    trainable_str = ""
    if trainable_kernel.lower()=="true":
        trainable_str = "_trainable"  
        
    return trainable_str

def delete_file_if_exists(file):
    if os.path.exists(file):
        os.remove(file)
    else:
        print("The file does not exist: " + file)

def train_and_evaluate(datasetname,n_classes,batch_size,
         model_name, kernel, trainable_kernel, cp, dp, gamma,pooling_method,
         epochs,lr,keep_prob,weight_decay,
         log_dir, num_folds, fold, target_accuracies=[0.98]):
    
    print('Using the ' + datasetname + ' dataset.')
    
    trainable_str = get_trainable_str(trainable_kernel)
    
    if trainable_kernel.lower()=='true':
        print('Using a trainable kernel.')
        trainable_kernel = True
    else:
        trainable_kernel = False

    # dataset
    train_dataset, train_samples = datasets.get_dataset(datasetname, batch_size, k_folds=num_folds, fold=fold)
    validate_dataset, _ = datasets.get_dataset(datasetname, batch_size, subset="validate", shuffle=False, k_folds=num_folds, fold=fold)
        
    #Network
    kernel_fn = get_kernel(kernel, cp=cp, dp=dp, gamma=gamma, trainable=trainable_kernel, lambda_param=0.1)

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
    validate_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

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

    #Summary writers
    train_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir,
                                                                      'summaries',
                                                                      'train',
                                                                      kernel + trainable_str,
                                                                      dt_string))
    validate_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir,
                                                                         'summaries',
                                                                         'validate',
                                                                         kernel + trainable_str,
                                                                         dt_string))                                            
    
    best_validate_accuracy = 0
    best_train_accuracy = 0
    converge_times = np.zeros(len(target_accuracies))
    total_time_training = 0
    
    for ep in tqdm.trange(epochs, desc='Epoch Loop'):
        if ep < ep_cnt:
            continue

            # update epoch counter
        ep_cnt.assign_add(1)
        
        # train for an epoch
        for step, (x,y) in enumerate(train_dataset):
            if len(x.shape)==3:
                x = tf.expand_dims(x,3)
            time_start = process_time()  
            loss, logits = train_step(x,y)#train step
            time_stop = process_time()
            total_time_training = total_time_training + (time_stop - time_start)

            # Log every 100 batch
            if (step + 1) % 100 == 0:
                train_acc = 0
                validate_acc = 0
                with train_summary_writer.as_default():
                    for x_batch, y_batch in train_dataset:
                        if len(x_batch.shape)==3:
                            x_batch = tf.expand_dims(x_batch, 3)
                        train_logits = model(x_batch, training=False)
                        # Update train metrics
                        train_acc_metric(y_batch, train_logits)
                    train_acc = train_acc_metric.result()
                    train_acc_metric.reset_states()
                    tf.summary.scalar("accuracy", train_acc, step=(step + 1) + ep * nrof_steps_per_epoch)


                with validate_summary_writer.as_default():
                    for x_batch, y_batch in validate_dataset:
                        if len(x_batch.shape)==3:
                            x_batch = tf.expand_dims(x_batch, 3)
                        validate_logits = model(x_batch, training=False)
                        # Update validate metrics
                        validate_acc_metric(y_batch, validate_logits)

                    validate_acc = validate_acc_metric.result()
                    validate_acc_metric.reset_states()
                    tf.summary.scalar("accuracy", validate_acc, step=(step + 1) + ep * nrof_steps_per_epoch)

                print('[Step {}] train acc: {}'.format(step + 1, float(train_acc)))
                print('[Step {}] validate acc: {}'.format(step + 1, float(validate_acc)))

                if best_train_accuracy <= float(train_acc):
                    print("got a better train accuracy, updating validate accuracy")
                    best_validate_accuracy = float(validate_acc)
                    best_train_accuracy = float(train_acc)
                print("best validate accuracy so far: " + str(best_validate_accuracy))
                
                for i in range(len(target_accuracies)):
                    if target_accuracies[i] <= best_validate_accuracy and converge_times[i] == 0:
                        converge_times[i] = total_time_training
                        print("new converge time of: " + str(converge_times[i]))
                        print("for convergance of: " + str(target_accuracies[i]))
            
    return best_validate_accuracy, converge_times

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
        accuracies = [[],[],[],[],[]]
        learning_rates = [[],[],[],[],[]]
        lr_search_log_path = os.path.join(log_dir,'summaries','learning_rate_searches',
                                          kernel + trainable_str,get_time_str())
        lr_search_validate_summary_writer = tf.summary.create_file_writer(os.path.join(lr_search_log_path))
        current_lr = 0.2

        current_lr_step = 0
        num_folds = 5
        target_accuracies=[0.85,0.86,0.87,0.88,0.89,0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99]
        recorded_converge_times=[]
        for i in range(len(target_accuracies)):
            recorded_converge_times.append([[],[],[],[],[]])
        while current_lr >= 0.0002:
            for fold in range(num_folds):
                print("current learning rate: " + str(current_lr))
                print("current fold: " + str(fold))
                best_accuracy, converge_times = train_and_evaluate(datasetname,n_classes,batch_size,
                                                   model_name, kernel, trainable_kernel, cp, dp, gamma,pooling_method,
                                                   epochs,current_lr,keep_prob,weight_decay,
                                                   log_dir, num_folds, fold, target_accuracies)
                print("best validation accuracy: " + str(best_accuracy) + " with a learning rate of: " + str(current_lr))
                with lr_search_validate_summary_writer.as_default():
                    tf.summary.scalar("accuracy", best_accuracy, step=current_lr_step)
                accuracies[fold].append(best_accuracy)
                learning_rates[fold].append(current_lr)
                for i in range(len(target_accuracies)):
                    recorded_converge_times[i][fold].append(converge_times[i])
                
            
            current_lr = current_lr / 2
            current_lr_step = current_lr_step + 1
            
            lr_folder = "lr_" + str(current_lr_step)
            if not os.path.exists(os.path.join(lr_search_log_path,lr_folder)):
                os.makedirs(os.path.join(lr_search_log_path,lr_folder))
            np_accuracies = np.array(accuracies)
            np.save(os.path.join(lr_search_log_path,lr_folder,'numpy_accuracies'),np_accuracies)
            np_learning_rates = np.array(learning_rates)
            np.save(os.path.join(lr_search_log_path,lr_folder,'numpy_learning_rates'),np_learning_rates)

            for i in range(len(target_accuracies)):
                recorded_converge_time = np.array(recorded_converge_times[i])
                np.save(os.path.join(lr_search_log_path,lr_folder,
                                     'numpy_target_convergence_'+str(int(target_accuracies[i]*100))),
                        recorded_converge_time)
        
         
    else:
        train_and_evaluate(datasetname,n_classes,batch_size,
                           model_name, kernel, trainable_kernel, cp, dp, gamma,pooling_method,
                           epochs,lr,keep_prob,weight_decay,
                           log_dir, 7, 0)


if __name__=="__main__":
    main()
