
##########################################################################################################################
# Importing the necessary libraries:
##########################################################################################################################

import os
import numpy as np
from termcolor import colored
from time import gmtime, strftime
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as misno
import warnings
import logging
logger = logging.getLogger('ftpuploader')
import random
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score, recall_score, f1_score, auc, matthews_corrcoef, roc_auc_score
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error, mean_squared_error, mean_squared_log_error
import seaborn as sns; sns.set(rc={"lines.linewidth":3})
warnings.filterwarnings("ignore")

#########################################################################################################################
## This class is used to generate the metrics table for different sets of thresold values for the classification, and
## outputs the set of metrics for the regression problem. (which will be later used to plot the graphs and to understand
## the model behavior and imporove the model performance)
#########################################################################################################################

class Evaluation():
    
    '''
    This Evaluation Class will deal with the classification & Regression models.
    This returns the metrics for different set of thresholds
    '''
    
    def __init__(self, actual, pred, threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
                 model_reference_name = 'sample_model', model_type = 'classification'):
        '''
        actual = Actual value (list format)
        pred_prob = Predicted probablity (list format)
        threshold = by default it takes [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                    * you can change the list with different set of values *
        '''
        self.actual = actual
        self.pred = pred
        self.threshold = threshold
        self.model_reference_name = model_reference_name
        self.model_type = model_type
        
    def get_confusion_matrix_values(self, pred_value):
        '''
        Getting the confusion martics based on actual and predicted values
        '''
        cm = confusion_matrix(self.actual, pred_value)
        return(cm[0][0], cm[0][1], cm[1][0], cm[1][1])
    
    def get_pred_value_threshold_lvl(self):
        '''
        Getting the predicted values as 0 and 1 based on different sets of threshold
        '''
        pred_value = pd.DataFrame()
        try:
            for i in self.threshold:
                col_name = "Threshold_"+str(i)
                pred_value[col_name] = [1 if j<= i else 0 for j in self.pred]
            return(pred_value)
        
        except BaseException as e:
            logger.error('Error: ' + str(e))
            return(None)
        
    def metrics_classification(self, pred_value):
        '''
        Calculating the metrics based on different sets of threshold:
        --------------------------------------------------------------
        metrics considered =  ['Threshold','TP','FP','FN','TN','Accuracy','Precision','recall','f1','mcc','roc_auc']
        '''
        # Creating a metrics dcictionary:
        key = ['Unique_ModelID','Model_Reference_name','Threshold','TP','FP','FN','TN','Accuracy',
               'Precision','recall','f1','mcc','roc_auc','actual_pred_details','Time_stamp']
        metrics_db = dict([(i, []) for i in key])
        
        try:
            # Getting the metrics for different threshold ranges:
            for i in self.threshold:
                col_name = "Threshold_"+str(i)
                metrics_db['Unique_ModelID'].append(str(self.model_reference_name)+'_'+str(int(round(time.time() * 1000)))[:10])
                metrics_db['Model_Reference_name'].append(self.model_reference_name)
                metrics_db['Threshold'].append(i)
                TP, FP, FN, TN = self.get_confusion_matrix_values(pred_value = pred_value[col_name])
                metrics_db['TP'].append(TP)
                metrics_db['FP'].append(FP)
                metrics_db['FN'].append(FN)
                metrics_db['TN'].append(TN)
                metrics_db['Accuracy'].append(accuracy_score(self.actual,pred_value[col_name]))
                metrics_db['Precision'].append(precision_score(self.actual,pred_value[col_name]))
                metrics_db['recall'].append(recall_score(self.actual,pred_value[col_name]))
                metrics_db['f1'].append(f1_score(self.actual,pred_value[col_name]))
                metrics_db['mcc'].append(matthews_corrcoef(self.actual,pred_value[col_name]))
                metrics_db['roc_auc'].append(roc_auc_score(self.actual,pred_value[col_name]))
                metrics_db['Time_stamp'].append(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
                metrics_db['actual_pred_details'].append({'actual':self.actual, 'pred':self.pred})

            # returning the metrics db in dataframe format:
            return(pd.DataFrame(metrics_db))
        
        except BaseException as e:
            logger.error('Error: ' + str(e))
            return(None)
    
    def metrics_regression(self):
        '''
        Calculating the below metrics for the regression model:
        -------------------------------------------------------
        - mean_absolute_error : The mean_absolute_error function computes mean absolute error, 
          a risk metric corresponding to the expected value of the absolute error loss or -norm loss.
        - mean_squared_error : The mean_squared_error function computes mean square error, 
          a risk metric corresponding to the expected value of the squared (quadratic) error or loss.
        - mean_squared_log_error: The mean_squared_log_error function computes a risk metric corresponding to the expected 
          value of the squared logarithmic (quadratic) error or loss.
        - median_absolute_error: The median_absolute_error is particularly interesting because it is robust to outliers. 
          The loss is calculated by taking the median of all absolute differences between the target and the prediction.
        - r2_score : The r2_score function computes the coefficient of determination, usually denoted as R².
          It represents the proportion of variance (of y) that has been explained by the independent variables in 
          the model. It provides an indication of goodness of fit and therefore a measure of how well unseen 
          samples are likely to be predicted by the model, through the proportion of explained variance.          
        '''
        key = ['Unique_ModelID','Model_Reference_name','mean_absolute_error', 'mean_squared_error', 'mean_squared_log_error',
               'median_absolute_error', 'r2_score', 'actual_pred_details','Time_stamp']
        metrics_db = dict([(i, []) for i in key])
        
        metrics_db['Unique_ModelID'].append(str(self.model_reference_name)+'_'+str(int(round(time.time() * 1000)))[:10])
        metrics_db['Model_Reference_name'].append(self.model_reference_name)
        metrics_db['mean_absolute_error'].append(mean_absolute_error(actual,pred))
        metrics_db['mean_squared_error'].append(mean_squared_error(actual,pred))
        metrics_db['mean_squared_log_error'].append(mean_squared_log_error(actual,pred))
        metrics_db['median_absolute_error'].append(median_absolute_error(actual,pred))
        metrics_db['r2_score'].append(r2_score(actual,pred))
        metrics_db['actual_pred_details'].append(actual_pred_details(actual,pred))
        metrics_db['mean_squared_error'].append(mean_squared_error(actual,pred))
        
    def metrics(self,pred_value = None):
        '''
        Calculating the metrics table for classification | regression problem.
        '''
        try:
            if self.model_type == 'classification':
                metrics_db = self.metrics_classification(pred_value = pred_value)
            elif self.model_type == 'regression':
                metrics_db = self.metrics_regression()
            return(metrics_db)
        except BaseException as e:
            logger.error('Error: ' + str(e))


#########################################################################################################################
## This class is used to generate the graphs based on the metrics table, which was generated by the above evaluation 
## module. These plots can be exported to the local storage for our future reference.
#########################################################################################################################

class evaluation_plots():
    
    def __init__(self,metrics_db,classification_metric = ['TP','FP','FN','TN','Accuracy', 'Precision','recall','f1','mcc','roc_auc']):
        '''
        metric_db - Should be passed as a pandas Dataframe
        model_id - Should be passed as a list
        '''
        self.classification_metric = classification_metric
        self.metrics_db = metrics_db
        self.hue_feat = 'Unique_ModelID'
        
        
    def metric_plots(self):
        '''
        Plotting the graphs for all|specified evaluation metrics
        '''                                  
        for i in self.classification_metric:
            sns.lineplot(x='Threshold', y=i,hue = self.hue_feat, markers=True, dashes=True, data=self.metrics_db)
            plt.show()

#########################################################################################################################
## This is the main evaluator module - which runs the above two classes and saves the results based on the user
## request:
#########################################################################################################################

class ModelEvaluation(Evaluation,evaluation_plots):

    def __init__(self, actual, pred, threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
            model_reference_name = 'sample_model', model_type = 'classification',
        plot_classification_metric = ['TP','FP','FN','TN','Accuracy', 'Precision','recall','f1','mcc','roc_auc']):
        '''
        actual = Actual value (list format)
        pred_prob = Predicted probablity (list format)
        threshold = by default it takes [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                    * you can change the list with different set of values *
        '''
        self.actual = actual
        self.pred = pred
        self.threshold = threshold
        self.model_reference_name = model_reference_name
        self.model_type = model_type
        self.plot_classification_metric = plot_classification_metric

    def evaluate(self,evaluate_save=False, plots_show = False, plots_save = False):
        '''
        This returns the metrics for different set of thresholds.
        plots_show = True -->Shows the plots for all the metrics
        evaluate_save = True --> saves the results in .csv file and displays the directory details.
        '''

        ## Initialising the evaluation class:
        evalu = Evaluation(actual = self.actual, pred = self.pred,threshold = self.threshold,model_reference_name=self.model_reference_name,model_type = self.model_type)
        ## Getting the predicted values for different thresholds:
        pred_value = evalu.get_pred_value_threshold_lvl()
        ## Getting the evaluation metrics for different thresholds:
        metrics_db = evalu.metrics(pred_value)
        if evaluate_save:
            metrics_db.to_csv('evaluation_result.csv')
            print("The results are save to - ",os.getcwd()+'\\evaluation_result.csv')
        if plots_show:
            eval_plt = evaluation_plots(metrics_db = metrics_db, classification_metric = self.plot_classification_metric)
            eval_plt.metric_plots()
            plt.close()

        return(metrics_db)

    def Compare_models(self, evaluate_db = None ,model_id = None, comparison_metrics = None):
        '''
        Comparing the different models based on the model metrics - with the help of visulization plots
        '''
        if comparison_metrics:
            _metric = comparison_metrics
        else:
            _metric = self.plot_classification_metric
        if model_id:
            metrics_db = self.evaluate()
            data = evaluate_db[evaluate_db['Unique_ModelID'].isin(model_id)]
            eval_plt = evaluation_plots(metrics_db = data, classification_metric = _metric)
            eval_plt.metric_plots()
            plt.close()


####################################################################################################################################
##                                             END - EVALUATION MODULE                                                            ##
####################################################################################################################################