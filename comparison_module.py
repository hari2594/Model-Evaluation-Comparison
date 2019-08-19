
##########################################################################################################################
##  Importing the necessary libraries:                                                                                  ##
##########################################################################################################################

import os
import numpy as np
from time import gmtime, strftime
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score,average_precision_score, recall_score, f1_score, roc_curve ,auc, matthews_corrcoef, roc_auc_score, classification_report
from sklearn.metrics import r2_score, median_absolute_error, precision_recall_curve, mean_absolute_error, mean_squared_error, mean_squared_log_error
import evaluation_module


class comparison_module():

    def __init__(self, actual, pred, threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
            model_reference_name = 'sample_model', model_type = 'classification',
        plot_classification_metric = ['TP','FP','FN','TN','Accuracy',
               'Precision0','Precision1','recall0','recall1','f1','mcc','roc_auc']):
        self.actual = actual
        self.pred = pred


    def metric_line_plot(self, evaluate_db = None ,model_id = None, comparison_metrics = None):
        '''
        Comparing the different models based on the model metrics - with the help of visulization plots
        '''
        # merging the metric results of different models:
        evaluate_db = pd.concat(evaluate_db,axis=0).reset_index(drop=True)

        if comparison_metrics:
            _metric = comparison_metrics
        else:
            _metric = self.plot_classification_metric
        if model_id:
            data = evaluate_db[evaluate_db['Unique_ModelID'].isin(model_id)]
            eval_plt = evaluation_module.evaluation_plots(metrics_db = data, classification_metric = _metric)
            eval_plt.metric_plots_1()
            plt.close()

    def metric_bar_plot(self, evaluate_db = None ,model_id = None, comparison_metrics = ['TP','FP','FN','TN','Accuracy',
               'Precision0','Precision1','recall0','recall1','f1','mcc','roc_auc'], threshold = None):
        # merging the metric results of different models:
        evaluate_db = pd.concat(evaluate_db,axis=0).reset_index(drop=True)
        # extracting the data for the required threshold
        _df = pd.DataFrame(columns=evaluate_db.columns)
        for i,j in zip(model_id,threshold):
            _df = pd.concat([_df,evaluate_db[(evaluate_db['Unique_ModelID'] == i) & (evaluate_db['Threshold'] == j)]],axis=0).reset_index(drop=True)
        model_metric_db_thr = _df.copy()
        del _df
        # plotting the bar plot for all specified metrics:
        for metric in comparison_metrics:
            fig, ax = plt.subplots(figsize = (5,5))
            # getting the values of the metrics to add it as a data label on the bar chart
            label = list(model_metric_db_thr[metric])
            ax.bar(x=model_metric_db_thr['Unique_ModelID'],height=model_metric_db_thr[metric],color='black')
            plt.ylabel(metric)
            if metric not in ['TN','TP','FN','FP']:
                plt.ylim([0.0,1.5])
            else:
                plt.ylim([0,max(label)+50])
            for i in range(len(model_id)):
                plt.text(x = model_id[i] , y = label[i]+0.01, s = round(label[i],2), size = 15)
            plt.show()

    def roc_auc_plot(self):
        # for setting the color for different curves:
        colour = ['red','orange','blue','black','green']
        model_count=0
        pred = self.pred
        for i in pred:
            # incrementing the model_count:
            model_count+=1
            #Preparing ROC curve
            false_positive_rate, true_positive_rate, thresholds = roc_curve(self.actual, i)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            #Plotting ROC Curve
            plt.title('ROC CURVE')
            plt.plot(false_positive_rate, true_positive_rate, 'b',color = colour[model_count],
            label='AUC_model -'+str(model_count)+ '= %0.2f'% roc_auc)
            plt.legend(loc='lower right')
            plt.plot([0,1],[0,1],'r--')
            plt.xlim([-0.1,1.2])
            plt.ylim([-0.1,1.2])
            plt.ylabel('True_Positive_Rate')
            plt.xlabel('False_Positive_Rate')
        plt.show()


    def lift_chart(self, data=None, X='decile', y='lift', model_id = None):
        '''
        Plotting the lift chart aginst various deciles
        '''
        # for setting the color for different curves:
        for i,j in zip(data,model_id):
            # incrementing the model_count:
            plt.title('Lift chart')
            plt.plot(i[X], i[y], 'b',label='Lift_chart -'+str(j))
            plt.legend(loc='lower right')
            plt.ylabel('lift')
            plt.xlabel('decile')
        plt.show()
