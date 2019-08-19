
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
import logging
logger = logging.getLogger('ftpuploader')
import random
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score,average_precision_score, recall_score, f1_score, roc_curve ,auc, matthews_corrcoef, roc_auc_score, classification_report
from sklearn.metrics import r2_score, median_absolute_error, precision_recall_curve, mean_absolute_error, mean_squared_error, mean_squared_log_error
import seaborn as sns; sns.set(rc={"lines.linewidth":3})
from inspect import signature
warnings.filterwarnings("ignore")
from plot_metric.functions import BinaryClassification

###########################################################################################################################
## This class is used to generate the metrics table for different sets of thresold values for the classification, and    ##
## outputs the set of metrics for the regression problem. (which will be later used to plot the graphs and to understand ##
## the model behavior and imporove the model performance)                                                                ##
###########################################################################################################################

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
        Getting the confusion martix based on actual and predicted values
        '''
        tn, fp, fn, tp = confusion_matrix(self.actual, pred_value).ravel()
        return(tn, fp, fn, tp)

    def get_pred_value_threshold_lvl(self):
        '''
        Getting the predicted values as 0 and 1 based on different sets of threshold
        '''
        pred_value = pd.DataFrame()
        try:
            for i in self.threshold:
                col_name = "Threshold_"+str(i)
                pred_value[col_name] = [1 if j>= i else 0 for j in self.pred]
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
               'Precision0','Precision1','recall0','recall1','f1','mcc','roc_auc','Time_stamp']
        metrics_db = dict([(i, []) for i in key])
        
        try:
            # Getting the metrics for different threshold ranges:
            id = str(self.model_reference_name)+'_'+str(int(round(time.time() * 1000)))[:10]
            for i in self.threshold:
                metrics_db['Unique_ModelID'].append(id)
                col_name = "Threshold_"+str(i)
                metrics_db['Model_Reference_name'].append(self.model_reference_name)
                metrics_db['Threshold'].append(round(i,2))
                TN, FP, FN, TP = self.get_confusion_matrix_values(pred_value = pred_value[col_name])
                metrics_db['TP'].append(TP)
                metrics_db['FP'].append(FP)
                metrics_db['FN'].append(FN)
                metrics_db['TN'].append(TN)
                metrics_db['Accuracy'].append(accuracy_score(self.actual,pred_value[col_name]))
                metrics_db['Precision0'].append(precision_score(self.actual,pred_value[col_name],pos_label=0))
                metrics_db['Precision1'].append(precision_score(self.actual,pred_value[col_name],pos_label=1))
                metrics_db['recall0'].append(recall_score(self.actual,pred_value[col_name],pos_label=0))
                metrics_db['recall1'].append(recall_score(self.actual,pred_value[col_name],pos_label=1))
                metrics_db['f1'].append(f1_score(self.actual,pred_value[col_name]))
                metrics_db['mcc'].append(matthews_corrcoef(self.actual,pred_value[col_name]))
                metrics_db['roc_auc'].append(roc_auc_score(self.actual,pred_value[col_name]))
                metrics_db['Time_stamp'].append(strftime("%Y-%m-%d %H:%M:%S", gmtime()))

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
               'median_absolute_error', 'r2_score','Time_stamp']
        metrics_db = dict([(i, []) for i in key])
        
        metrics_db['Unique_ModelID'].append(str(self.model_reference_name)+'_'+str(int(round(time.time() * 1000)))[:10])
        metrics_db['Model_Reference_name'].append(self.model_reference_name)
        metrics_db['mean_absolute_error'].append(mean_absolute_error(self.actual,self.pred))
        metrics_db['mean_squared_error'].append(mean_squared_error(self.actual,self.pred))
        metrics_db['mean_squared_log_error'].append(mean_squared_log_error(self.actual,self.pred))
        metrics_db['median_absolute_error'].append(median_absolute_error(self.actual,self.pred))
        metrics_db['r2_score'].append(r2_score(self.actual,self.pred))
        metrics_db['mean_squared_error'].append(mean_squared_error(self.actual,self.pred))
        
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

    def create_decile(self,model_class=0, bins=10):
        '''
        Preparing the decile table
        '''
        valid_df = pd.DataFrame()
        valid_df['actual']=self.actual
        valid_df['p1']=self.pred
        if model_class == 0 :
            # Low probability to followed by high probability
            valid_df=valid_df.sort_values(by='p1', ascending=True) 
        else :
            # High probability followed by low probability
            valid_df=valid_df.sort_values(by='p1', ascending=False)
        # splitting the probablity into bins:
        split=np.array_split(valid_df, bins)

        col=['decile','count_actual_true','total_count_in_decile','Accuracy_in_decile',
         'Total_true_in_population','True_covered','min_confidence','max_confidence']

        df=pd.DataFrame(columns=col)
        total_count_in_decile=[]
        min_confidence=[]
        max_confidence=[]
        count_actual_true=[]
        #Total_true_in_population=[]
        for i in split:
            total_count_in_decile.append(len(i))
            min_confidence.append(min(i['p1']))
            max_confidence.append(max(i['p1']))
            if model_class == 0:
                count_actual_true.append(len(i[i['actual']==0]))
                # Plot only for the fail students 
            else :
                count_actual_true.append(len(i[i['actual']==1]))
                # Plot only for the pass students
        
        df['decile']=np.arange(1,11)
        df['total_count_in_decile']=total_count_in_decile
        df['min_confidence']=min_confidence
        df['max_confidence']=max_confidence
        df['count_actual_true']=count_actual_true
        df['Accuracy_in_decile']=(df['count_actual_true']/df['total_count_in_decile'])*100
        df['Total_true_in_population']=sum(df['count_actual_true'])
        df['cum_sum_count_actual_T']=df['count_actual_true'].cumsum()
        df['True_covered']=(df['cum_sum_count_actual_T']/df['Total_true_in_population'])*100
        df['Pop'] = (df['total_count_in_decile'].cumsum()/sum(df['total_count_in_decile']))*100
        df['lift']=df['True_covered']/df['Pop']
        return(df)


#########################################################################################################################
## This class is used to generate the graphs based on the metrics table, which was generated by the above evaluation   ##
## module. These plots can be exported to the local storage for our future reference.                                  ##
#########################################################################################################################

class evaluation_plots():
    
    def __init__(self,metrics_db,classification_metric = ['TP','FP','FN','TN','Accuracy',
               'Precision0','Precision1','recall0','recall1','f1','mcc','roc_auc']):
        '''
        metric_db - Should be passed as a pandas Dataframe
        model_id - Should be passed as a list
        '''
        self.classification_metric = classification_metric
        self.metrics_db = metrics_db
        self.hue_feat = 'Unique_ModelID'
        
        
    def metric_plots_1(self):
        '''
        Plotting the graphs for all|specified evaluation metrics
        '''                                  
        for i in self.classification_metric:
            sns.lineplot(x='Threshold', y=i,hue = self.hue_feat, markers=True, dashes=True, data=self.metrics_db)
            plt.show()

    def metric_plots_2(self,actual = None, pred=None, threshold = 0.5):
        '''
        TODO : due to package issue this function is not used (need to check)
        Plotting the roc Curve, Precision recall curve, confusion matirx
        '''
        if ((actual is not None) & (pred is not None)):                           
            bc = BinaryClassification(y_true = actual, y_pred = pred, labels=["Class 0", "Class 1"], threshold = threshold)
            # Figures
            plt.figure(figsize=(20,15))
            plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
            # Roc curve:
            bc.plot_roc_curve(threshold = threshold)
            plt.subplot2grid((2,6), (0,2), colspan=2)
            # precision recall curve:
            bc.plot_precision_recall_curve(threshold = threshold)
            plt.subplot2grid((2,6), (0,4), colspan=2)
            # class distribution curve:
            bc.plot_class_distribution(threshold = threshold)
            plt.subplot2grid((2,6), (1,1), colspan=2)
            # confusion matrix:
            bc.plot_confusion_matrix(threshold = threshold)
            plt.subplot2grid((2,6), (1,3), colspan=2)
            # normalised confusion matrix:
            bc.plot_confusion_matrix(threshold = threshold, normalize=True)
            plt.show()
            # Classification report:
            print(classification_report(actual, [0 if i<=threshold else 1 for i in pred]))

    def classification_report_chart(self, actual=None, pred=None, best_threshold = 0.5):
    	# printing the classification report:
    	print(classification_report(actual, [0 if i<=best_threshold else 1 for i in pred]))
    
    def roc_auc_plot(self, actual = None, pred = None):
	    #Preparing ROC curve
	    false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, pred)
	    roc_auc = auc(false_positive_rate, true_positive_rate)
	    #Plotting ROC Curve
	    plt.title('ROC CURVE')
	    plt.plot(false_positive_rate, true_positive_rate, 'b',
	    label='AUC = %0.2f'% roc_auc)
	    plt.legend(loc='lower right')
	    plt.plot([0,1],[0,1],'r--')
	    plt.xlim([-0.1,1.2])
	    plt.ylim([-0.1,1.2])
	    plt.ylabel('True_Positive_Rate')
	    plt.xlabel('False_Positive_Rate')
	    plt.show()
        
    def precision_recall_plot(self, actual = None, pred = None):
        precision, recall, _ = precision_recall_curve(actual, pred)
        average_precision = average_precision_score(actual, pred)
        # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
        step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
        plt.step(recall, precision, color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall curve: Avg Precision ={0:0.2f}'.format(average_precision))

    def confusion_matrix_plot(self,actual = None, pred = None, best_threshold = 0.5):
        '''
        Plotting the Confusion matrix based on the best threshold
        '''
        if ((actual is not None) & (pred is not None)):
        	pred_value = [1 if i >= best_threshold else 0 for i in pred]
        	cm = confusion_matrix(actual, pred_value)
        	plt.clf()
        	plt.imshow(cm, interpolation='nearest', cmap=plt.cm.gray_r)
        	classNames = ['Negative','Positive']
        	plt_title = "Confusion Matrix plot - Threshold ("+str(best_threshold)+")"
        	plt.title(plt_title)
        	plt.ylabel('True label')
        	plt.xlabel('Predicted label')
        	tick_marks = np.arange(len(classNames))
        	plt.xticks(tick_marks, classNames, rotation=45)
        	plt.yticks(tick_marks, classNames)
        	s = [['TN','FP'], ['FN', 'TP']]
        	for i in range(2):
        		for j in range(2):
        			plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]),bbox=dict(facecolor='white', alpha=0.5))
        	plt.show()

    def plot_decile(self,df=None,model_class=0):
        '''
        Plotting the decile chart
        '''
        plt.style.use('seaborn-pastel')
        f, (ax1) = plt.subplots(1,figsize=(14,7))
        #fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        if model_class == 0 :
            ax1.bar(x='Pop',height='Accuracy_in_decile',width=6,color='red',data=df,label="Accuracy_in_decile")
        else :
            ax1.bar(x='Pop',height='Accuracy_in_decile',width=6,color='green',data=df,label="Accuracy_in_decile")
            
        ax2.plot('Pop','True_covered',data=df,linestyle='-',color='black', marker='o',label="True_covered")
        for index, row in df.iterrows():
            if row.name < 3 :
                ax1.text((row.name + 0.9)*12 ,row['Accuracy_in_decile'], str(round(row['Accuracy_in_decile'],2))+'%',
                     color='black', ha="right",fontsize=10)
                ax2.text((row.name+0.6)*11,row['True_covered'], str(round(row['True_covered'],2))+'%', color='black', ha="center",fontsize=10)
            elif row.name >= 3 and row.name < 6 :
                ax1.text((row.name + 0.5)*12,row['Accuracy_in_decile'], str(round(row['Accuracy_in_decile'],2))+'%',
                     color='black', ha="right",fontsize=10)
                ax2.text((row.name+0.1)*11,row['True_covered'], str(round(row['True_covered'],2))+'%', color='black', ha="center",fontsize=10)
            else :
                ax1.text((row.name - 0.2)*12,row['Accuracy_in_decile'], str(round(row['Accuracy_in_decile'],2))+'%',
                     color='black', ha="right",fontsize=10)
                ax2.text((row.name-0.1)*11,row['True_covered'],str(round(row['True_covered'],2))+'%', color='black', ha="center",fontsize=10)
        ax2.grid(b=False)
        ax1.set_title('Lift Chart')
        ax1.set_ylabel('Accuracy in Decile')
        ax2.set_ylabel('True Covered')
        ax1.set_xlabel('Cumulative Population')
        plt.show()


    def lift_chart(self, data=None, X='decile', y='lift'):
        '''
        Plotting the lift chart against the deciles
        '''
        if data is not None:
            sns.lineplot(x=X, y=y, markers=True, dashes=True, data=data)
            plt.show()

#########################################################################################################################
## This is the main evaluator module - which runs the above two classes and saves the results based on the user        ##
## request:                                                                                                            ##
#########################################################################################################################

class ModelEvaluation(Evaluation,evaluation_plots):

    def __init__(self, actual, pred, threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
            model_reference_name = 'sample_model', model_type = 'classification',
        plot_classification_metric = ['TP','FP','FN','TN','Accuracy',
               'Precision0','Precision1','recall0','recall1','f1','mcc','roc_auc'],maximising_metrics='f1'):
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
        self.maximising_metrics = maximising_metrics

    def evaluate(self,evaluate_save=False, plots_show = True, model_class=0, bins=10):
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
        ## Getting the decide data:
        decile_db = evalu.create_decile(model_class=model_class, bins=bins)
        if evaluate_save:
            writer = pd.ExcelWriter('evaluation_results.xlsx', engine='xlsxwriter')
            metrics_db.to_excel(writer, sheet_name='Metrics_details')
            decile_db.to_excel(writer, sheet_name='decile_details')
            writer.save()
            print("The results are save to - ",os.getcwd()+'\\evaluation_results.xlsx')
        if plots_show:
            # Plotting the line chart for all the metrics:
            eval_plt = evaluation_plots(metrics_db = metrics_db, classification_metric = self.plot_classification_metric)
            eval_plt.metric_plots_1()
            # Plotting the confusion matrix | precision recall curve | roc curve | class distribution:
            best_threshold = metrics_db[metrics_db[self.maximising_metrics] == max(metrics_db[self.maximising_metrics])]['Threshold'].reset_index(drop=True)[0]
            #eval_plt.metric_plots_2(actual = self.actual, pred=self.pred, threshold = best_threshold)
            eval_plt.classification_report_chart(actual = self.actual, pred=self.pred, best_threshold = best_threshold)
            eval_plt.confusion_matrix_plot(actual = self.actual, pred=self.pred, best_threshold = best_threshold)
            eval_plt.roc_auc_plot(actual = self.actual, pred=self.pred)
            eval_plt.precision_recall_plot(actual = self.actual, pred=self.pred)
            # Plotting the decile chart:
            eval_plt.plot_decile(df=decile_db,model_class=model_class)
            # plotting the lift chart:
            eval_plt.lift_chart(data=decile_db, X='decile', y='lift')
            plt.close()
        return(metrics_db, decile_db, best_threshold, self.maximising_metrics)

    def Compare_models(self, evaluate_db = None ,model_id = None, comparison_metrics = None):
        '''
        Comparing the different models based on the model metrics - with the help of visulization plots
        '''
        if comparison_metrics:
            _metric = comparison_metrics
        else:
            _metric = self.plot_classification_metric
        if model_id:
            data = evaluate_db[evaluate_db['Unique_ModelID'].isin(model_id)]
            eval_plt = evaluation_plots(metrics_db = data, classification_metric = _metric)
            eval_plt.metric_plots()
            plt.close()


####################################################################################################################################
##                                             END - EVALUATION MODULE                                                            ##
####################################################################################################################################
