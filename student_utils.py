from sklearn.metrics import roc_curve, precision_recall_curve, accuracy_score, f1_score, classification_report, roc_auc_score, confusion_matrix, precision_score, recall_score
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import functools
sns.set_style('darkgrid')


####### STUDENTS FILL THIS OUT ######
# Question 1
def df_level_test(ehr_df):
    if len(ehr_df) > ehr_df['encounter_id'].nunique():
        print('EHR dataset could be at the line level.')
    elif len(ehr_df) == ehr_df['encounter_id'].nunique():
        print('EHR dataset could be at the encounter level.')
    else:
        print('EHR dataset could be at longitudinal level.')
    return

# Question 2
def num_col_dist(df):
    num_col = df.select_dtypes(include='number')
    ncol = int(np.sqrt(len(num_col.columns)))
    nrows = int(len(num_col.columns)/ncol)
    fig, axes = plt.subplots(ncols=ncol, nrows=nrows, figsize=(ncol*5, 15))
    for col, ax in zip(num_col, axes.flatten()):
        g = sns.distplot(num_col[col], ax=ax)
        g.set(title=col, xlabel=None)
    return


def cardinality_count(df, top=None):
    schema_path = '../data_schema_references/project_data_schema.csv'
    schema_df = pd.read_csv(schema_path)
    categorical_col = [col for col in schema_df[schema_df.Type.str.contains(
        'categorical')]['Feature Name\n']]
    col_val = [{'col_name': col, 'num_unique': df[col].nunique()}
               for col in categorical_col]
    col_val = pd.DataFrame(col_val).sort_values('num_unique', ascending=False)
    if top:
        return col_val.head(top)
    else:
        return col_val


def demo_plot(df, keys, rotation=45, save_fig=False):
    fig, axes = plt.subplots(ncols=len(keys), figsize=(15, 8))
    for k, ax in zip(keys, axes.flatten()):
        g = sns.countplot(x=k, data=df, ax=ax, hue='gender')
        g.set(title=k)
        for rect in ax.patches:
            ax.text(rect.get_x()+rect.get_width()/2,
                    rect.get_height()+0.75,
                    rect.get_height(),
                    ha='center',
                    va='baseline',
                    rotation=rotation,
                    fontdict={'color': 'gray', 'size': 9})
    fig.autofmt_xdate()
    if save_fig == True:
        fig.save_fig('demographics.png')
    else:
        return


# Question 3
def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''
    code_ck = ndc_df.set_index('NDC_Code')['Non-proprietary Name']
    df['generic_drug_name'] = df.ndc_code.replace(code_ck)
    print('The dimensionality of NDC_Code is reduced from {} to {}.'.format(
        df.ndc_code.nunique(), df.generic_drug_name.nunique()))
    return df

# Question 4
def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first 
            encounter for a given patient.
    '''
    df = df.sort_values('encounter_id')
    first_encounter_values = df.groupby(
        'patient_nbr')['encounter_id'].head(1).values
    return df[df['encounter_id'].isin(first_encounter_values)]

# Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr', val_size=0.2, test_size=0.2):
    df = df.iloc[np.random.permutation(len(df))]

    unique_patients = df[patient_key].unique()
    total_unique_patients = len(unique_patients)

    train_size = round(total_unique_patients * (1-test_size-val_size))
    val_size = round(total_unique_patients*val_size)
    test_size = round(total_unique_patients*test_size)
    train_df = df[df[patient_key].isin(
        unique_patients[:train_size])].reset_index(drop=True)
    val_df = df[df[patient_key].isin(
        unique_patients[train_size:train_size+val_size])].reset_index(drop=True)
    test_df = df[df[patient_key].isin(
        unique_patients[-test_size+1:])].reset_index(drop=True)

    print('Dataset is splited into train ({}, {:.2%}), validation ({}, {:.2%}), and test ({}, {:.2%})'.format(
        len(train_df), len(train_df)/len(df), len(val_df), len(val_df)/len(df), len(test_df), len(test_df)/len(df)))

    print("Training partition has a shape = ", train_df.shape)
    print("Validation partition has a shape = ", val_df.shape)
    print("Test partition has a shape = ", test_df.shape)

    try:
        assert len(set.intersection(set(train_df[patient_key].unique()),
                                    set(val_df[patient_key].unique()),
                                    set(test_df[patient_key].unique()))) == 0
        print('No overlapping among partitions.')
    except:
        print('Detected overlapping among different partitions!')

    return train_df, val_df, test_df

# Question 7
def create_tf_categorical_feature_cols(categorical_col_list,
                                       df,
                                       dims=10,
                                       vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed 
    with TF feature column vocab_dir: string, the path where the vocabulary text
    files are located. 
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        cat_col = tf.feature_column.categorical_column_with_vocabulary_file(
            key=c, vocabulary_file=vocab_file_path, num_oov_buckets=1)
        if df[c].nunique() > 100:
            tf_categorical_feature_column = tf.feature_column.embedding_column(
                cat_col, dimension=dims)
        else:
            tf_categorical_feature_column = tf.feature_column.indicator_column(
                cat_col)
        output_tf_list.append(tf_categorical_feature_column)
    return output_tf_list

# Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for 
        normalization
    '''
    return (col - mean)/std


def create_tf_numeric_feature(col, mean, std, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field
    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    normalizer = functools.partial(
        normalize_numeric_with_zscore, mean=mean, std=std)
    tf_numeric_feature = tf.feature_column.numeric_column(
        key=col, default_value=default_value, normalizer_fn=normalizer, dtype=tf.float64)

    return tf_numeric_feature

# Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    return m, s

# Question 10
def get_student_binary_prediction(df, col, threshold=5):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to
            flattened numpy array and binary labels
    '''
    student_binary_prediction = (
        df[col] >= threshold).replace({True: 1, False: 0})
    return student_binary_prediction

# Metric Plots
def plot_history(history):
    N = len(history.history["loss"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.plot(np.arange(0, N), history.history["loss"], label="train_loss")
    ax1.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
    ax2.plot(np.arange(0, N), history.history["mse"], label="mse")
    ax2.plot(np.arange(0, N), history.history["val_mse"], label="val_mse")
    ax1.set(title="Training and Validation Loss",
            xlabel="Epoch #",
            ylabel="Loss")
    ax1.legend(loc="best")
    ax2.set(title="Training and Validation Accuracy",
            xlabel="Epoch #",
            ylabel="MSE")
    ax2.legend(loc="best")
    fig.savefig(f'training_performance.png')
    return


def plot_auc(labels, score):

    fpr, tpr, _ = roc_curve(labels, score)
    auc = roc_auc_score(labels, score)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label='(AUC: {:.3f})'.format(auc))
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2,
            color='r', label='Chance', alpha=.8)
    ax.set(xlabel='False Positive Rate',
           ylabel='True Positive Rate',
           )
    plt.show()
    fig.savefig(f'model_ROC_Curve.png')
    return auc


def calf1(precision, recall):
    return 2*(precision*recall)/(precision+recall) if precision and recall else 0


def plot_precision_recall(labels, score):
    precision, recall, thresholds = precision_recall_curve(
        labels, score, pos_label=1)
    f1 = [calf1(precision[i], recall[i]) for i in range(len(thresholds))]
    ind = np.argmax(f1)  # index of the maximum f1 score
    print('Max F1 score is {:.3f}\nThreshold={:.3f}\nPrecision={:.3f}\nRecall={:.3f} '.format(
        f1[ind], thresholds[ind], precision[ind], recall[ind]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.plot(recall[:-1], precision[:-1], color='red', lw=2)
    ax1.set(xlabel='recall',
            ylabel='precision',
            title='Precision-Recall Curve')
    ax2.plot(thresholds, f1)
    ax2.set(xlabel='thresholds',
            ylabel='F1 score',
            title='F1_Score vs. Threshold')
    fig.savefig(f'model_precision_recall.png')

    return
