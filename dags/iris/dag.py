import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from datetime import timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from airflow.hooks.mysql_hook import MySqlHook
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


dag = DAG(
    'iris_train',
    default_args={
        'owner': 'jm.kim',
        'email': ['jm.kim@xbrain.team'],
        'retries': 0,
        'retry_delay': timedelta(minutes=5),
        'start_date': days_ago(0, hour=1),
    },
    # schedule_interval="1 0 * * *",
    schedule_interval=None,
    concurrency=10
)


def scale():
    import os
    print(os.getcwd())
    iris = pd.read_csv("dags/iris/Iris.csv")

    with open('scaled_iris.pkl', 'wb') as f:
        s_col = [c for c in iris.columns if c != 'Species']

        scaler = StandardScaler()

        scaled_iris = pd.merge(
            pd.DataFrame(scaler.fit_transform(iris[s_col]), columns=s_col),
            iris['Species'].to_frame(),
            left_index=True,
            right_index=True
        )

        pickle.dump(scaled_iris, f)


def load_fs():
    with open('scaled_iris.pkl', 'rb') as f:
        scaled_iris = pickle.load(f)
        # hook = MySqlHook('local_fs')
        # hook.insert_rows(
        #     'iris',
        #     scaled_iris.values.tolist(),
        #     target_fields=[
        #         'SepalLengthCm',
        #         'SepalWidthCm',
        #         'PetalLengthCm',
        #         'PetalWidthCm',
        #         'Species',
        #     ]
        # )
        # save csv
        scaled_iris.to_csv('today_iris.csv', index=False, header=True)


def split():
    iris = pd.read_csv('today_iris.csv', header=0)

    train, test = train_test_split(iris, test_size=0.3)
    train_X = train[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    train_y = train['Species']
    test_X = test[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    test_y = test['Species']

    with open('train_X', 'wb') as f:
        pickle.dump(train_X, f)

    with open('train_y', 'wb') as f:
        pickle.dump(train_y, f)

    with open('test_X', 'wb') as f:
        pickle.dump(test_X, f)

    with open('test_y', 'wb') as f:
        pickle.dump(test_y, f)


def _load_tr_ts():
    import os
    print(os.listdir('.'))
    with open('train_X', 'rb') as f:
        train_X = pickle.load(f)

    with open('train_y', 'rb') as f:
        train_y = pickle.load(f)

    with open('test_X', 'rb') as f:
        test_X = pickle.load(f)

    with open('test_y', 'rb') as f:
        test_y = pickle.load(f)

    return train_X, train_y, test_X, test_y


def train_svc():
    train_X, train_y, test_X, test_y = _load_tr_ts()

    svc_model = svm.SVC()
    svc_model.fit(train_X, train_y)
    svc_prediction = svc_model.predict(test_X)
    svc_acc = metrics.accuracy_score(svc_prediction, test_y)

    return svc_acc


def train_lr():
    train_X, train_y, test_X, test_y = _load_tr_ts()

    lr_model = LogisticRegression()
    lr_model.fit(train_X, train_y)
    lr_prediction = lr_model.predict(test_X)
    lr_acc = metrics.accuracy_score(lr_prediction, test_y)

    return lr_acc


def train_dtc():
    train_X, train_y, test_X, test_y = _load_tr_ts()

    dtc_model = DecisionTreeClassifier()
    dtc_model.fit(train_X, train_y)
    dtc_prediction = dtc_model.predict(test_X)
    dtc_acc = metrics.accuracy_score(dtc_prediction, test_y)

    return dtc_acc


def train_knc():
    train_X, train_y, test_X, test_y = _load_tr_ts()

    knc_model = KNeighborsClassifier(n_neighbors=3)
    knc_model.fit(train_X, train_y)
    knc_prediction = knc_model.predict(test_X)
    knc_acc = metrics.accuracy_score(knc_prediction, test_y)

    return knc_acc


def select_best(**context):
    svc_acc = context['task_instance'].xcom_pull(task_ids='train_SVM')
    lr_acc = context['task_instance'].xcom_pull(task_ids='train_LogisticRegression')
    dtc_acc = context['task_instance'].xcom_pull(task_ids='train_DecisionTreeClassifier')
    knc_acc = context['task_instance'].xcom_pull(task_ids='train_KNeighborsClassifier')

    print(f"svc_acc : {svc_acc}\tlr_acc : {lr_acc}\tdtc_acc : {dtc_acc}\tknc_acc : {knc_acc}")


scale_op = PythonOperator(
    dag=dag,
    task_id='scale',
    python_callable=scale
)

load_fs_op = PythonOperator(
    dag=dag,
    task_id='load_Feature_store',
    python_callable=load_fs
)

split_op = PythonOperator(
    dag=dag,
    task_id='split_Train_Test',
    python_callable=split
)


train_svc_op = PythonOperator(
    dag=dag,
    task_id='train_SVM',
    python_callable=train_svc
)

train_lr_op = PythonOperator(
    dag=dag,
    task_id='train_LogisticRegression',
    python_callable=train_lr
)

train_dtc_op = PythonOperator(
    dag=dag,
    task_id='train_DecisionTreeClassifier',
    python_callable=train_dtc
)

train_knc_op = PythonOperator(
    dag=dag,
    task_id='train_KNeighborsClassifier',
    python_callable=train_knc
)

select_best_op = PythonOperator(
    dag=dag,
    task_id='select_Best_Model',
    provide_context=True,
    python_callable=select_best
)

load_fs_op << scale_op
load_fs_op >> split_op >> [train_svc_op, train_lr_op, train_dtc_op, train_knc_op]
select_best_op << [train_svc_op, train_lr_op, train_dtc_op, train_knc_op]