from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.dummy import DummyOperator
from airflow.operators.bash_operator import BashOperator
from airflow.utils.task_group import TaskGroup

pathScript = "/home/arnaldo/airflow/dags/avaliamusica"

default_args = {
   'owner': 'Arnaldo',
   'depends_on_past': False,
   'start_date': datetime(2021, 7, 25),
   'retries': 0,
   }

with DAG(
   'dag-pipeline-popula-feature-store',
   schedule_interval=timedelta(minutes=60),
   catchup=False,
   default_args=default_args
   ) as dag:

    start = DummyOperator(task_id="start")

    with TaskGroup("etl", tooltip="etl") as etl:
        
        t1 = BashOperator(
            dag=dag,
            task_id='ObtemUsers_MusCurtem',
            bash_command="""
            cd {0}
            python3 ObtemUsers_MusCurtem.py
            """.format(pathScript)
        )
        t2 = BashOperator(
            dag=dag,
            task_id='Obtem600kMus_Features',
            bash_command="""
            cd {0}
            python3 Obtem600kMus_Features.py
            """.format(pathScript)
        )
        t3 = BashOperator(
            dag=dag,
            task_id='ObtemUserA_MusCurte_MusNaoCurte',
            bash_command="""
            cd {0}
            python3 ObtemUserA_MusCurte_MusNaoCurte.py
            """.format(pathScript)
        )
        t4 = BashOperator(
            dag=dag,
            task_id='ObtemUserAMus_Features',
            bash_command="""
            cd {0}
            python3 ObtemUserAMus_Features.py
            """.format(pathScript)
        )
        t5 = BashOperator(
            dag=dag,
            task_id='Merge_UsersMusCurtem_UserA',
            bash_command="""
            cd {0}
            python3 Merge_UsersMusCurtem_UserA.py
            """.format(pathScript)
        )
        t6 = BashOperator(
            dag=dag,
            task_id='MontaFeatureStore',
            bash_command="""
            cd {0}
            mv ./'arquivos intermediarios'/UserA_MusCurte.pickle ./FeatureStore
            mv ./'arquivos intermediarios'/UserA_MusNaoCurte.pickle ./FeatureStore
            mv ./'arquivos intermediarios'/Users_MusCurtem.pickle ./FeatureStore
            mv 
            """.format(pathScript)
        )
        t3 >> t4
        t1 >> t5
        t3 >> t5
        
        


    end = DummyOperator(task_id='end')
    start >> etl >> end