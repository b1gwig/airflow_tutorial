from puckel/docker-airflow:latest

USER root
RUN apt-get update
RUN apt-get install -y vim

USER airflow

RUN cd ~
ADD ./requirements.txt .
RUN pip install -r requirements.txt