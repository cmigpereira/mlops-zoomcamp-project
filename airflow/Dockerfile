# First-time build can take upto 10 mins.
FROM apache/airflow:latest-python3.9

RUN pip install -U pip --upgrade pip
RUN pip install pipenv 

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

USER root
RUN apt-get update -qq && apt-get install vim -qqq

SHELL ["/bin/bash", "-o", "pipefail", "-e", "-u", "-x", "-c"]

WORKDIR $AIRFLOW_HOME

USER $AIRFLOW_UID


RUN pip install -U pip --upgrade pip
RUN pip install pipenv 

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy