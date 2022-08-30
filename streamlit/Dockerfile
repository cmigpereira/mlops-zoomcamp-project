FROM python:3.9

RUN pip install -U pip
RUN pip install pipenv 

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "app.py", "app.py" ]

COPY imgs/* imgs/

EXPOSE 8501

ENTRYPOINT [ "streamlit", "run", "app.py"]