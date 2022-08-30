FROM python:3.9

RUN pip install -U pip --upgrade pip
RUN pip install pipenv

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "service_tests.py", "service_tests.py" ]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "service_tests:app" ]