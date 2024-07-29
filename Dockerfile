FROM python:3.11-slim

RUN pip install --upgrade pip
RUN pip install pipenv 

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

#ENV AWS_ACCESS_KEY_ID=YOUR_AWS_ACCESS_KEY_ID
#ENV AWS_SECRET_ACCESS_KEY=YOUR_AWS_SECRET_ACCESS_KEY

COPY set_env.sh /app/set_env.sh

RUN chmod +x /app/set_env.sh
RUN /app/set_env.sh

RUN pipenv install --system --deploy

COPY [ "src/predict.py", "./" ]

EXPOSE 9696
EXPOSE 5000
EXPOSE 4200

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]