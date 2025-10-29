FROM python:3.10-slim
WORKDIR /app
COPY . /appRUN apt-get update && apt-get install -y build-essential
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "5000"]