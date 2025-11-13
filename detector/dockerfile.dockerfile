FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY app/ /app/
EXPOSE 5000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "5000"]