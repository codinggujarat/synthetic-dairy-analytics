# Run with Docker Compose (recommended)

# Build & start containers
docker compose up --build

# App will be available at http://localhost:5000
# Worker will start (Celery). Redis is available at redis://redis:6379

# To trigger async training via the UI:
# - log in
# - click Retrain (calls /train_async), or call the endpoint:
# curl http://localhost:5000/train_async

# To run locally without Docker:
# - create venv, install requirements.txt
# - generate data: python data_generator.py
# - train models: python train_models.py
# - run app: python app.py
