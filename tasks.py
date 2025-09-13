from celery import Celery
from train_models import train_and_save
import os

broker = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
backend = os.environ.get("CELERY_RESULT_BACKEND", broker)

celery = Celery("tasks", broker=broker, backend=backend)

@celery.task(bind=True)
def train_models_task(self, n_estimators=100):
    try:
        train_and_save(n_estimators=n_estimators)
        return {"status": "success", "detail": "Models trained"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
