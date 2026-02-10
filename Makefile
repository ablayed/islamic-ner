.PHONY: install download train evaluate serve demo test clean

install:
		pip install -r requirements.txt

download:
		python scripts/download_data.py

train:
		python -m src.ner.model --train

evaluate:
		python -m src.evaluation.metrics

serve:
		uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

demo:
		streamlit run demo/app.py

test:
		pytest tests/ -v

clean:
		find . -type d -name __pycache__ -exec rm -rf {} +
		find . -type f -name "*.pyc" -delete
