#!/bin/bash
echo "🚀 Запуск Smart Washer AI..."
cd backend
pip install -r requirements.txt -q
echo "✅ Зависимости установлены"
uvicorn main:app --host 0.0.0.0 --port 8000