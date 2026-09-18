FROM python:3.13-slim

WORKDIR /app

# Instala dependencias primero (aprovecha el cache de Docker en rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia solo lo que la API necesita para correr (código + modelo entrenado)
COPY mlops_pipeline/ mlops_pipeline/

# Crea un usuario sin privilegios de administrador y corre la app con él
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "mlops_pipeline.src.model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]