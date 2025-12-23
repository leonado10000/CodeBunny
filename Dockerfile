FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy our application source code
COPY src/ /app/src/
COPY .env /app/.env
# Expose the port the app runs on
EXPOSE 8080

# Define the command to run your app
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]