# 1. Use a lightweight Python base image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy only the requirements first (this makes future builds faster)
COPY requirements.txt .

# 4. Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your application code and the model
COPY . .

# 6. Expose the port the app runs on
EXPOSE 8000

# 7. The command to start the server when the container boots
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]