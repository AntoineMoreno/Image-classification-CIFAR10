# Use a base image with Python
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the necessary files into the container
COPY requirements.txt .

# Install the required Python libraries
RUN pip install--no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

# Run the Python script when the container starts
CMD ["python", "app.py"]

