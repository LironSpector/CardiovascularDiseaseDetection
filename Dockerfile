# Use the official Python image as the base image
FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy the application files into the working directory
COPY . /app

# Install the application dependencies
RUN pip install -r requirements.txt

EXPOSE 1337

# Use Gunicorn as the WSGI server to serve the Flask application
CMD ["python", "main.py"]
