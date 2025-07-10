FROM python:3.12-alpine

# Install system dependencies
RUN apk add --no-cache git gcc musl-dev python3-dev libffi-dev openssl-dev

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose port (will be overridden by cloud platforms)
EXPOSE 8000

# Set default environment variables
ENV YOUTRACK_CLOUD=false
ENV PORT=8000

# Run the server
CMD python main.py --transport http --host 0.0.0.0 --port ${PORT}