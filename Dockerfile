# Dockerfile

# Set the base image to Python 3.10
FROM python:3.10

# Set the working directory within the container
WORKDIR /app

# Upgrade pip
RUN pip3 install --upgrade pip

# Copy the requirements file and install the dependencies
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Copy the required files into the container
COPY serve_ml_duplicates_classifier.py /app/
COPY best_model.sav /app/
COPY db.db /app/

# Expose ports 8000 for external access
EXPOSE 8000

# Define the command to run when the container starts
CMD ray start --head --disable-usage-stats ; \
 serve build serve_ml_duplicates_classifier:duplicate_classifier_app -o serve_config_ori.yaml ; \
 sed 's/runtime_env: {}/runtime_env: {pip: [torch, transformers]}/g' serve_config_ori.yaml > serve_config.yaml ; \
 serve run serve_config.yaml
