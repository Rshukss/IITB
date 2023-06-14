# Use the NVIDIA CUDA runtime as the base image
#FROM nvidia/cuda:11.0-runtime
FROM nvidia/cuda:12.1.0-runtime-ubuntu20.04
# Set the working directory to /app
# WORKDIR /app
RUN mkdir new_folder

# Copy the Flask app files to the container

ADD app.py new_folder
# COPY requirements.txt .

# Install dependencies
RUN apt-get update
RUN apt-get install -y python3 python3-pip
RUN pip3 install timm
RUN pip3 install pipreqs
RUN pipreqs --force --mode no-pin new_folder
RUN pip3 install -r new_folder/requirements.txt


# Expose port 5000 for the Flask app
EXPOSE 5127

# Start the Flask app when the container starts
CMD ["python3", "new_folder/app.py"]
