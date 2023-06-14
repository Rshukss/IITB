# Use the NVIDIA CUDA runtime as the base image
#FROM nvidia/cuda:11.0-runtime
FROM nvidia/cuda:12.1.0-runtime-ubuntu20.04
# Set the working directory to /app
# WORKDIR /app


# Copy the Flask app files to the container
RUN mkdir new_folder
ADD app.py new_folder
# COPY requirements.txt .

# Install dependencies
RUN apt-get update
RUN apt-get install -y python3 python3-pip
RUN python3 --version
RUN pip3 --version
RUN pip --version
# RUN python -m pip install -U --force-reinstall pip
# RUN pip3 install setuptools-rust
# RUN pip3 install --upgrade pip
# RUN pip --version
# RUN pip3 --version
# # RUN which pip
# # RUN which pip3
#RUN pip3 install timm
RUN pip3 install pipreqs
# RUN pip3 install setuptools123456

RUN pipreqs --force --mode no-pin new_folder
# RUN ls
RUN ls new_folder
RUN pip3 install -r new_folder/requirements.txt


# Expose port 5000 for the Flask app
EXPOSE 5124

# Start the Flask app when the container starts
CMD ["python3", "new_folder/app.py"]

