# pyPaSWAS Docker Containers

This folder contains the Docker files for building Containers containing the `pyPaSWAS` software. These containers are based on Ubuntu 16.04 and come supplied with Python3 and the Nvidia CUDA software as they are based on the `nvidia/cuda:8.0-devel-ubuntu16.04` container image [supplied by Nvidia](https://hub.docker.com/r/nvidia/cuda/).

## Running existing Docker Containers

The Docker engine is required for running the container, see their [excellent installation instructions](https://docs.docker.com/engine/installation/) for further details.
Next, these containers require low-level access to the hardware (i.e. the GPU) and therefore the use of the `nvidia-docker` utility, installation instructions are available on its [github page](https://github.com/NVIDIA/nvidia-docker/tree/2.0). 

`nvidia-docker run --rm -ti mkempenaar/pypaswas:nvidia-opencl_cuda8.0 bash` will download the container, start and attach to a bash session running inside the container. Here you will find the software at `/root/pyPasWAS`. Running the performance tests on a clean container is as simple as (note: this will take a while):

```
cd /root/pyPaSWAS
sh data/runPerformanceTests.sh
```

* ## Container(s) available on [Docker Hub](https://hub.docker.com/r/mkempenaar/pypaswas/)

    **`mkempenaar/pypaswas:nvidia-opencl_cuda8.0` [*Docker file*](https://raw.githubusercontent.com/swarris/pyPaSWAS/master/data/docker/nvidia/Dockerfile)**

    This container can be used for testing all availabilities of the `pyPaSWAS` sequence aligner as it contains the Intel and Nvidia OpenCL runtime libraries and Nvidia CUDA support.


## Building custom Docker Containers

As most hardware manufacturers have their own acceleration libraries (multiple versions of OpenCL, Nvidia CUDA, etc.) the available containers might not work for your hardware. Therefore, a few custom build files are available depending on your hardware and requirements (i.e. only CUDA support or only Intel OpenCL). 

### Downloading and Building

Cloning this repository gives the currently available Dockerfiles for building custom images which can be found in the `pyPaSWAS/data/docker` folder. Building a container locally can be done by going to the folder of choice (each contains a single `Dockerfile`; a container description) and running:

```
docker build -t pypaswas:custom .
```

Currently available:

* [Intel OpenCL + Nvidia CUDA](https://raw.githubusercontent.com/swarris/pyPaSWAS/master/data/docker/intel/Dockerfile), `pyPaSWAS/data/docker/intel/Dockerfile`: Suitable for Intel Core and Xeon CPUs and GPUs from the 3rd generation (Ivy Bridge) and newer, combined with Nvidia CUDA from the base container image.
* [Intel OpenCL + Nvidia CUDA](https://raw.githubusercontent.com/swarris/pyPaSWAS/master/data/docker/intel/sandybridge/Dockerfile), `pyPaSWAS/data/docker/intel/sandybridge/Dockerfile`: Only suitable for 2nd generation (Sandy Bridge) Intel Core CPUs, combined with Nvidia CUDA from the base container image.
* [Intel OpenCL + Nvidia OpenCL + Nvidia CUDA](https://raw.githubusercontent.com/swarris/pyPaSWAS/master/data/docker/nvidia/Dockerfile), `pyPaSWAS/data/docker/nvidia/Dockerfile`: Full package for 3rd generation and newer Intel Core and Xeon CPUs and GPUs, combined with Nvidia OpenCL and CUDA support.