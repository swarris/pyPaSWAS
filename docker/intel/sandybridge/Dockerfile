FROM nvidia/cuda:8.0-devel-ubuntu16.04

MAINTAINER Marcel Kempenaar (m.kempenaar@pl.hanze.nl)

## OpenCL dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
	rpm alien libnuma1 curl fakeroot libffi-dev clinfo && \
    rm -rf /var/lib/apt/lists/*

## Intel 2nd Generation OpenCL 1.2 support
RUN curl http://registrationcenter-download.intel.com/akdlm/irc_nas/9019/opencl_runtime_16.1.1_x64_ubuntu_6.4.0.25.tgz | tar xz

RUN cd opencl_runtime_16.1.1_x64_ubuntu_6.4.0.25/rpm && \
    fakeroot alien --to-deb opencl-1.2-base-6.4.0.25-1.x86_64.rpm && \
    fakeroot alien --to-deb opencl-1.2-intel-cpu-6.4.0.25-1.x86_64.rpm

RUN cd opencl_runtime_16.1.1_x64_ubuntu_6.4.0.25/rpm && \
    dpkg -i opencl-1.2-base_6.4.0.25-2_amd64.deb && \
    dpkg -i opencl-1.2-intel-cpu_6.4.0.25-2_amd64.deb && \
    rm -Rf /opencl_runtime_16.1.1_x64_ubuntu_6.4.0.25

RUN echo "/opt/intel/opencl-1.2-6.4.0.25/lib64/clinfo" > /etc/ld.so.conf.d/intelOpenCL.conf

RUN mkdir -p /etc/OpenCL/vendors && \
    ln /opt/intel/opencl-1.2-6.4.0.25/etc/intel64.icd /etc/OpenCL/vendors/intel64.icd && \
    ldconfig

ENV PATH /usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/cuda/lib:/usr/local/cuda/lib64

## Python3 and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip python3-setuptools git opencl-headers \
    autoconf libtool pkg-config && \
    ln -s /usr/bin/python3 /usr/bin/python

RUN ln -s /usr/local/cuda/lib64/libOpenCL* /usr/lib/ && \
    pip3 install --upgrade pip

RUN pip3 install wheel

RUN pip3 install numpy

RUN pip3 install biopython

RUN export PATH=/usr/local/cuda/bin:$PATH && pip3 install pycuda

## Custom pyOpenCL installation forcing the use of version 1.2
RUN export PATH=/usr/local/cuda/bin:$PATH && \
    export LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/lib64 && \
    export LDFLAGS=-L/usr/local/cuda/lib64 && \
    git clone https://github.com/pyopencl/pyopencl.git && \
    cd pyopencl && python3 configure.py && \
    echo 'CL_PRETEND_VERSION = "1.2"' >> siteconf.py && \
    pip3 install .

## pyPaSWAS installation
RUN git clone https://github.com/swarris/pyPaSWAS.git /root/pyPaSWAS
