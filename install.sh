#!/bin/bash
#
# Installation script for pyPaSWAS (trac.nbic.nl/pypaswas)
# Run with 'sh install.sh'
#
# Author: M Kempenaar (marcel.kempenaar@nbic.nl)
# Date: 24/01/2013
#
# Script will exit if one of the commands will fail
set -e

# Version to install
PYPASWAS="pyPaSWAS-0.1.0"

# Required software versions and modules
PYTHON_VERSION=2.6
NV_DRIVER_VERSION=270
CUDA_NVCC_VERSION=3.2
GCC_VERSION=4.6

ECHO=`which echo`
PRINTF=`which printf`

# Transform the required version string into a number that can be used in comparisons
REQUIRED_PYTHON_VERSION=`$ECHO $PYTHON_VERSION | sed -e 's;\.;0;g'`
REQUIRED_NV_DRIVER_VERSION=`$ECHO $NV_DRIVER_VERSION | sed -e 's;\.;0;g'`
REQUIRED_NVCC_VERSION=`$ECHO $CUDA_NVCC_VERSION | sed -e 's;\.;0;g'`
REQUIRED_GCC_VERSION=`$ECHO $GCC_VERSION | sed -e 's;\.;0;g'`

$PRINTF "Performing pyPaSWAS requirement checks\n"
$PRINTF "\tPython version check...\n"
PYTHON_EXE=`which python`
if [ $PYTHON_EXE ]; then
    $PYTHON_EXE -V 2> tmp.ver
    VERSION=`cat tmp.ver | awk '{ print substr($2, 1); }'`
    REAL_PYTHON_VERSION=$VERSION
    VERSION=`$ECHO $VERSION | awk '{ print substr($1, 1, 3); }' | sed -e 's;\.;0;g'`
    if [ $VERSION ]; then
        if [ $VERSION -ge $REQUIRED_PYTHON_VERSION ]; then
            $PRINTF "\t\tPython is up to date ($REAL_PYTHON_VERSION)\n"
        else
            $PRINTF "\t\tPython is outdated (required: $PYTHON_VERSION.*, present: $REAL_PYTHON_VERSION), aborting\n"
            exit
        fi
    fi
else
    $PRINTF "\t\tCannot find Python executable, aborting\n"
    exit
fi
rm tmp.ver

# CUDA Download / install link:
cuda_download(){
    $PRINTF "\nError: cannot complete installation due to missing components:\n"
    $PRINTF "\tCheck if your system contains a supported GPU by going to: https://developer.nvidia.com/cuda-gpus\n"
    $PRINTF "\tnoting that the required 'Compute Capability' => 1.2\n"
    $PRINTF "\n\tPlease go to: https://developer.nvidia.com/cuda-downloads and download the appropriate file\n"
    $PRINTF "\tfor your system. For Linux, this download is an executable file containing an updated driver,\n"
    $PRINTF "\tthe CUDA Toolskit and CUDA SDK. Rerun this file after installing the CUDA components.\n"
    $PRINTF "\tFor a Driver only download, go to: http://www.nvidia.com/Download/index.aspx\n"

    # CUDA is incompatible with certain gcc versions, this checks to see if the correct one is present
    GCC_EXE=`which gcc`
    if [ $GCC_EXE ]; then
        $GCC_EXE --version > tmp_gcc.ver
        VERSION=`head -n 1 tmp_gcc.ver | awk '{print substr($4, 1, 3); }'`
        REAL_GCC_VERSION=$VERSION
        VERSION=`$ECHO $VERSION | sed -e 's;\.;0;g'`
        if [ $VERSION ]; then
            if [ $VERSION -gt $REQUIRED_GCC_VERSION ]; then
               $PRINTF "\tThe 'gcc' compiler detected has a version >= $GCC_VERSION which is incompatible\n"
               $PRINTF "\twith CUDA. Please install a gcc compiler with a lower version.\n"
            fi
        fi
    fi
    if [ -f tmp.ver ]; then
        rm tmp.ver
    fi
    exit
}

# Check if CUDA is installed by inspecting the required LD_LIBRARY_PATH environment variable
# TODO: actually do something with this or remove complete check
$PRINTF "\tChecking for CUDA installation\n"
if [ "$LD_LIBRARY_PATH" ]; then
    _LD_LIBRARY_PATH=`$ECHO $LD_LIBRARY_PATH | grep cuda`
    if [ $_LD_LIBRARY_PATH ]; then
        $PRINTF "\t\tCUDA reference found in LD_LIBRARY_PATH, assuming CUDA is installed\n"
    else
        $PRINTF "\t\tNo CUDA references found in environment variables, continuing checks..\n"
  fi
else
    $PRINTF "\t\tNo LD_LIBRARY_PATH, assuming CUDA is not installed, continuing checks..\n"
fi

# TODO: check for driver version on OSX
$PRINTF "\tNVIDIA CUDA Driver version check...\n"
DRIVER_LOCATION=/proc/driver/nvidia/version
if [ -f $DRIVER_LOCATION ]; then
    VERSION=$(head -n 1 $DRIVER_LOCATION)
    VERSION=`$ECHO $VERSION | awk '{ print substr($8, 1, 6); }'`
    REAL_DRIVER_VERSION=$VERSION
    VERSION=`$ECHO $VERSION | awk '{ print substr($1, 1, 3); }' | sed -e 's;\.;0;g'`
    if [ $VERSION ]; then
        if [ $VERSION -ge $REQUIRED_NV_DRIVER_VERSION ]; then
            $PRINTF "\t\tCUDA Driver is up to date ($REAL_DRIVER_VERSION)\n"
        else
            $PRINTF "\t\tCUDA Driver is outdated (required: $NV_DRIVER_VERSION.*, present: $REAL_DRIVER_VERSION), aborting\n"
            exit
        fi
    fi
else
    $PRINTF "\t\tCannot find the NVIDIA CUDA driver\n"
    cuda_download
fi

$PRINTF "\tNVIDIA Compiler version check...\n"
NVCC_EXE=`which nvcc`
if [ $NVCC_EXE ]; then
    $NVCC_EXE -V > tmp.ver
    VERSION=`cat tmp.ver | grep release | awk '{ print substr($5, 1, 3); }'`
    REAL_NVCC_VERSION=$VERSION
    VERSION=`$ECHO $VERSION | sed -e 's;\.;0;g'`
    if [ $VERSION ]; then
        if [ $VERSION -ge $REQUIRED_NVCC_VERSION ]; then
            $PRINTF "\t\tNVIDIA Compiler (nvcc) is up to date ($REAL_NVCC_VERSION)\n"
        else
            $PRINTF "\t\tNVIDIA Compiler (nvcc) is outdated (required: $CUDA_NVCC_VERSION, present: $REAL_NVCC_VERSION), aborting\n"
            cuda_download
        fi
    fi
else
    $PRINTF "\t\tCannot find the NVIDIA Compiler (nvcc), aborting\n"
    cuda_download
fi
rm tmp.ver

# Python pip presence test
$ECHO -e "try:\n\timport pip\n\tprint '1'\nexcept:\n\tprint '0'\n" > test.py
$PRINTF "\tChecking if Python pip is installed...\n"
$PYTHON_EXE test.py 1> tmp.ver
VERSION=`cat tmp.ver`
PIP=0
if [ $VERSION -eq '1' ]; then
    $PRINTF "\t\tpip is available\n"
    PIP=1
else
    # Even though pip is not present, maybe the requirements are already there?
    $ECHO -e "try:\n\timport numpy" > test.py
    $ECHO -e "\timport pycuda.driver as driver" >> test.py
    $ECHO -e "\tfrom Bio.Seq import Seq\n\tprint'1'\nexcept:\n\tprint '0'\n" >> test.py
    $PYTHON_EXE test.py 1> tmp.ver
    VERSION=`cat tmp.ver`
    if [ $VERSION -eq '1' ]; then
        $PRINTF "\t\tPython pip is unavailable, but the required packages can be loaded, continuing..\n"
    else
        $PRINTF "\t\tPython pip cannot be found and we are missing required Python packages, aborting\n"
        $PRINTF "\t\tPlease install Python pip (https://pypi.python.org/pypi/pip) before continuing.\n"
        rm test.py tmp.ver
        exit
    fi
fi
rm test.py tmp.ver

#TEMP
PIP=0

# TODO: open pypaswas download location
# Download pypaswas if not present and install
#wget https://brs.nbic.nl/ci/job/pypaswas/lastSuccessfulBuild/artifact/dist/pyPaSWAS-0.1.0.tar.gz -O pypaswas_latest.zip
# Finally, install Python packages and pyPaSWAS
if [ $1 ]; then
    p_install=$1
else
    $PRINTF "\n\tShould pyPaSWAS be installed as a loadable module? Only use when loading part of the project in Python, Y(es) or N(o): "
    read p_install
fi

if [ "$p_install" = "Y" ]; then
    if [ $PIP -eq 0 ]; then
	# Unpack and run setup.py to install
	tar -xf $PYPASWAS.tar.gz
	cd $PYPASWAS
	python setup.py install
    else
	pip install $PYPASWAS.tar.gz -r requirements.txt
    fi
else
    $PRINTF "\n\tPlease specify the path where pyPaSWAS should be installed (extracted): "
    read p_install_path
    if [ $PIP -eq 1 ]; then
	pip install -r requirements.txt
    fi
    tar -C $p_install_path -xf $PYPASWAS.tar.gz
    $PRINTF "\t\tExtracted.."
fi

# Test if pyCUDA module is installed correctly by trying an import
# Create small Python test script for loading the required module
$ECHO -e "try:\n\timport pycuda.driver as drv\n\tprint '1'\nexcept:\n\tprint '0'\n" > test.py

$PRINTF "\tChecking if pyCUDA can be loaded...\n"
$PYTHON_EXE test.py 1> tmp.ver
VERSION=`cat tmp.ver`
if [ $VERSION -eq '1' ]; then
    $PRINTF "\t\tpyCUDA is correctly installed\n"
else
    $PRINTF "\t\tpyCUDA module failed to install, aborting\n"
    rm test.py tmp.ver
    exit
fi
rm test.py tmp.ver
