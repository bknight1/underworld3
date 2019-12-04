FROM debian:latest as base_runtime
ENV LANG=C.UTF-8
# Setup some things in anticipation of virtualenvs
ENV VIRTUAL_ENV=/opt/venv
# The following ensures that the venv takes precedence if available
ENV PATH=${VIRTUAL_ENV}/bin:$PATH
# The following ensures that system packages are visible to venv Python, and vice versa
ENV PYTHONPATH=/usr/lib/python3.7/site-packages:${VIRTUAL_ENV}/lib/python3.7/site-packages
# install runtime requirements
RUN apt-get update -qq \
&&  DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
        bash-completion \
        python3-minimal \
        python3-virtualenv \
        python3-pip \
        libopenblas-base
RUN pip3 install -U setuptools  \
&&  pip3 install --no-cache-dir \
        packaging \
        appdirs \
        numpy \
        jupyter \
        jupytext \
        ipyparallel 

FROM base_runtime AS build_base
# install build requirements
RUN apt-get update -qq 
RUN DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
        build-essential \
        cmake \
        wget \
        gfortran \
        python3-dev \
        libopenblas-dev \
        libz-dev \ 
        less \ 
        vim \
        valgrind valgrind-dbg valgrind-mpi
# build mpi
WORKDIR /tmp/mpich-build
ARG MPICH_VERSION="3.1.4"
RUN wget http://www.mpich.org/static/downloads/${MPICH_VERSION}/mpich-${MPICH_VERSION}.tar.gz 
RUN  tar xvzf mpich-${MPICH_VERSION}.tar.gz 
WORKDIR /tmp/mpich-build/mpich-${MPICH_VERSION}
ARG MPICH_CONFIGURE_OPTIONS="--prefix=/usr/local --enable-g=option=all --disable-fast"
# ARG MPICH_CONFIGURE_OPTIONS="--prefix=/usr/local --enable-g=option=none --disable-debuginfo --enable-fast=O3,ndebug --without-timing --without-mpit-pvars"
ARG MPICH_MAKE_OPTIONS="-j8"
RUN ./configure ${MPICH_CONFIGURE_OPTIONS} 
RUN make ${MPICH_MAKE_OPTIONS}             
RUN make install                           
RUN ldconfig
# create venv now for forthcoming python packages
RUN /usr/bin/python3 -m virtualenv --python=/usr/bin/python3 ${VIRTUAL_ENV} 
RUN pip3 install --no-cache-dir mpi4py
# build petsc
WORKDIR /tmp/petsc-build

ENV PETSC_BRANCH=v3.12
RUN DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends git
RUN git clone --single-branch --branch ${PETSC_BRANCH} https://gitlab.com/petsc/petsc/
WORKDIR /tmp/petsc-build/petsc
RUN python3 ./configure --with-debugging=1 --prefix=/usr/local \
                --with-zlib=1                   \
                --download-mumps=1              \
                --download-parmetis=1           \
                --download-metis=1              \
                --download-scalapack=1          \
                --useThreads=0                  \
                --with-shared-libraries         \
                --with-cxx-dialect=C++11        \
                --download-hdf5=1               \
                --with-make-np=8                \
                --download-ctetgen              \
                --download-eigen                \
                --download-triangle
RUN make PETSC_DIR=/tmp/petsc-build/petsc PETSC_ARCH=arch-linux-c-debug all
RUN make PETSC_DIR=/tmp/petsc-build/petsc PETSC_ARCH=arch-linux-c-debug install
ENV PETSC_DIR=/usr/local
ENV PETSC_ARCH=arch-linux-c-debug
RUN pip3 install cython
RUN pip3 install --no-cache-dir petsc4py
RUN CC=h5pcc HDF5_MPI="ON" HDF5_DIR=${PETSC_DIR} pip3 install --no-cache-dir --no-binary=h5py h5py
ENV LD_LIBRARY_PATH=/usr/local/lib

# # FROM base_runtime AS minimal
# # COPY --from=build_base $VIRTUAL_ENV $VIRTUAL_ENV
# # COPY --from=build_base /usr/local /usr/local
# # # Record Python packages, but only record system packages! 
# # # Not venv packages, which will be copied directly in.
# # RUN PYTHONPATH= /usr/bin/pip3 freeze >/opt/requirements.txt
# # # Record manually install apt packages.
# # RUN apt-mark showmanual >/opt/installed.txt

# # # Add user environment
# # FROM minimal
# # EXPOSE 8888
# # RUN ipcluster nbextension enable
# # ADD https://github.com/krallin/tini/releases/download/v0.18.0/tini /tini
# # RUN chmod +x /tini
# # ENV NB_USER jovyan
# # RUN useradd -m -s /bin/bash -N jovyan
# # USER $NB_USER
# # ENV NB_WORK /home/$NB_USER
# # RUN ipython profile create --parallel --profile=mpi && \
# #     echo "c.IPClusterEngines.engine_launcher_class = 'MPIEngineSetLauncher'" >> $NB_WORK/.ipython/profile_mpi/ipcluster_config.py
# # VOLUME $NB_WORK/workspace
# # WORKDIR $NB_WORK
# # CMD ["jupyter", "notebook", "--no-browser", "--ip='0.0.0.0'"]
