FROM jupyter/minimal-notebook

USER root

# Set directory for software installation.
ARG SW_DIR="/opt"
WORKDIR "${SW_DIR}"

# Install specific version of python.
RUN conda install python=3.10

# Install scikit-rt.
RUN git clone https://github.com/scikit-rt/scikit-rt \
    && python -m pip install -e scikit-rt

# Install elastix.
ARG ELASTIX_VERSION="5.1.0"
ARG ELASTIX_LINUX="elastix-${ELASTIX_VERSION}-linux"
ARG ELASTIX_ZIP="${ELASTIX_LINUX}.zip"
RUN wget https://github.com/SuperElastix/elastix/releases/download/${ELASTIX_VERSION}/${ELASTIX_ZIP} \
    && unzip ${ELASTIX_ZIP} -d ${ELASTIX_LINUX} \
    && rm ${ELASTIX_ZIP} \
    && chmod a+x ${ELASTIX_LINUX}/bin/elastix \
    && chmod a+x ${ELASTIX_LINUX}/bin/transformix

# Set up environment for running elastix.
ARG ELASTIX_DIR="${SW_DIR}/${ELASTIX_LINUX}"
ENV PATH="${ELASTIX_DIR}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${ELASTIX_DIR}/lib"

# Install NiftyReg.
ARG NIFTYREG_VERSION="1.3.9"
ARG NIFTYREG_LINUX="NiftyReg-${NIFTYREG_VERSION}-Linux-x86_64-Release"
ARG NIFTYREG_TARBALL="${NIFTYREG_LINUX}.tar.gz"
RUN wget https://sourceforge.net/projects/niftyreg/files/nifty_reg-${NIFTYREG_VERSION}/${NIFTYREG_TARBALL} \
    && tar -zxvf ${NIFTYREG_TARBALL} \
    && rm ${NIFTYREG_TARBALL}

# Set up environment for running NiftyReg.
ARG NIFTYREG_DIR="${SW_DIR}/${NIFTYREG_LINUX}"
ENV PATH="${NIFTYREG_DIR}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${NIFTYREG_DIR}/lib"

# Install TexLive with basic scheme.
ARG SCHEME="basic"
ARG TEX_DIR="${SW_DIR}/tex/latest"
RUN wget https://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz \
    && tar -zxvf install-tl-unx.tar.gz \
    && ./install-tl*/install-tl --scheme ${SCHEME} --texdir ${TEX_DIR} --no-interaction \
    && rm -rf install-tl*

# Add TexLive executables directory to path,
# and install packages needed to use LaTeX with Matplotlib.
ENV PATH="${TEX_DIR}/bin/x86_64-linux:${PATH}"
RUN tlmgr install cm-super collection-fontsrecommended dvipng type1cm underscore

# Install library needed by elastix
RUN apt-get update \
    && apt-get -y install libgomp1

USER ${NB_UID}
# Copy examples to home directory.
WORKDIR "${HOME}"
RUN mkdir -p ./examples \
    && cp -rp ${SW_DIR}/scikit-rt/examples/notebooks/* ./examples

# Enable jupyter lab
ENV JUPYTER_ENABLE_LAB="yes"

LABEL "org.opencontainers.image.description"="Toolkit for radiotherapy image analysis"
LABEL "org.opencontainers.image.documentation"="https://scikit-rt.github.io/scikit-rt/"
LABEL "org.opencontainers.image.source"="https://github.com/scikit-rt/scikit-rt"
