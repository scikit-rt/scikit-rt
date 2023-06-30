FROM jupyter/minimal-notebook

USER root

# Set directory for storing files.
# This shouldn't be ${HOME}, as this will be recreated
# when image is used with jupyterhub on a kubernetes cluseter.
ARG STORE="/opt"
WORKDIR "${STORE}"

# Install specific version of python.
RUN conda install python=3.10

# Install scikit-rt.
RUN git clone https://github.com/scikit-rt/scikit-rt \
    && python -m pip install -e scikit-rt

# Copy example Jupyter notebooks.
ARG NOTEBOOKS="examples/notebooks"
ARG EXAMPLES="${STORE}/examples"
RUN mkdir -p ${EXAMPLES}
COPY ${NOTEBOOKS}/patient_datasets.ipynb ${EXAMPLES}
COPY ${NOTEBOOKS}/plotting_demo.ipynb ${EXAMPLES}
COPY ${NOTEBOOKS}/application_demo.ipynb ${EXAMPLES}
COPY ${NOTEBOOKS}/image_processing.ipynb ${EXAMPLES}
COPY ${NOTEBOOKS}/roi_intensities.ipynb ${EXAMPLES}
COPY ${NOTEBOOKS}/dose_volume_rois.ipynb ${EXAMPLES}
COPY ${NOTEBOOKS}/eqd.ipynb ${EXAMPLES}
COPY ${NOTEBOOKS}/synthetic_dicom_dataset.ipynb ${EXAMPLES}
COPY ${NOTEBOOKS}/grid_creation.ipynb ${EXAMPLES}
COPY ${NOTEBOOKS}/image_registration_checks.ipynb ${EXAMPLES}

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
ARG ELASTIX_DIR="${STORE}/${ELASTIX_LINUX}"
ENV PATH="${ELASTIX_DIR}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${ELASTIX_DIR}/lib"

# Install TexLive with basic scheme.
ARG SCHEME="basic"
ARG TEX_DIR="${STORE}/tex/latest"
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
# Copy examples to home directory,
# for when not using image with jupyterhub on a kubernetes cluseter.
WORKDIR "${HOME}"
RUN mkdir -p ./examples \
    && cp -rp ${EXAMPLES} ./examples

# Enable jupyter lab
ENV JUPYTER_ENABLE_LAB="yes"

LABEL "org.opencontainers.image.description"="Toolkit for radiotherapy image analysis"
LABEL "org.opencontainers.image.documentation"="https://scikit-rt.github.io/scikit-rt/"
LABEL "org.opencontainers.image.source"="https://github.com/scikit-rt/scikit-rt"
