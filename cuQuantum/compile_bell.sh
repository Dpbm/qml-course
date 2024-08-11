export CUQUANTUM_ROOT=${CONDA_PREFIX}
export LD_LIBRARY_PATH=${CUQUANTUM_ROOT}/lib:${LD_LIBRARY_PATH}

FILE=./bell.cu

nvcc ${FILE} -I${CUQUANTUM_ROOT}/include -L${CUQUANTUM_ROOT}/lib -lcustatevec -o bell
