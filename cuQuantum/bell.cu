#include <cuda_runtime_api.h>
#include <cuComplex.h>
#include <custatevec.h>
#include <iostream>
#include <cmath>

#include "./h.h"

using std::cout;
using std::endl;

int main(void)
{

  cudaSetDevice(0);

  int n_devices;
  cudaError_t cuda_error = cudaGetDeviceCount(&n_devices);

  if (cuda_error != cudaSuccess)
  {
    cout << "cudaGetDeviceCount Error: "
         << cudaGetErrorString(cuda_error) << endl;
  }
  else
  {
    cout << "Total CUDA Devices: " << n_devices << endl;
  }

  custatevecHandle_t handle;
  custatevecCreate(&handle);

  const int n_qubits = 1;
  const int n_targets = 1;
  const int n_controls = 0;
  const int adjoint = 0;
  const int state_size = std::pow(2, n_qubits);

  cuDoubleComplex *state = new cuDoubleComplex[state_size];

  cuDoubleComplex *statevector;
  cudaMalloc((void **)&statevector, state_size * sizeof(cuDoubleComplex));

  cuDoubleComplex *hadamard;
  cudaMalloc((void **)&hadamard, 4 * sizeof(cuDoubleComplex));

  cudaMemcpy(
      statevector,
      state,
      state_size * sizeof(cuDoubleComplex),
      cudaMemcpyHostToDevice);

  cudaMemcpy(
      hadamard,
      H,
      4 * sizeof(cuDoubleComplex),
      cudaMemcpyHostToDevice);

  cout << "H Matrix" << endl;
  for (int i = 0; i < 4; i++)
  {
    cuDoubleComplex c = H[i];
    cout << (double)c.x << endl;
  }

  custatevecInitializeStateVector(handle, statevector, CUDA_C_64F, n_qubits, CUSTATEVEC_STATE_VECTOR_TYPE_ZERO);

  int targets[] = {0};

  custatevecApplyMatrix(
      handle,
      statevector,
      CUDA_C_64F,
      n_qubits,
      hadamard,
      CUDA_C_64F,
      CUSTATEVEC_MATRIX_LAYOUT_COL,
      adjoint,
      targets,
      n_targets,
      nullptr, // controls
      nullptr, // control bit-string
      0,       // n controls
      CUSTATEVEC_COMPUTE_64F,
      nullptr,
      0);

  cudaMemcpy(
      state,
      statevector,
      state_size * sizeof(cuDoubleComplex),
      cudaMemcpyDeviceToHost);

  for (int i = 0; i < state_size; i++)
  {
    cuDoubleComplex c = state[i];
    cout << (double)c.x << endl;
  }

  custatevecDestroy(handle);
  cudaFree(statevector);
  cudaFree(hadamard);
  delete state;

  return 0;
}
