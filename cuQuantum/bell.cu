#include <cuda_runtime_api.h>
#include <cuComplex.h>
#include <custatevec.h>
#include <iostream>
#include <cmath>

#include "./x.h"
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

  const int n_qubits = 2;
  const int state_size = std::pow(2, n_qubits);

  cuDoubleComplex *state = new cuDoubleComplex[state_size];

  cuDoubleComplex *statevector;
  cudaMalloc((void **)&statevector, state_size * sizeof(cuDoubleComplex));

  cudaMemcpy(
      statevector,
      state,
      state_size * sizeof(cuDoubleComplex),
      cudaMemcpyHostToDevice);


  custatevecInitializeStateVector(handle, statevector, CUDA_C_64F, n_qubits, CUSTATEVEC_STATE_VECTOR_TYPE_ZERO);
  
  int hadamard_target[] = {0};
  custatevecApplyMatrix(
      handle,
      statevector,
      CUDA_C_64F,
      1,//n_qubits
      H, //hadamard gate
      CUDA_C_64F,
      CUSTATEVEC_MATRIX_LAYOUT_COL,
      0, //no adjoint
      hadamard_target, //target
      1, //n_targets
      nullptr, // controls
      nullptr, // control bit-string
      0,       // n controls
      CUSTATEVEC_COMPUTE_64F,
      nullptr,
      0);

  int cnot_target[] = {1};
  int cnot_control[] = {0};
  custatevecApplyMatrix(
      handle,
      statevector,
      CUDA_C_64F,
      n_qubits,//n_qubits
      X, //cnot(x controlled) gate
      CUDA_C_64F,
      CUSTATEVEC_MATRIX_LAYOUT_COL,
      0, //no adjoint
      cnot_target, //target
      1, //n_targets
      cnot_control, // controls
      nullptr, // control bit-string
      1,       // n controls
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
  delete state;

  return 0;
}
