#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuComplex.h>        // cuDoubleComplex
#include <cutensornet.h>

#include <vector>

#include "./cnot.h"
#include "./h.h"

int main(void) {

  const int32_t n_qubits = 1;

  const std::vector<int64_t> qubits (n_qubits, 2);

  cudaSetDevice(0);
  cutensornetHandle_t handle;
  cutensornetCreate(&handle);

  void *h_gate_for_device{nullptr};

  cudaMalloc(&h_gate_for_device, 4*sizeof(cuDoubleComplex));
  cudaMemcpy(h_gate_for_device, H, 4*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
  


  return EXIT_SUCCESS;
}
