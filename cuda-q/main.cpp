#include <cudaq.h>
#include <iostream>

using std::cout;
using cudaq::qvector;
using cudaq::ctrl;
using cudaq::sample;

__qpu__ void kernel(){
  qvector qubits(2);

  h(qubits[0]);
  x<ctrl>(qubits[0], qubits[1]);

  mz(qubits);
}

int main(){
  auto result = sample(kernel);
  result.dump();
}
