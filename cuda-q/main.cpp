#include <cudaq.h>
#include <cudaq/algorithms/draw.h>
#include <iostream>

using std::cout;
using std::endl;
using cudaq::qvector;
using cudaq::ctrl;
using cudaq::sample;
using cudaq::draw;

__qpu__ void phi_plus(){
  qvector qubits(2);

  h(qubits[0]);
  x<ctrl>(qubits[0], qubits[1]);

  mz(qubits);
}

__qpu__ void phi_minus(){
  qvector qubits(2);

  x(qubits[0]);
  h(qubits[0]);
  x<ctrl>(qubits[0], qubits[1]);

  mz(qubits);
}

__qpu__ void psi_plus(){
  qvector qubits(2);

  x(qubits[1]);
  h(qubits[0]);
  x<ctrl>(qubits[0], qubits[1]);

  mz(qubits);
}

__qpu__ void psi_minus(){
  qvector qubits(2);

  x(qubits[1]);
  h(qubits[0]);
  z(qubits[0]);
  z(qubits[1]);
  x<ctrl>(qubits[0], qubits[1]);

  mz(qubits);
}

int main(){
  
  auto result_phi_plus = sample(phi_plus);
  cout << "result phi plus:" << endl; 
  result_phi_plus.dump();
  cout << draw(phi_plus);
  
  auto result_phi_minus = sample(phi_minus);
  cout << "result phi minus:" << endl; 
  result_phi_minus.dump();
  cout << draw(phi_minus);

  auto result_psi_plus = sample(psi_plus);
  cout << "result psi plus:" << endl; 
  result_psi_plus.dump();
  cout << draw(psi_plus);
  
  auto result_psi_minus = sample(psi_minus);
  cout << "result psi minus:" << endl; 
  result_psi_minus.dump();
  cout << draw(psi_minus);
}
