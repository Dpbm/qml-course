# QML Course

Neste repositório, estão contidos todos os arquivos referentes ao minicurso ministrado por [Alexandre](https://github.com/Dpbm) e Hugo durante a Semana de tecnologia de 2024 na [UNIVEM - Centro Universitário Eurípides de Marília](https://www.univem.edu.br/).

O minicurso possui 4 aulas dividas em 2 dias durante a semana (12 e 13 de novembro).

## Setup

Para utilizar os códigos aqui presentes, será necessário:

- python 3.10
- pip
- conda/mamba/conda-lock (opcional)

Com essas ferramentas instaladas execute no terminal:

```bash
pip install -r requirements.txt 

# ou para conda/mamba
conda env create -f ./environment.yml
conda activate qml-course

# ou usando o conda-lock
conda-lock install ./conda-lock.yml -n qml-course
conda activate qml-course
```

Após isso, basta entrar na pasta [./qiskit](./qiskit/) e abrir o jupyter lab usando:

```bash
jupyter lab
```

## Divisão do minicurso

- Primeiro dia
    - Introdução a computação quântica
    - Demonstração das ferramentas usadas para Computação quântica
    - Demonstração das ferramentas usadas para ML e QML

- Segundo dia
    - Introdução aos modelos convolucionais e modelos quânticos
    - Demonstração Classical ConvNet
    - Demonstração QConvNet
