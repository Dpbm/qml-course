# QML Course

Neste repositório, estão contidos todos os arquivos referentes ao minicurso ministrado por [Alexandre](https://github.com/Dpbm) durante a Semana de tecnologia de 2024 na [UNIVEM - Centro Universitário Eurípides de Marília](https://www.univem.edu.br/).

O minicurso possui 4 aulas dividas em 2 dias durante a semana (12 e 13 de novembro).

## Setup

Para utilizar os códigos aqui presentes, será necessário:

- python 3.10
- pip
- conda/mamba/conda-lock (opcional)

Com essas ferramentas instaladas execute no terminal:

```bash

# para dependências do primeiro dia 
pip install -r first-requirements.txt 
# para dependências do segundo dia
pip install -r second-requirements.txt


#-----------------------------------------------

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

---

Para os códigos utilizando cuda, será necessário possuir uma placa da Nvidia que suporta [CuQuantum](https://developer.nvidia.com/cuquantum-sdk). Caso você possua, você pode executar os código normalmente utilizando a SDK nativa como mostrado em [CuQuantum](https://developer.nvidia.com/cuquantum-sdk) e [cuda-q](https://developer.nvidia.com/cuda-q).

Caso contrário faça o seguinte:

- CudaQ
    1. Instale o [docker](https://www.docker.com/)
    2. rode o script [build-run.sh](./cuda-q/build-run.sh)

    ```bash
        # para ambientes unix-like
        chmod +x build-run.sh
        ./build-run.sh

        # ou
        docker build . -t cuda-q
        docker run cuda-q

    ```

   Caso você possua uma placa da Nvidia que suporte cuda, mas não CuQuantum, esse método pode funcionar. Caso contrário:

    1. Clone o repo em um [Notebook do Google Colab](https://colab.research.google.com/)
    2. Instale o docker
    3. execute o script [build-run.sh](./cuda-q/build-run.sh)
    ```bash
        chmod +x build-run.sh
        ./build-run.sh
    ```

- CuQuantum
    1. Clone o repo em um [Notebook do Google Colab](https://colab.research.google.com/)
    2. Abra o notebook [cuQuantum-BellState.ipynb](./cuQuantum/cuQuantum-BellState.ipynb)
    3. Habilite a GPU
    4. execute tudo

## Divisão do minicurso

- Primeiro dia
    - Introdução a computação quântica
    - Demonstração das ferramentas usadas para Computação quântica
    - Demonstração das ferramentas usadas para ML e QML

- Segundo dia
    - Introdução aos modelos convolucionais e modelos quânticos
    - Demonstração Classical ConvNet
    - Demonstração QConvNet
