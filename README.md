# model-monitoring
 FastAPI application for ML model monitoring (using python:3.10.4)
 
### Instalação
Ao clonar este repositório, navegue para o diretório [./monitoring](monitoring) e crie um python virtual environment. No Windows, pode-se utilizar o seguinte comando no terminal:
 
```
python -m venv monitoring-venv
```
 
Esse comando cria o ambiente virtual com o nome de *monitoring-venv* e para ativá-lo basta rodar o seguinte comando, também no Windows:
 
```
.\monitoring-venv\Scripts\activate
```

#### Dependências

Após ativar o ambiente virtual, utilize o ([pip](https://pip.pypa.io/en/stable/installation/)) para instalar as dependências contidas em [./monitoring/requirements.txt](monitoring/requirements.txt). Para isso, navegue para o diretório [./monitoring](monitoring) e instale as dependências executando:

```
cd monitoring
pip install -r requirements.txt
```

Agora, a API está pronta para ser executada. Para isso, navegue para o diretório [./monitoring/app](monitoring/app) e inicie o app utilizando uvicorn fazendo, por exemplo:

```
cd app
uvicorn main:app --host 0.0.0.0 --port 8001
```

### API

Em [./monitoring/app/main.py](monitoring/app/main.py) estão referenciados dois endpoints: "/performance/" e "/aderencia/".

1. /performance/{model_from}
   - Ao fazer uma requisição POST para o servidor, espera-se, também, um parâmetro para esse endpoint. Isso ocorre pelo fato de haver um método de enhancing do [./monitoring/model.pkl](monitoring/model.pkl) e o usuário poder escolher entre ter como resposta seja o ROC *score* do modelo default ou daquele com melhor performance.
   - Assim, deve-se definir na URL da requisição "performance/default" ou "performance/enhanced". Algo diferente disso deve retornar um score e volumetria nulos.
   - O "body" da requisição para esse endpoint foi definido como ```List[dict]```, já que serão passados vários registros no formato JSON para ser determinada a volumetria e ROC score.
   - Após realizar um método POST nesse endpoint, o usuário pode também escolher visualizar a última resposta dessa requisição realizando um GET em /performance/.
   
2. /aderencia
   - Ao fazer uma requisição POST para esse endpoint, espera-se um "body" da requisição como um  ```dict```. Esse dicionário contém uma única chave com valor referente ao caminho para o dataset de input definido localmente.
   - Após realizar um método POST nesse endpoint, o usuário pode também escolher visualizar a última resposta dessa requisição realizando um GET em /aderencia/.





















