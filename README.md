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

### Performance

Ao enviar um POST para se calcular volumetria e ROC score do modelo referenciado no parâmetro do endpoint, o módulo [performance.py](monitoring/app/api/endpoints/performance.py) é o responsável por determinar tais respostas.

   - Inicialmente, é verificado se o parâmetro do endpoint correspondente à versão do modelo é adequado. Dessa maneira, o modelo [./monitoring/model.pkl](monitoring/model.pkl) é lido utilizando *pickle* e o enhancing é feito caso se deseje a métrica do modelo melhorado. Para realizar tal feito, foi verificado que as bases de dados com as quais o modelo é alimentado são desbalanceadas quanto às classes do TARGET. Assim, foi definido um peso de classes no *estimator* da pipeline default e retreinado o modelo considerando [./datasets/credit_01/train.gz](datasets/credit_01/train.gz). Após isso, é utilizada a pipeline para realizar a predição da lista de registros passada no "body" da requisição e definir finalmente o ROC score.

   - Para o caso da volumetria, a lista de registros é transformada em um ```DataFrame``` e é aplicada à coluna REF_DATE um mapping para se ter os meses do ano equivalentes. Com isso, a volumetria é determinada fazendo ```REF_DATE.value_counts()``` e salvando o resultado em um novo JSON.

   - Os dois valores são colocados em um novo dicionário que é retornado como resposta à requisição como um JSON.


### Aderência

Ao enviar um POST para se calcular as métricas resultantes do teste estatístico Kolmogorov-Smirnov entre as bases da requisição e a de [referência](datasets/credit_01/test.gz), o módulo [aderencia.py](monitoring/app/api/endpoints/aderencia.py) é o responsável por determinar tais valores.

   - Inicialmente, o modelo [./monitoring/model.pkl](monitoring/model.pkl) é lido utilizando *pickle* e o encoder da pipeline é reconfigurado setando ```handle_unknown='ignore'```. Foi verificado que, para a base [./datasets/credit_01/oot.gz](datasets/credit_01/train.gz), o ```ColumnTransformer``` identificava novas categorias em determinadas features dado que é uma distribuição de dados coletada após aquelas utilizadas no treinamento do modelo.

   - Com isso, tanto o dataset utilizado de referência e aquele cujo caminho local foi passado na requisição são lidos como um ```DataFrame``` e suas distribuições de score determinadas pelo modelo são calculadas. Finalmente, utiliza-se [ks_2samp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html) para realizar o teste estatístico e as duas métricas *statistic* e *p-value* são colocadas em um dicionário e retornadas como resposta à requisição como um JSON.


### Pasta utils

   - Nessa pasta estão alguns módulos utilizados frequentemente na API para fazer leitura de arquivos de dados, leitura do modelo e um pré-processamento das [bases](datasets/credit_01). Foi verificado que a pipeline referente ao [modelo](monitoring/model.pkl) espera dados estruturados com 120 colunas, sendo uma delas o TARGET. Assim, no módulo [preprocessing.py](monitoring/app/utils/preprocessing.py) é feita uma inferência e dropadas as colunas com porcentagem de valores nulos maior que 75% e quatro outras que não agregam na predição. Esse processo nos retorna um ```DataFrame``` adequado à pipeline.


### Logging

   - Em [./monitoring/monitoring.log](monitoring/monitoring.log) é possível verificar algumas informações acerca do processamento das requisições feitas para a API. Essas informações são formatadas e adicionadas nesse arquivo utilizando ```logging``` da *standard library* de python.


### Jupyter Notebook

   - Em [./monitoring/monitoring-examples.ipynb](/monitoring/monitoring-examples.ipynb) tem-se exemplos de execução de POSTs nos endpoints da API.






