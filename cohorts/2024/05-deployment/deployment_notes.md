## Deployment

To deploy the model in a server there are some steps:
1. **Train and Save the Model**: After training the model, save it as a file, to use it for making predictions in future (session 02-pickle).
2. **Create API Endpoints**: Make the API endpoints in order to request predictions. It is possible to use the Flask framework to create web service API endpoints that other services can interact with (session 03-flask-intro and 04-flask-deployment).
3. **Some other server deployment options** (sessions 5 to 9):
   - **Pipenv**: Create isolated environments to manage the Python dependencies of the web service, ensuring they don’t interfere with other services on the machine.
   - **Docker**: Package the service in a Docker container, which includes both system and Python dependencies, making it easier to deploy consistently across different environments. 
4. **Deploy to the Cloud**: Finally, deploy the Docker container to a cloud service like AWS to make the model accessible globally, ensuring scalability and reliability.

## 1. Save the Model

- To save the model we made before there is an option using the pickle library:
  - First install the library with the command ```pip install pickle-mixin``` if you don't have it.
    ```python
    import pickle

    with open('model.bin', 'wb') as f_out: # 'wb' means write-binary
        pickle.dump((dict_vectorizer, model), f_out)
    ```
  - To be able to use the model in future without running the code, We need to open the binary file we saved before.
    ```python
    import pickle
    
    with open('model.bin', 'rb') as f_in: # very important to use 'rb' here, it means read-binary 
        dict_vectorizer, model = pickle.load(f_in)
    ## Note: never open a binary file you do not trust the source!
    ```

## 2. Create API Endpoints

- If you haven't installed the library just try installing it with the code ```pip install Flask```.

```python
from flask import Flask

app = Flask('ping') # give an identity to your web service

@app.route('/ping', methods=['GET']) # use decorator to add Flask's functionality to our function
def ping():
    return 'PONG'

if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0', port=9696) # run the code in local machine with the debugging mode true and port 9696
```
- To test it, just use the ```cURL``` command in a new terminal by typing ```curl http://localhost:9696/ping```, or simply open your browser and search ```localhost:9696/ping```, You'll see that the 'PONG' string is received.

## 3. Pipenv

- If you haven't installed the library just try installing it with the code ```pip install pipenv```.

- Creating the virtual environment:
```bash
pipenv install scikit-learn==1.5.2
```

- Accessing the virtual environement:
```bash
pipenv shell
```

## Flask and Gunicorn

- The name of the app has to be the same as file name.
```python
from flask import Flask
app = Flask('subscription_serving')
```
- Deploy to production with gunicorn
```bash
gunicorn --bind 0.0.0.0:9696 subscription_serving:app
```

## Docker

```bash
docker build -t subscription-prediction .  
docker run -it -p 9696:9696 subscription-prediction:latest
```

## AWS CLOUD

- Install EB CLI
```bash
pipenv install awsebcli --dev
```
- Generate Access Keys. Click on User at right up corner > Security Credentials > Access Keys
- Initiate docker container on Elastic Bean. A prompt to enter ID and Key will show.
```shell
eb init -p docker subscription_serving
```
- Testing it locally.
```shell
eb local run --port 9696
```
- If prompted: ERROR: NotSupportedError - You can use "eb local" only with preconfigured, generic and multicontainer Docker platforms.
```shell
eb init -i
```
- Choose the following Docker option: Docker running on 64bit Amazon Linux 2023

- Deploying to the cloud. Use the flag which automatically uses Launch Templates to avoid error.
```shell
eb create subscription-serving-env --enable-spot
```
- A URL will show at the end.

- Terminate the cloud serving
```shell
eb terminate subscription-serving-env
```

