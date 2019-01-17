# Import standard Python modules
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib


# Import Azure ML SDK modules
import azureml.core
from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core import Experiment
from azureml.core.webservice import Webservice
from azureml.core.image import ContainerImage
from azureml.core.webservice import AciWebservice
from azureml.core.conda_dependencies import CondaDependencies 

# Check Azure ML SDK version
print(azureml.core.VERSION)

# Create Azure ML Workspace
ws = Workspace.create(name='salary',
                      subscription_id='9be00a6f-5335-4d37-9847-2f7013522146', 
                      resource_group='mi2',
                      create_resource_group=True,
                      location='southeastasia'
                     )

# Write configuration to local file
ws.write_config()  

# Create Azure ML Experiment
exp = Experiment(workspace=ws, name='salexp')

# Start logging metrics
run = exp.start_logging()                   
run.log("Experiment start time", str(datetime.datetime.now()))

# Load salary dataset
sal = pd.read_csv('data/sal.csv',header=0, index_col=None)
X = sal[['x']]
y = sal['y']

# Split the train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

# Train the model
lm = LinearRegression() 
lm.fit(X_train,y_train) 

# Freeze the model
filename = 'outputs/sal_model.pkl'
joblib.dump(lm, filename)

# Test the model
filename = 'outputs/sal_model.pkl'
loaded_model=joblib.load(filename)
y=loaded_model.predict([[21]])[0]
print(y)

# Log metrics to Azure ML Experiment
run.log('Intercept :', lm.intercept_)
run.log('Slope :', lm.coef_[0])

# End Azure ML Experiment
run.log("Experiment end time", str(datetime.datetime.now()))
run.complete()

# Get Portal URL
print(run.get_portal_url())

# Register the model
model = Model.register(model_path = "outputs/sal_model.pkl",
                       model_name = "sal_model",
                       tags = {"key": "1"},
                       description = "Salary Prediction",
                       workspace = ws)

# Define Azure ML Deploymemt configuration
aciconfig = AciWebservice.deploy_configuration(cpu_cores=1, 
                                               memory_gb=1, 
                                               tags={"data": "Salary",  "method" : "sklearn"}, 
                                               description='Predict Stackoverflow Salary')

# Create enviroment configuration file
salenv = CondaDependencies()
salenv.add_conda_package("scikit-learn")

with open("salenv.yml","w") as f:
    f.write(salenv.serialize_to_string())
with open("salenv.yml","r") as f:
    print(f.read())    


# Create Azure ML Scoring file
''' 
%%writefile score.py
import json
import numpy as np
import os
import pickle
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression

from azureml.core.model import Model

def init():
    global model
    # retrieve the path to the model file using the model name
    model_path = Model.get_model_path('sal_model')
    model = joblib.load(model_path)

def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    # make prediction
    y_hat = model.predict(data)
    return json.dumps(y_hat.tolist())
'''
# Deploy the model to Azure Container Instance
# %%time
image_config = ContainerImage.image_configuration(execution_script="score.py", 
                                                  runtime="python", 
                                                  conda_file="salenv.yml")
# Expose web service
service = Webservice.deploy_from_model(workspace=ws,
                                       name='salary-svc',
                                       deployment_config=aciconfig,
                                       models=[model],
                                       image_config=image_config)

service.wait_for_deployment(show_output=True)

# Get the Web Service URL
print(service.scoring_uri)

# Clean up resources
ws.delete()


