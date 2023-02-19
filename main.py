# Import Uvicorn & the necessary modules from FastAPI
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
# Import other necessary packages
from dotenv import load_dotenv
import pickle
import os
# Load the environment variables from the .env file into the application
load_dotenv() 
# Initialize the FastAPI application
app = FastAPI()
# Create a class to load the Model from MLFLOW Registry & use it for prediction
class Model:
    def __init__(self, model_path,input):
        """
        To initalize the model Details
        """
        self.model_path = model_path
        self.input = input

    def predict(self):
        """
        To use the loaded model to make predictions on the data
        """
        loaded_model = pickle.load(open(self.model_path, 'rb'))
        #Score Value
        target_names=['setosa' ,'versicolor' ,'virginica']
        predVal = self.input #[[5.9,3.,5.1,1.8]] #Sepal Lenght, Sepal Width, Petal Lenght, Petal Width
        prediction = loaded_model.predict(predVal)
        prediction = [round(p) for p in prediction]
        prediction = [target_names[p] for p in prediction]
        # Return the predictions  
        return prediction

# Create the POST endpoint with path '/predict'
@app.post("/predict")
async def create_score_input(input: str):
    #input =[5.9,3.,5.1,1.8]
    model_path = "./model.pkl"
    input = [input.strip('][').split(',')]
    model =  Model(model_path,input)
    return {
        str(model.predict())
    }

if __name__ == '__main__':
    app.run(debug=True)