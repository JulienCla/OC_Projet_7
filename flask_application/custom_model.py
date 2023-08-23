# Création d'une fonction personnalisée pour effectuer des prédictions en fonction d'un seuil
class CustomModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold

    def predict(self, context, model_input):
        # Prédiction personnalisée avec le paramètre threshold
        probabilities = self.model.predict_proba(model_input)
        predictions = (probabilities[:, 1] >= self.threshold).astype(int)
        return predictions
    
    def predict_proba(self, model_input, context=None):
        return self.model.predict_proba(model_input)
    
    def model(self, context=None):
        return self.model