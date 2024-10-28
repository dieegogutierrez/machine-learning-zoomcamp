import pickle

def predict_customer(customer, dv, model):
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]
    
with open('model1.bin', 'rb') as f_in:
    model = pickle.load(f_in)

with open('dv.bin', 'rb') as f_in:
    dv = pickle.load(f_in)

customer = {"job": "management", "duration": 400, "poutcome": "success"}

prediction = predict_customer(customer, dv, model)

print('Prediction: %.3f' % prediction)