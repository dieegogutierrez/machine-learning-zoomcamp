import pickle
from flask import Flask, request, jsonify

def predict_customer(customer, dv, model):
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]


with open('model2.bin', 'rb') as f_in:
    model = pickle.load(f_in)

with open('dv.bin', 'rb') as f_in:
    dv = pickle.load(f_in)


app = Flask('subscription_serving')


@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    prediction = predict_customer(customer, dv, model)
    subscription = prediction >= 0.5
    
    result = {
        'subscription_probability': float(prediction),
        'subscription': bool(subscription),
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)