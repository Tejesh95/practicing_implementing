from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('house_model.pkl', 'rb'))

def encode_yes_no(value):
    return 1 if value.lower() == 'yes' else 0

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            area = float(request.form['area'])
            bedrooms = int(request.form['bedrooms'])
            bathrooms = int(request.form['bathrooms'])
            parking = int(request.form['parking'])
            mainroad = encode_yes_no(request.form['mainroad'])
            guestroom = encode_yes_no(request.form['guestroom'])
            basement = encode_yes_no(request.form['basement'])
            hotwaterheating = encode_yes_no(request.form['hotwaterheating'])
            airconditioning = encode_yes_no(request.form['airconditioning'])

            features = [area, bedrooms, bathrooms, parking,
                        mainroad, guestroom, basement,
                        hotwaterheating, airconditioning]

            prediction = round(model.predict([features])[0], 2)
        except Exception as e:
            prediction = f"Error: {str(e)}"
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
