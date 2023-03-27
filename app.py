from flask import Flask, render_template, request
import pandas as pd
import pickle
import sklearn

file = open("resturant_rating_predictor.pkl", "rb")
model = pickle.load(file)

app = Flask(__name__)

@app.route("/")
def Homepage():
    return render_template("index.html")

@app.route("/predictor", methods = ["GET", "POST"])
def predictor():
    
    if request.method == "POST":
        
        # Online Order Facility
        online_order = request.form["online_order"]
        
        # Book Table Facility
        book_table = request.form["book_table"]
        
        # Number of Votes
        votes = int(request.form["votes"])
        
        # Approx Cost for Two People
        approx_cost_for_2_People = float(request.form["approx_cost"])
        
        # Location of the Restaurant
        Location=['Koramangala 6th Block', 'Electronic City', 'Kammanahalli', 'Koramangala 4th Block', 'Basavanagudi', 'Sarjapur Road', 'Residency Road', 'Indiranagar', 'Frazer Town', 'MG Road', 'HSR', 'Bannerghatta Road', 'Church Street', 'Jayanagar', 'Lavelle Road', 'Malleshwaram', 'Rajajinagar', 'Brigade Road', 'Koramangala 5th Block', 'Bellandur', 'Brookefield', 'Kalyan Nagar', 'BTM', 'JP Nagar', 'Marathahalli', 'Whitefield', 'Koramangala 7th Block', 'Old Airport Road', 'New BEL Road', 'Banashankari']
        city_ = request.form["city"]
        city=Location.index(city_)+1
        # Restaurant Type
        Resturant=['Cafes', 'Buffet', 'Delivery', 'Pubs and bars', 'Dine-out', 'Drinks & nightlife', 'Desserts']
        Resturant_ = request.form["Resturant_type"]
        Resturant_type=Resturant.index(Resturant_) +1
        # Cuisines
        cuisines = int(request.form["cuisines"])
        
        # Making Predictions
        x_input = [[online_order,book_table,votes,cuisines,approx_cost_for_2_People,Resturant_type,city]]
        
        rate = model.predict(x_input)
        
        
        return render_template("predictor.html", prediction_text=f" {round(rate, 1)} / 5")
     
    return render_template("predictor.html")

@app.route("/aboutUs")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug = True)