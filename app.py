# -*- coding: utf-8 -*-
"""
@author: Sermkiat
"""

import numpy as np
import flask
import pickle

# app
app = flask.Flask(__name__)

# load model
model = pickle.load(open("pipe_mlp.pkcls","rb"))



# routes
@app.route("/")
def home():
    return """
           <body> 
           <h1>Welcome to Seroma prediction<h1>
           <a href="/page">Begin</a>
           </body>"""

    
@app.route("/page")
def page():
   with open("page.html", 'r') as viz_file:
       return viz_file.read()


@app.route("/result", methods=["GET", "POST"])
def result():
    """Gets prediction using the HTML form"""
    if flask.request.method == "POST":
        inputs = flask.request.form
        bmi = inputs["bmi"]
        ht = inputs["ht"]
        dm = inputs["dm"]
        neo = inputs["neo"]
        op = inputs["op"]
        patho = inputs["patho"]



    X_new = np.array([(ht)] + 
                     [(dm)] + [(neo)] + [(op)] + 
                     [(patho)] + [float(bmi)]).reshape(1, -1)
    yhat = model.predict(X_new)
    #y_prob = model.predict_proba(X_new)
    if yhat[0] == 1:
        outcome = ""
    else:
        outcome = "not "
    
    result = results = """
              <body>
              <h2> Seroma prediction <h2>
              <h3>"NeuralNetwork model predict outcome of " + outcome + "having seroma."</h3>
              <a href="/page">Start over</a>
              </body>
    
    return results



   

if __name__ == '__main__':
    """Connect to Server"""
    HOST = "127.0.0.1"
    PORT = "4000"
    app.run(HOST, PORT, debug=True)


