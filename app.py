from flask import Flask,render_template, jsonify, request
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    english = ""
    translate =""
    if request.method == "POST":
        # get url that the user has entered
        try:
            english = request.form.get('textbox')
            translate = english
        except:
           translate = "Unable to get URL. Please make sure it's valid and try again."
            
    return render_template('index.html', english=english, translate= translate)

if __name__ == '__main__':
    app.run()