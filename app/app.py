from datetime import datetime
from PIL import Image
from flask import Flask, jsonify, request, render_template, redirect, url_for, send_file
import re
import crosstitch
from io import BytesIO 
import base64
import boto3


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024 # 3MB

# home page
@app.route('/', methods=["GET"])
def home():
    return render_template('home.html')

@app.route('/', methods=["POST"])
def retrieve_image():
    original_image = request.files["original_image"]
    file_name = f"{datetime.today().strftime('%Y-%m-%d-%H%M%S')}_{original_image.filename.replace(' ','')}"
    if original_image.filename != '':
        pillow_image = Image.open(original_image.stream)
        output = BytesIO()
        create = crosstitch.cross_pattern(pillow_image,
                                          int(request.form["aida"]),
                                          int(request.form["width"]),
                                          int(request.form["clusters"]))
        #create.pdf_create()
        l = create.create_grid()
        l.save(output, format='PNG')
        jam = base64.b64encode(output.getvalue()).decode('utf-8')

        return render_template("selection.html", image=jam)
    
    return redirect(url_for('home')) # if file is not uploaded

    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
