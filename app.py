import os
import PIL
from datetime import datetime
from PIL import Image
from flask import Flask, jsonify, request, render_template, redirect, url_for, send_file
import re
import crosstitch
from io import BytesIO 
import base64


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024 # 3MB

# home page
@app.route('/', methods=["GET"])
def home():
    return render_template('home.html')

@app.route('/', methods=["POST"])
def retrieve_image():
    original_image = request.files["original_image"]
    file_name = f"{datetime.today().strftime('%Y-%m-%d')}_{original_image.filename.replace(' ','')}"
    if original_image.filename != '':
        #original_image.save(os.path.join("originals",file_name))
        pillow_image = Image.open(original_image.stream)
        output = BytesIO()
        cluster_samples = []
        cluster_list = [2,8,20,35]
        csobj =[]
        for i,cluster_no in enumerate(cluster_list):
            csobj.insert(i,crosstitch.cross_pattern(pillow_image,int(request.form["aida"]),15,cluster_no))
            l = csobj[i].create_grid()
            l.save(output, format='PNG')
            cluster_samples.insert(i,base64.b64encode(output.getvalue()).decode('utf-8'))

        return render_template("selection.html", images=cluster_samples, clusts=cluster_list)
    
    return redirect(url_for('home')) # if file is not uploaded

    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
