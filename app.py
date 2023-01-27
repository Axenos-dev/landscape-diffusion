from flask import Flask, render_template

app = Flask("Landscape Diffusion")

@app.route("/")
def ren():
    return render_template("main.html", img_path=".\static\generated_images\sample.jpg")
