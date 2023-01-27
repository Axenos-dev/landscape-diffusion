from flask import Flask, render_template, redirect, url_for
from generate_images import create_sample

app = Flask("Landscape Diffusion")

@app.route("/", methods = ['GET'])
def main_page():
    return render_template("main.html", img_path=".\static\generated_images\sample.jpg")


@app.route("/generate", methods = ['GET', 'POST'])
def generate_images():
    create_sample(8)
    
    return redirect(url_for("main_page"))

if __name__ == "__main__":
    app.run("0.0.0.0", 5000)