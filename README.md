# Landscape Diffusion
Landscape 64x64 image generator, based on PyTorch diffusion model.

# About
Diffusion model that can generate random landscapes, which don`t exist on the Earth. 

There is standart diffusion algorithm, which generates random noise image, and AI tries to predict which noise was used. After some linear algebra formulas we recieving some images

![Image](https://github.com/Axenos-dev/landscape-diffusion/blob/main/generated_images/sample.jpg?raw=true)

# How to run it on your computer
1. Install dependencies `pip install requirements.txt`

*Can be problems on computer without GPU or without installed CUDA and cuDNN*

# How to generate images
After you installed dependencies, you can run `generate_images.py` scripts with cmd line `python genarate_images.py num_of_images`, where `num_of_images` is integer, and describes how many images will be generated (*more images -> slower proccess*)

Example of command: `python generate_images.py 8` *generates image with 8 landscapes*

# Some experiments
### Experiment 1

What will happen if we take a drawing, put some noising(*not a lot*), and pass it through algorithm?

![Image](https://github.com/Axenos-dev/landscape-diffusion/blob/main/img/Experiment.jpg?raw=true)

This experiment probably describes an AI which redraws images in different styles.

# How to try this feature on your images
First of all put your images in `/images/example` folder

You can run `generate_from_images.py` script with cmd line `python generate_from_images.py noise_steps batch_size`

`noise_steps` -  describes how much noise will be on your image before it goes through algorithm, (*I recommend to set it to 160*)

`batch_size` - describes how many images from your dataset will be used

Example of command: `python generate_from_image.py 160 4` *generates image with 4 landscapes, based on 4 images in the folder*

# Now you can run it on site

Use `python app.py` in cmd

It's runs on flask server on `http://127.0.0.1:5000/`

It's not finaly done, but somehow works( *Requires page refresh to see results* )
