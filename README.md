# Relightable 3D Gaussian Splatting



## Repository structure

- [`train.py`](train.py): script containing training loop, as well as data loading, shading, optimization and checkpointing
- [`render_compare.py`](render_compare.py): script that makes side-by-side comparisons of inferred renders versus ground truths for beauty, normals and albedo
- [`render_video.py`](render_video.py): script that makes an animation/video demonstrating what renders look like with different views and light positions 
- [`run_project_in_colab.ipynb`](run_project_in_colab.ipynb): notebook that is used to run essentially the whole project
- [`requirements.txt`](requirements.txt)
- [`suzanne.blend`](suzanne.blend): Blender file used to generate the checkered_suzanne dataset

## About the Dataset
I created a custom dataset for this project using Blender. I created it using Suzanne, a 3D model of a chimpanzee head, which is provided in Blender. My dataset contains 400 training images, as well as 30 test images. The training set contains images of the model captured with 4 distinct light positions. The test set contains images captured with yet another distinct light position.

## Setup and Usage

**Step 1. Download the Dataset:** Dowload the `checkered_suzanne.zip` dataset from the Google Drive link [here](https://drive.google.com/file/d/1PFs0D2AOdhcZ-qM9OGh8_mko_RXkTnm3/view?usp=sharing). Then upload the .zip file to your own Google Drive.

**Step 2. Run the project in Google Colab:**
Open [`run_project_in_colab.ipynb`](run_project_in_colab.ipynb) in Colab and follow the instructions provided in the notebook. The notebook mounts Drive, clones the repo, installs dependencies, copies the dataset locally, and runs [`train.py`](train.py), [`render_compare.py`](render_compare.py) and [`render_video.py`](render_video.py). 


## Acknowledgments

This project uses `gsplat.rasterization` and `gsplat.strategy.DefaultStrategy` from [gsplat](https://github.com/nerfstudio-project/gsplat) (from Nerfstudio team), used as an unmodified dependency.