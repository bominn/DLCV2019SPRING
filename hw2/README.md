
# HW2 ― Object Detection
In this assignment, we are given a dataset of aerial images. Your task is to detect and classify the objects present in the images by determining their bounding boxes.

![enter image description here](https://lh3.googleusercontent.com/jUokHJn3aphsNTopJSh_tMxOvCTHK65EJLCVV-RBW-2LRxSIla7aS8KmbtKn05mcwUxDuIxF8b4)

For more details, please click [this link](https://docs.google.com/presentation/d/1CiO0rZzYbPabMjcgDGfRS6V85bRTLvR5cY3jiEngeLc/edit?usp=sharing) to view the slides of HW2.

### Evaluation
To evaluate your model, you can run the provided evaluation script provided in the starter code by using the following command.

    python3 hw2_evaluation_task.py <PredictionDir> <AnnotationDir>

 - `<PredictionDir>` should be the directory to output your prediction files (e.g. `hw2_train_val/val1500/labelTxt_hbb_pred/`)
 - `<AnnotationDir>` should be the directory of ground truth (e.g. `hw2_train_val/val1500/labelTxt_hbb/`)

Note that your predicted label file should have the same filename as that of its corresponding ground truth label file (both of extension ``.txt``).

### Visualization
To visualization the ground truth or predicted bounding boxes in an image, you can run the provided visualization script provided in the starter code by using the following command.

    python3 visualize_bbox.py <image.jpg> <label.txt>


### Shell script

 1.   `hw2.sh`  
The shell script file for running your `YoloV1-vgg16bn` model.
 2.   `hw2_best.sh`  
The shell script file for running your improved model.

Run code in the following manner:

    bash ./hw2.sh $1 $2
    bash ./hw2_best.sh $1 $2
where `$1` is the testing images directory (e.g. `test/images`), and `$2` is the output prediction directory (e.g. `test/labelTxt_hbb_pred/` ).

### Packages
Below is a list of packages you are allowed to import in this assignment:

> [`python`](https://www.python.org/): 3.5+  
> [`tensorflow`](https://www.tensorflow.org/): 1.13  
> [`keras`](https://keras.io/): 2.2+  
> [`torch`](https://pytorch.org/): 1.0  
> [`h5py`](https://www.h5py.org/): 2.9.0  
> [`numpy`](http://www.numpy.org/): 1.16.2  
> [`pandas`](https://pandas.pydata.org/): 0.24.0  
> [`torchvision`](https://pypi.org/project/torchvision/), [`cv2`](https://pypi.org/project/opencv-python/), [`matplotlib`](https://matplotlib.org/), [`skimage`](https://scikit-image.org/), [`Pillow`](https://pillow.readthedocs.io/en/stable/), [`scipy`](https://www.scipy.org/)  
> [The Python Standard Library](https://docs.python.org/3/library/)


