




# HW4 ― Videos
In this assignment, we will learn to perform both trimmed action recognition and temporal action segmentation in full-length videos.

<p align="center">
  <img width="750" height="250" src="https://lh3.googleusercontent.com/j48uA36UbZp3KR41opZUzntxhlJWoX_R5joeNsTGMN2_cSXI0UFNKuKVu8em_txzOIVbnU8p_oOb">
</p>

For more details, please click [this link](https://docs.google.com/presentation/d/1goz0OCo31GH2YS4l8qODr_ITL7YD-aeFdfqJ-XJt6nU/edit?usp=sharing) to view the slides of HW4.

# Usage


For this dataset, the action labels are defined as below:

|       Action      | Label |
|:-----------------:|:-----:|
| Other             | 0     |
| Inspect/Read      | 1     |
| Open              | 2     |
| Take              | 3     |
| Cut               | 4     |
| Put               | 5     |
| Close             | 6     |
| Move Around       | 7     |
| Divide/Pull Apart | 8     |
| Pour              | 9     |
| Transfer          | 10    |

### Utility
We have also provided a Python script for reading video files and retrieving labeled videos as a dictionary. For more information, please read the comments in [`reader.py`](reader.py).


### Shell script

 1.   `hw4_p1.sh`  
The shell script file for data preprocessing. This script takes as input two folders: the first one contains the video data, and the second one is where you should output the label file named `p1_valid.txt`.
 2.   `hw4_p2.sh`  
The shell script file for trimmed action recognition. This script takes as input two folders: the first one contains the video data, and the second one is where you should output the label file named `p2_result.txt`.
 3.   `hw4_p3.sh`  
The shell script file for temporal action segmentation. This script takes as input two folders: the first one contains the video data, and the second one is where you should output the label files named `<video_category>.txt`. Note that you should replace `<video_category>` accordingly, and a total of **7** files should be generated in this script.

Run code in the following manner:

**Problem 1**

    bash ./hw4_p1.sh $1 $2 $3
-   `$1` is the folder containing the ***trimmed*** validation videos (e.g. `TrimmedVideos/video/valid/`).
-   `$2` is the path to the ground truth label file for the videos (e.g. `TrimmedVideos/label/gt_valid.csv`).
-   `$3` is the folder to which you should output your predicted labels (e.g. `./output/`).

**Problem 2**

    bash ./hw4_p2.sh $1 $2 $3
-   `$1` is the folder containing the ***trimmed*** validation/test videos.
-   `$2` is the path to the ground truth label file for the videos (e.g. `TrimmedVideos/label/gt_valid.csv` or `TrimmedVideos/label/gt_test.csv`).
-   `$3` is the folder to which you should output your predicted labels (e.g. `./output/`).

**Problem 3**

    bash ./hw4_p3.sh $1 $2
-   `$1` is the folder containing the ***full-length*** validation videos.
-   `$2` is the folder to which you should output your predicted labels (e.g. `./output/`).


### Packages
Below is a list of packages you are allowed to import in this assignment:

> [`python`](https://www.python.org/): 3.5+  
> [`tensorflow`](https://www.tensorflow.org/): 1.13  
> [`keras`](https://keras.io/): 2.2+  
> [`torch`](https://pytorch.org/): 1.0  
> [`h5py`](https://www.h5py.org/): 2.9.0  
> [`numpy`](http://www.numpy.org/): 1.16.2  
> [`pandas`](https://pandas.pydata.org/): 0.24.0  
> [`torchvision`](https://pypi.org/project/torchvision/): 0.2.2  
> [`cv2`](https://pypi.org/project/opencv-python/), [`matplotlib`](https://matplotlib.org/), [`skimage`](https://scikit-image.org/), [`Pillow`](https://pillow.readthedocs.io/en/stable/), [`scipy`](https://www.scipy.org/), [`imageio`](https://pypi.org/project/imageio/)    
> [`scikit-video`](http://www.scikit-video.org/stable/): 1.1.11  
> [The Python Standard Library](https://docs.python.org/3/library/)


