# AI for medical imaging — Fall 2025 group project

## Project overview

The project is based around the SegTHOR challenge data, which was kindly allowed by Caroline Petitjean (challenge organizer) to use for the course. The challenge was originally on the segmentation of different organs: heart, aorta, esophagus and trachea.

### Structure

The project is decomposed in two main parts: weekly individual assignments, and group project:

- [Assignment 01: slicing of the data and 2D, 3D visualization](/weekly_assignments/01_slicing.md);
- [Assignment 02: (to be revealed) something affine](/weekly_assignments/02_affine.md);
- [Assignment 03: Running the baseline, stitching slices to 3D volume and computing metrics](/weekly_assignments/03_train_metrics.md);
- Group project: improve the baseline by adding elements and ideas from the course.

For deadlines and submission guidelines, see the individual assignment pages and the [Canvas page](https://canvas.uva.nl/courses/52878/assignments).

## Codebase features

This codebase is given as a starting point, to provide an initial neural network that converges during training. (For broader context, this is itself a fork of an [older conference tutorial](https://github.com/LIVIAETS/miccai_weakly_supervised_tutorial) we gave few years ago.) It also provides facilities to locally run some test on a laptop, with a toy dataset and dummy network.

Summary of codebase (in PyTorch)

- slicing the 3D Nifti files to 2D `.png`; **To implement as assignment 01**
- stitching 2D `.png` slices to 3D volume compatible with initial nifti files; **To implement as assignment 03**
- basic 2D segmentation network;
- basic training and printing with cross-entroly loss and Adam;
- partial cross-entropy alternative as a loss (to disable one class during training);
- debug options and facilities (cpu version, "dummy" network, smaller datasets);
- saving of predictions as `.png`;
- log the 2D DSC and cross-entropy over time, with basic plotting;
- tool to compare different segmentations (`viewer/viewer.py`).

**Some recurrent questions might be addressed here directly.** As such, it is expected that small change or additions to this readme to be made.

## Codebase use

### Setting up the environment

```
git clone https://github.com/julianbibo/ai4mi-project.git
cd ai4mi-project
git submodule init
git submodule update
```

This codebase was written for a somewhat recent python (3.10 or more recent). (**Note: Ubuntu and some other Linux distributions might make the distasteful choice to have `python` pointing to 2.+ version, and require to type `python3` explicitly.**) The required packages are listed in [`requirements.txt`](requirements.txt) and a [virtual environment](https://docs.python.org/3/library/venv.html) can easily be created from it through [pip](https://pypi.org/):

```
python -m venv .venv
source .venv/bin/activate
which python  # ensure this is not your system's python anymore
python -m pip install -r requirements.txt
```

Conda is an alternative to pip, but is recommended not to mix `conda install` and `pip install`. Conda can be used to get a later version of Python, when the system's one is too old. Here is a script for setting up a temporary conda environment and setup .venv with Python 3.11.4:

```
conda create -n tmp python=3.11.4 -y
conda activate tmp
python -m venv .venv
source .venv/bin/activate
which python  # ensure this is not your system's python anymore
python -m pip install -r requirements.txt
conda deactivate
conda env remove -n tmp
```

### Setting up distorch

Run the following in the root folder:

```
git clone https://github.com/jeromerony/distorch.git
rm -rf distorch/.git  # cleanup git history
```

### Getting the data

The synthetic dataset is generated randomly, whereas for Segthor it is required to put the file [`segthor_train.zip`](https://amsuni-my.sharepoint.com/:u:/g/personal/h_t_g_kervadec_uva_nl/EfMdFte7pExAnPwt4tYUcxcBbJJO8dqxJP9r-5pm9M_ARw?e=ZNdjee) (required a UvA account) in the `data/` folder. If the computer running it is powerful enough, the recipe for `data/SEGTHOR` can be modified in the [Makefile](Makefile) to enable multi-processing (`-p -1` option, see `python slice_segthor.py --help` or its code directly).

Either run the jobfile `sbatch jobs/make_data`, or manually:

```

$ make data/TOY2
$ make data/SEGTHOR
$ make data/SEGTHOR_CLEAN CFLAGS=-O

```

### Viewing the data

The data can be viewed in different ways:

- looking directly at the `.png` in the sliced folder (`data/TOY2`, `data/SEGTHOR`);
- using the provided "viewer" to compare segmentations ([see below](#viewing-the-results));
- opening the Nifti files from `data/segthor_train` with [3D Slicer](https://www.slicer.org/) or [ITK Snap](http://www.itksnap.org).

#### 2D viewer

Comparing some predictions with the provided [viewer](viewer/viewer.py) (right-click to go to the next set of images, left-click to go back), or simply looking at the data:

```

$ python viewer/viewer.py --img_source data/TOY2/val/img \
 data/TOY2/val/gt \
 --show_img -C 256 --no_contour

```

![Example of the viewer on the TOY example](viewer_toy.png)
**Note:** if using it from a SSH session, it requires X to be forwarded ([Unix/BSD](https://man.archlinux.org/man/ssh.1#X), [Windows](https://mobaxterm.mobatek.net/documentation.html#1_4)) for it to work. Note that X forwarding also needs to be enabled on the server side.

For Segthor, comparing for instance the original data, the fixed ones, and the validation predictions at epoch 10:

```

$ python viewer/viewer.py --img_source data/SEGTHOR_CLEAN/val/img \
 data/SEGTHOR/val/gt data/SEGTHOR_CLEAN/val/gt results/segthor/ce/iter010/val \
 -n 2 -C 5 --remap "{63: 1, 126: 2, 189: 3, 252: 4}" \
 --legend --class_names background esophagus heart trachea aorta

```

![Example of the viewer on the SEGTHOR pre-processed sets](viewer_segthor.png)

#### 3D viewers

[3D Slicer](https://www.slicer.org/) and [ITK Snap](http://www.itksnap.org) are two popular viewers for medical data, here comparing `GT.nii.gz` and the corresponding stitched prediction `Patient_01.nii.gz`:
![Viewing label and prediction](3dslicer.png)

Zooming on the prediction with smoothing disabled:
![Viewing the prediction without smoothing](3dslicer_zoom.png)

## Submission and scoring

Groups will have to submit:

- archive of the git repo with the whole project, which includes:
  - slicing (if any) and any other pre-processing;
  - training;
  - post-processing when applicable;
  - inference;
  - metrics computation/scripts to run the metrics submodule;
- the best trained model;
- predictions on the [test set (required @uva.nl account)](https://amsuni-my.sharepoint.com/:u:/g/personal/h_t_g_kervadec_uva_nl/EWZH7ylUUFFCg3lEzzLzJqMBG7OrPw1K4M78wq9t5iBj_w?e=Yejv5d) (`sha256sum -c data/test.zip.sha256` as optional checksum);
- predictions on the group's internal validation set, the labels of their validation set, and the metrics they computed (akin to Assignment 3).

The main criterions for scoring will include (listed here only for convenience, please see Canvas for reference rubric):

- improvement or lack thereof of performances over baseline;
- code quality/clear [git use](git.md);
- the [final choice of metrics](https://metrics-reloaded.dkfz.de/) (they need to be in 3D);
- correctness of the computed metrics (on the validation set);
- oral presentation.

### Packing the code

`$ git bundle create group-XX.bundle master`

### Saving the best model

`torch.save(net, args.dest / "bestmodel-group-XX.pkl")`

### Archiving everything for submission

All files should be grouped in single folder with the following structure

```

group-XX/
test/
pred/
Patient_41.nii.gz
Patient_42.nii.gz
...
val/
pred/
Patient_21.nii.gz
Patient_32.nii.gz
...
gt/
Patient_21.nii.gz
Patient_32.nii.gz
...
metric01.npz
metric02.npz
...
group-XX.bundle
bestmodel-group-XX.pkl

```

The metrics should be a `.npz` archives, that maps patient ID (e.g., `Patient_21`) to a `ndarray` with shape `KxD` (or `K` if `D = 1`), with `K` the number of classes and `D` the eventual dimensionality of the metric (can be simply 1). Ultimately it is the same format as Distorch from Assignment 3.

The folder should then be [tarred](https://xkcd.com/1168/) and compressed, e.g.:

```

Example using Zstandard:
$ tar cf - group-XX/ | zstd -T0 -3 > group-XX.tar.zst
Example using gunzip:
$ tar cf group-XX.tar.gz - group-XX/

```

## Known issues

### Cannot pickle lambda in the dataloader

Some installs (probably due to Python/Pytorch version mismatch) throw an error about an inability to pickle lambda functions (at the dataloader stage). Short of reinstalling everything, setting the number of workers to 0 seems to get around the problem (`--num_workers 0`).

### Pytorch not compiled for Numpy 2.0

It may happen that Pytorch, when installed through pip, was compiled for Numpy 1.x, which creates some inconsistencies. Downgrading Numpy seems to solve it: `pip install --upgrade "numpy<2"`

```

```
