# Arquitecturas Multinucleo y de Proposito Especifico P1

You should check [AMPE_practicas_01.pdf](https://github.com/quico637/dnn-ampe/blob/main/AMPE_practicas_01.pdf) in order to keep track of the exercises we are dealing with. From now on we will be referring to **AMPE_practicas_01.pdf** as **`the document`**.

## 0. Requisites

You need to have conda installed and create a virtual environment, following the steps described in section 4 of the document.


## 1. Training

To run the first exercise in the document **(section 6)**:

```
python3 1.py 
```

If you want to train the model with a different number of **epochs**, simply exec the following command:

```
python3 train.py -e <epochs_number>
```

## 2. Validation

Similarly to the previous exercise, you can run it simply by:

```
python3 2.py 
```

And just in case that you want to validate an other set of weights files with appropiated formatted names, you ccan run:

```
python3 validation.py -f <directory_path>
```


## 3. Inference

To test the model, we can select 10 random images from the validation dataset and make it predict them. To do that, just exec the following command:

```
python3 3.py 
```

## 4. Own digit recognition

In order to recognize a digit from a given image file, you can simply go to **paint** in windows or **GIMP** (free source) and write you own digits. The background colour should be **black** and the digit it self should be **white**. 

To run the script you should specify:

- The weights file that you want to use.
- The path to the file containing the digit.

To run it: 

```
python3 inferencia.py -f <directory_path>
```

## 5. Preprocessing
If the previous script does not recognize your digit, you can try running `preprocessing-inference.py` with the same syntax.
