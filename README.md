# Dialogue Coherence Assessment Without Explicit Dialogue Act Labels #

A neural local coherence model trained in a multi-task learning scenario where dialogue act prediction is used as an auxilary task. 
Special thanks to my student Sebastian Bücker for implementing many parts of my ideas for this project. 
We appreciate your interest to this project. 
Please cite [this paper](https://to.appear) if you use the code in this repository.  
Also don't forget to give the repo a Github star (on top right). 
Thanks. 



### Setup ###

* platform: linux-64
* anaconda: 2019.07
* conda: 4.7.10
* Python: 3.6.8
* GCC 7.3.0


### Conda environment ###

You can install all required packages as follows: 
 
```bash

conda create --name dicoh --file conda_env.txt

conda activate dicoh

pip install -r pip_spec.txt

``` 

### Data ###

We conduct our experiments on two English dialogue corpora: 

* [DailyDialog](https://www.aclweb.org/anthology/I17-1099/)

The dataset can be downloaded from [here](https://www.aclweb.org/anthology/I17-1099/) or from [here](http://yanran.li/dailydialog.html)

However, we do it in the ```exec_dataset_creation.sh``` script. 
	
* SwitchBoard

You can get the data and scripts for processing SwitchBoard from [here](https://github.com/cgpotts/swda.git).
The ```exec_dataset_creation.sh``` script will get them automatically. 

```bash
bash exec_dataset_creation.sh
```


### Procedure ###

```bash

bash run.sh

```

### License ###

This project is licensed under the terms of the MIT license.

### Publication ###

Mohsen Mesgar, Sebastian Bücker, and Iryna Gurevych. ACL 2020. 