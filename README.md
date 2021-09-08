# Ensembled Membership Auditing
This folder contains the code for our paper: EMA: Auditing Data Removal from Trained Models. Here is the file structure:

```
EMA
├── /arch                   (network architectures)
├── /data           
├── /MIA                    (library to run membership inference)
├── /saves_new              (saved models and auditing results)
├── config.py               (configuration file)
├── data_utils.py 
├── run_audit.py            (script to run auditing)
├── train_model.py          (script to train base and calibration models)
├── trainer.py 
├── aggregate_result.ipynb  (script to aggregate auditing results and get the final score)
└── requirements.yaml
```

## How to run
### Install dependencies
```
conda env create --file environment.yml
```

### Download the COVIDx dataset and the Child X-ray dataset
Please download [the COVIDx dataset](https://drive.google.com/file/d/1PAgABtynz6nivKtDXad8eRxiAMzaC3Z1/view?usp=sharing) and [the ChildX dataset](https://drive.google.com/file/d/1_bOTjFzTwKiWwdugZ7FKevbw6ZbMIbrp/view?usp=sharing), and unzip them to the directory `data/`. 

### Test EMA on benchmark datasets
```bash
epoch=50

# Train the base model
python train_model.py --mode base --dataset MNIST --batch_size 64 --epoch $epoch --train_size 10000 

## Train the calibration model and run audit
for k in 0 10 20 30 40 50
do
    python train_model.py --mode cal --dataset MNIST --batch_size 64 --epoch $epoch --train_size 10000 --k $k --cal_data MNIST
    python run_audit.py --k $k --fold 0 --audit EMA --epoch $epoch --cal_data MNIST --dataset MNIST --cal_size 10000
    python run_audit.py --k $k --fold 1 --audit EMA --epoch $epoch --cal_data MNIST --dataset MNIST --cal_size 10000
    python run_audit.py --k $k --fold 2 --audit EMA --epoch $epoch --cal_data MNIST --dataset MNIST --cal_size 10000 
    python run_audit.py --k $k --fold 3 --audit EMA --epoch $epoch --cal_data MNIST --dataset MNIST --cal_size 10000
    python run_audit.py --k $k --fold 4 --audit EMA --epoch $epoch --cal_data MNIST --dataset MNIST --cal_size 10000
    python run_audit.py --k $k --fold 5 --audit EMA --epoch $epoch --cal_data MNIST --dataset MNIST --cal_size 10000
    python run_audit.py --k $k --fold 6 --audit EMA --epoch $epoch --cal_data MNIST --dataset MNIST --cal_size 10000
done
```

### Test EMA on chest X-ray datasets
```bash
epoch=30

# Train the base model
python train_model.py --mode base --dataset COVIDx --batch_size 64 --epoch $epoch --train_size 4000 

# Train the calibration model and run audit
for k in 0 10 20 30 40 50
do
    python train_model.py --mode cal --dataset COVIDx --batch_size 64 --epoch $epoch --train_size 4000 --k $k --cal_data COVIDx
    python run_audit.py --k $k --fold 0 --audit EMA --epoch $epoch --cal_data COVIDx --dataset COVIDx --cal_size 4000
    python run_audit.py --k $k --fold 1 --audit EMA --epoch $epoch --cal_data COVIDx --dataset COVIDx --cal_size 4000
    python run_audit.py --k $k --fold 2 --audit EMA --epoch $epoch --cal_data COVIDx --dataset COVIDx --cal_size 4000 
    python run_audit.py --k $k --fold 3 --audit EMA --epoch $epoch --cal_data COVIDx --dataset COVIDx --cal_size 4000
    python run_audit.py --k $k --fold 4 --audit EMA --epoch $epoch --cal_data COVIDx --dataset COVIDx --cal_size 4000
    python run_audit.py --k $k --fold 5 --audit EMA --epoch $epoch --cal_data COVIDx --dataset COVIDx --cal_size 4000
    python run_audit.py --k $k --fold 6 --audit EMA --epoch $epoch --cal_data COVIDx --dataset COVIDx --cal_size 4000
done
```
### Aggregate results
To aggreagte results and reproduce our Table 2.b and 3.b, you may run the notebook `aggregate_result.ipynb`.
