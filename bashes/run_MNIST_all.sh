epoch=50
mixup=0  # Set to 1 if want to apply mixup
# Train base model
python train_model.py --mode base --epoch $epoch --mixup $mixup

# Train the cal model, set is MNIST
python train_model.py --epoch 100 --k 0 --cal_data MNIST --mode cal --epoch $epoch --mixup $mixup
python train_model.py --epoch 100 --k 10 --cal_data MNIST --mode cal --epoch $epoch --mixup $mixup
python train_model.py --epoch 100 --k 20 --cal_data MNIST --mode cal --epoch $epoch --mixup $mixup
python train_model.py --epoch 100 --k 30 --cal_data MNIST --mode cal --epoch $epoch --mixup $mixup
python train_model.py --epoch 100 --k 40 --cal_data MNIST --mode cal --epoch $epoch --mixup $mixup
python train_model.py --epoch 100 --k 50 --cal_data MNIST --mode cal --epoch $epoch --mixup $mixup

# Run auditing
for epoch in 20 30 40 50 
    do
    for caldata in MNIST
        do
            for k in 0 10 20 30 40 50
                do
                    python run_audit.py --k $k --fold 0 --audit EMA --epoch $epoch --cal_data $caldata --mixup $mixup
                    python run_audit.py --k $k --fold 1 --audit EMA --epoch $epoch --cal_data $caldata --mixup $mixup
                    python run_audit.py --k $k --fold 2 --audit EMA --epoch $epoch --cal_data $caldata --mixup $mixup
                    python run_audit.py --k $k --fold 3 --audit EMA --epoch $epoch --cal_data $caldata --mixup $mixup
                    python run_audit.py --k $k --fold 4 --audit EMA --epoch $epoch --cal_data $caldata --mixup $mixup
                    python run_audit.py --k $k --fold 5 --audit EMA --epoch $epoch --cal_data $caldata --mixup $mixup
                    python run_audit.py --k $k --fold 6 --audit EMA --epoch $epoch --cal_data $caldata --mixup $mixup
                done
        done
    done