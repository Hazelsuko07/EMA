max_epoch=50
mixup=0  # Set to 1 if want to apply mixup
k=0  # No noise!
caldata='MNIST'
# # Train base model
# python train_model.py --mode base --epoch $epoch --mixup $mixup


# Train the cal model using different calibartion sizes
python train_model.py --epoch 100 --k 0 --cal_data MNIST --mode cal --epoch 50 --train_size 10000
python train_model.py --epoch 100 --k 0 --cal_data MNIST --mode cal --epoch 50 --train_size 5000
python train_model.py --epoch 100 --k 0 --cal_data MNIST --mode cal --epoch 50 --train_size 2000
python train_model.py --epoch 100 --k 0 --cal_data MNIST --mode cal --epoch 50 --train_size 1000
python train_model.py --epoch 100 --k 0 --cal_data MNIST --mode cal --epoch 50 --train_size 500
python train_model.py --epoch 100 --k 0 --cal_data MNIST --mode cal --epoch 50 --train_size 200
python train_model.py --epoch 100 --k 0 --cal_data MNIST --mode cal --epoch 50 --train_size 100

# Run audit with different calibartion sizes
for csize in 10000 5000 2000 1000 500 200 100
    do
        python run_audit.py --k $k --fold 0 --audit EMA --epoch $max_epoch --cal_data $caldata --mixup $mixup --cal_size $csize
        python run_audit.py --k $k --fold 1 --audit EMA --epoch $max_epoch --cal_data $caldata --mixup $mixup --cal_size $csize
        python run_audit.py --k $k --fold 2 --audit EMA --epoch $max_epoch --cal_data $caldata --mixup $mixup --cal_size $csize
        python run_audit.py --k $k --fold 3 --audit EMA --epoch $max_epoch --cal_data $caldata --mixup $mixup --cal_size $csize
        python run_audit.py --k $k --fold 4 --audit EMA --epoch $max_epoch --cal_data $caldata --mixup $mixup --cal_size $csize
        python run_audit.py --k $k --fold 5 --audit EMA --epoch $max_epoch --cal_data $caldata --mixup $mixup --cal_size $csize
        python run_audit.py --k $k --fold 6 --audit EMA --epoch $max_epoch --cal_data $caldata --mixup $mixup --cal_size $csize
    done