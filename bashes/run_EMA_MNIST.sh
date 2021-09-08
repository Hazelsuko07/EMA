k=0
epoch=40

## Train the base model
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