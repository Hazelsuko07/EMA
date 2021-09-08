epoch=30

## Train the base model
# python train_model.py --mode base --dataset COVIDx --batch_size 64 --epoch $epoch --train_size 4000 

## Train the calibration model and run audit
for k in 0 10 20 30 40 50
do
    # python train_model.py --mode cal --dataset COVIDx --batch_size 64 --epoch $epoch --train_size 4000 --k $k --cal_data COVIDx
    python run_audit.py --k $k --fold 0 --audit EMA --epoch $epoch --cal_data COVIDx --dataset COVIDx --cal_size 4000
    python run_audit.py --k $k --fold 1 --audit EMA --epoch $epoch --cal_data COVIDx --dataset COVIDx --cal_size 4000
    python run_audit.py --k $k --fold 2 --audit EMA --epoch $epoch --cal_data COVIDx --dataset COVIDx --cal_size 4000 
    python run_audit.py --k $k --fold 3 --audit EMA --epoch $epoch --cal_data COVIDx --dataset COVIDx --cal_size 4000
    python run_audit.py --k $k --fold 4 --audit EMA --epoch $epoch --cal_data COVIDx --dataset COVIDx --cal_size 4000
    python run_audit.py --k $k --fold 5 --audit EMA --epoch $epoch --cal_data COVIDx --dataset COVIDx --cal_size 4000
    python run_audit.py --k $k --fold 6 --audit EMA --epoch $epoch --cal_data COVIDx --dataset COVIDx --cal_size 4000
done