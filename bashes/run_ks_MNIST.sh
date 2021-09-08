## Script to reproduce Table 2.a, result saved in saved/ks/log_MNIST.txt
for caldata in MNIST FMNIST
    do
        for size in 2000 #100 200 500 1000 
        do
            for fold in 0 1 2 3 4 5 6
                do
                    python run_audit.py --a 100 --b 0 --fold $fold --epoch 100 --dataset MNIST --qsize $size --cal_data $caldata
                    python run_audit.py --a 90 --b 5 --fold $fold --epoch 100 --dataset MNIST --qsize $size --cal_data $caldata
                    python run_audit.py --a 80 --b 10 --fold $fold --epoch 100 --dataset MNIST --qsize $size --cal_data $caldata
                    python run_audit.py --a 70 --b 15 --fold $fold --epoch 100 --dataset MNIST --qsize $size --cal_data $caldata
                    python run_audit.py --a 60 --b 20 --fold $fold --epoch 100 --dataset MNIST --qsize $size --cal_data $caldata
                    python run_audit.py --a 50 --b 25 --fold $fold --epoch 100 --dataset MNIST --qsize $size --cal_data $caldata
                done
        done
    done
