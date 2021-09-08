## Script to reproduce Table 3.a, result saved in saved/ks/log_COVID.txt
for fold in 0 1 2 3 4 5 6
    do
        python run_audit.py --a 100 --b 0 --fold $fold --epoch 40 --dataset COVID
        python run_audit.py --a 90 --b 10 --fold $fold --epoch 40 --dataset COVID
        python run_audit.py --a 80 --b 20 --fold $fold --epoch 40 --dataset COVID
        python run_audit.py --a 70 --b 30 --fold $fold --epoch 40 --dataset COVID
        python run_audit.py --a 60 --b 40 --fold $fold --epoch 40 --dataset COVID
        python run_audit.py --a 50 --b 50 --fold $fold --epoch 40 --dataset COVID
    done