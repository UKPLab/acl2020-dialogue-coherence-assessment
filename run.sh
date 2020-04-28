#!/bin/bash

#####################
#SBATCH --job-name=dicoh
#SBATCH --output=/ukp-storage-1/mesgar/DiCoh/_dicoh-test.output
#SBATCH --mail-user=mesgar@ukp.informatik.tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH --partition=ukp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=16GB


#####################
DiCoh=/ukp-storage-1/mesgar/DiCoh
DataDailyDialog=$DiCoh/data/daily_dialog
DataSwitchBoard=$DiCoh/data/switchboard
Results=$DiCoh/results
################

source /ukp-storage-1/mesgar/anaconda3/bin/activate dicoh
module load cuda/9
################

# if you wonder about the mapping between the task names mentioned in the paper and the code
: '
 up is UO in the paper
 hup is EUO in the paper
 ui is Ui in the paper
 us is UR in the paper
 '

TASKS=(us)      # (up ui us hup)
MODELS=(random) # (random cosine model-3) # model-3 is our MTL model
LOSSES=(mtl)    # (mtl coin da coh) # which loss function to use for training
NUMEXPRIMENTS=1
################
SEED=0            # ? One seed for the whole script which can be used for reseeding RANDOM variable
RANDOM=$SEED      # Setting RANDOM seeds the random number generator.
echo "Random seed = $SEED"
################
CORPUS=DailyDialog

for TASK in ${TASKS[@]};
do
    for MODEL in ${MODELS[@]};
    do
        rm -rf $Results/$CORPUS/$TASK/$MODEL

        for ((CNT = 0 ; CNT < $NUMEXPRIMENTS ; CNT++));
        do

            echo $TASK $MODEL $CNT+1
            python $DiCoh/mtl_coherency.py  --logdir $Results/$CORPUS/$TASK/$MODEL \
                                            --datadir $DataDailyDialog \
                                            --task $TASK \
                                            --do_train \
                                            --do_eval \
                                            --model $MODEL \
                                            --loss mtl \
                                            --seed 24011 \
                                            --cuda 0
        done
    done
done

################

