#!/bin/bash

#####################
#SBATCH --job-name=dicoh-dataset-creation
#SBATCH --output=/ukp-storage-1/mesgar/DiCoh/_dataset_creation.output
#SBATCH --mail-user=mesgar@ukp.informatik.tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH --partition=ukp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB


#####################
DiCoh=/ukp-storage-1/mesgar/DiCoh
DataDailyDialog=$DiCoh/data/daily_dialog
DataSwitchBoard=$DiCoh/data/switchboard

################

source /ukp-storage-1/mesgar/anaconda3/bin/activate dicoh

################

if [[ -d  $DataDailyDialog ]];
then
    rm -r $DataDailyDialog
    mkdir $DataDailyDialog
fi

cd $DataDailyDialog

wget http://yanran.li/files/ijcnlp_dailydialog.zip
unzip -qq ijcnlp_dailydialog.zip
unzip -qq ijcnlp_dailydialog/train.zip
unzip -qq ijcnlp_dailydialog/validation.zip
unzip -qq ijcnlp_dailydialog/test.zip

rm -r ijcnlp_dailydialog
rm ijcnlp_dailydialog.zip
################

TASKS=(up)
: '
 up is UO in the paper
 hup is EUO in the paper
 ui is Ui in the paper
 us is UR in the paper
 '

DATASETS=(train validation test)

################
CORPUS="DailyDialog"

for TASK in ${TASKS[@]};
do
    for DSET in ${DATASETS[@]};
    do
            echo $CORPUS $DSET $TASK

            python $DiCoh/create_coherency_dataset.py --corpus $CORPUS \
                                                      --seed 135486 \
                                                      --datadir $DataDailyDialog/$DSET \
                                                      --amount 20 \
                                                      --task $TASK
    done
done

################
echo ****************************
################

if [[ -d  $DataSwitchBoard ]];
then
    rm -r $DataSwitchBoard
    mkdir $DataSwitchBoard
fi

cd $DiCoh

if [[ -d  $DiCoh/swda ]];
then
    rm -rf $DiCoh/swda
fi

git clone https://github.com/cgpotts/swda.git

cp $DiCoh/swda/swda.zip $DataSwitchBoard/

cd $DataSwitchBoard

unzip -qq swda.zip 'swda/*'

cd $DiCoh

mv $DataSwitchBoard/swda/* $DataSwitchBoard/

rm -rf $DataSwitchBoard/swda


################

CORPUS="Switchboard"
for TASK in ${TASKS[@]};
do

        echo $CORPUS  $TASK

        python $DiCoh/create_coherency_dataset.py --corpus $CORPUS \
                                                  --seed 135486 \
                                                  --datadir $DataSwitchBoard \
                                                  --task $TASK \
                                                  --amount 20
done

################

