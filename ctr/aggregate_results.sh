  
dir=$1
cd $dir
pwd
ids=$(ls val_loss_* | sed 's/val_loss_//g' | sed 's/.txt//g'  | xargs)
echo "epoch,iteration,value,key,config" > $dir/aggregate.csv

for id in $ids;
do
  index=index_$id.txt
  vl=val_loss_$id.txt
  vacc=val_acc_$id.txt
  preds=$(ls final_prediction_$id*)
  paste $index $vacc |  awk '{print $1,$2,$3,"val_acc", "'$id'"}'  | sed 's/ /,/g'
  paste $index $vl |  awk '{print $1,$2,$3,"val_loss","'$id'"}' | sed 's/ /,/g'
  for f in $preds
  do
    auc=$(/home/apd10/anaconda3/bin/python3 $dir/../../compute_auc.py $dir/../data/test_ylabels.txt $f 2> /dev/null)
    if [ "$auc" == "" ]
    then 
      auc=-1
    fi
    epoch=$(echo $f | sed 's/.*txt_//g' | awk -F_ '{print $1}' | sed 's/E//g')
    itr=$(echo $f | sed 's/.*txt_//g' | awk -F_ '{print $2}' | sed 's/IT//g')
    echo $epoch $itr $auc "auc" $id | sed 's/ /,/g'
  done
done >> $dir/aggregate.csv