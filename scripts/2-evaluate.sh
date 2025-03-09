#!/bin/bash

dataset=$1;

if [ ! -z "$2" ]
then
    savepath=$2
fi

case "$dataset" in

    "midair")
        if [ -z "$2" ]
        then
            savepath="weights/midair_weights"
        fi
        db_seq_len=""
        data="data/midair/test_data"
        ;;

    *)
        echo "ERROR: Wrong dataset argument supplied"
        ;;
esac

python main.py --mode=eval --dataset="$dataset" $db_seq_len --arch_depth=5 --ckpt_dir="$savepath" --records="$data" $3
