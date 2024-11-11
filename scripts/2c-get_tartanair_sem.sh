#!/bin/bash
# Downloads and extracts the Kitti dataset
# First argument == path to desired dataset destination

echo "Starting to download TartanAir"

[ -z "$1" ] && echo "ERROR: No dataset destination path argument supplied"

if [ -z "$1" ]
then
	exit 1
fi

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

mkdir -p "$1"
DB_DIR="$(realpath "$1")"

if [ ! -d "$SCRIPT_DIR/../datasets/TartanAir" ] && [ ! -L "$SCRIPT_DIR/../datasets/TartanAir" ]
then
    mkdir -p "$SCRIPT_DIR/../datasets"
    ln -rs "$DB_DIR" "$SCRIPT_DIR/../datasets/TartanAir"
fi

cd "$DB_DIR"

files=(gascola/Easy/seg
       neighborhood/Easy/seg
       oldtown/Easy/seg
       seasonsforest_winter/Easy/seg
       gascola/Hard/seg
       neighborhood/Hard/seg
       oldtown/Hard/seg
       seasonsforest_winter/Hard/seg)

for i in ${files[@]}; do

        shortname=$i'_left.zip'
        #fullname='https://tartanair.blob.core.windows.net/tartanair-release1/'$i'_left.zip'
        fullname='http://airlab-share.andrew.cmu.edu/tartanair/'$i'_left.zip'       
	echo "Downloading: "$shortname
        wget --content-disposition -x -nH $fullname
done 

echo "Unzip dataset "
#find tartanair-release1/ -name "*_left.zip" | while read filename; do unzip -o -d ./ "$filename"; rm "$filename"; done;

#rm -r tartanair-release1

find tartanair/ -name "*_left.zip" | while read filename; do unzip -o -d ./ "$filename"; rm "$filename"; done;

rm -r tartanair
