#!/bin/bash

# https://askubuntu.com/questions/594964/is-there-a-way-to-fill-an-image-with-its-background-color-to-square-size
pic=$1; convert $pic -trim $pic ; width=$(identify -format "%w" $pic); height=$(identify -format "%h" $pic); new_dim=$((width > height ? width+10 : height+10)); convert $pic -gravity center -extent "${new_dim}x${new_dim}" $pic
