#!/bin/bash
for ((i=0; i<$1; i+=1))
do
	echo "running "$i" th iter..." 
	./resnet_enc_100 $i
done
