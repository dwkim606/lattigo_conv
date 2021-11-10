#!/bin/bash
for ((i=0; i<$1; i+=4))
do
	j=$(($i+4))
	echo "running "$i" to "$j "th iter..." 
	./resnet_enc_100 $i $j
done
