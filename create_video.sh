#!/bin/bash
for file in $1/*; do
	echo $file
	ffmpeg -y  -start_number 1 -framerate 5 -i $file/snapshot_%d.png -b:v 3M -vcodec mpeg4 $file.mp4
	rm -rf $file
done

