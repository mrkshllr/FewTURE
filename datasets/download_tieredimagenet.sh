#!/bin/sh
# Copyright (c) Markus Hiller and Rongkai Ma -- 2022
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
# Download the tieredImageNet dataset
# id for download is: 1ANczVwnI1BDHIF65TgulaGALFnXBvRfs

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ANczVwnI1BDHIF65TgulaGALFnXBvRfs' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ANczVwnI1BDHIF65TgulaGALFnXBvRfs" -O tieredimagenet.tar && rm -rf /tmp/cookies.txt

tar -xvf tieredimagenet.tar tiered_imagenet_224/
