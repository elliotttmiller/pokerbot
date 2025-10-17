#!/bin/bash


for f in ${1}s*.out; do
  grep -h '^[0-9]*:' $f ${f/_s/_r} | sort -n -s -t : -k 1,1 | tr -d ')' | tr '(' ':'| cut -d ':' -f1,2,6
done | awk -F ":" \
'{
if (substr($2,1,1) == "P" ) num = -$3
else num = $3  
if (last==""){
  last = num
} else {
  v = (last + num)/2
  n += 1
  s += v
  ssq += v*v
  last =""
}}
END {
  printf "N=%d VR=%.3f(%.3f)\n", n, s/n, 1.96*sqrt((ssq-s*s/n)/n)/sqrt(n)
}'
