import os, sys
import torch

os.system('ls data_list*_2025-06-03* | sort > sorted_data_lists.txt')
f = open('sorted_data_lists.txt', 'r')
l = f.readline()
list_list = []
while l:
    fi = open(l.strip(), 'r')
    li = fi.readline()
    cur_list = []
    while li:
        cur_list.append(li.strip())
        li = fi.readline()
    fi.close()
    list_list.append(cur_list) 
    l = f.readline()
f.close()
print(list_list)

for cur_list in list_list:
    L = len(cur_list)
    print(len(cur_list))

overall_list = []
for l in range(L):
    for cur_list in list_list:
        overall_list.append(cur_list[l])

fo = open('full_list_consecutive.txt', 'w')
for o in overall_list:
    fo.write(o.strip() + '\n')
fo.close

print(overall_list)
