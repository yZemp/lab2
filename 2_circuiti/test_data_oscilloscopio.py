import matplotlib.pyplot as plt
import csv
# import sys
# sys.path.append("/home/yzemp/Documents/Programming/lab2")
# sys.path.append("/home/yzemp/Documents/Programming/lab2/2_circuiti")

x1 = []
y1 = []
y1err = []
x2 = []
y2 = []
y2err = []
x3 = []
y3 = []
y3err = []

with open('2_circuiti/HD_data.CSV', mode = 'r') as file:
  csvFile = csv.reader(file)
  for line in csvFile:
        x1.append(float(line[3]))
        y1.append(float(line[4]))
        y1err.append(abs(float(line[4]) * .03))
        # y1err.append(.2)

with open('2_circuiti/media.CSV', mode = 'r') as file:
  csvFile = csv.reader(file)
  for line in csvFile:
        x2.append(float(line[3]))
        y2.append(float(line[4]))
        y2err.append(abs(float(line[4]) * .03))
        # y2err.append(.2)

with open('2_circuiti/no_media.CSV', mode = 'r') as file:
  csvFile = csv.reader(file)
  for line in csvFile:
        x3.append(float(line[3]) + .0001) 
        y3.append(float(line[4]))
        y3err.append(abs(float(line[4]) * .03))
        # y3err.append(.2)

msize = 2

plt.errorbar(x1, y1, y1err, linestyle = "None", markersize = msize, marker = "o")

# plt.figure(0)
# plt.errorbar(x2, y2, y2err, linestyle = "None", markersize = msize, marker = "o")
# plt.figure(1)
# plt.errorbar(x3, y3, y3err, linestyle = "None", markersize = msize, marker = "o")
plt.show()