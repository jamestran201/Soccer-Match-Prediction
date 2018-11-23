# Delete this file later on
import csv

outf = open("formated.csv", "r", encoding="utf8")

header = outf.readline()
print("Number of columns: {}".format(len(header.strip().split(","))))

row = outf.readline()
print("Number of values in row: {}".format(len(row.strip().split(","))))
outf.close()
	
