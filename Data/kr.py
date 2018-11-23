import csv
from collections import defaultdict

def clense(d):
    f = [0 for i in range(len(d))]
    j = 0
    for i in d:
        if not "NA" in i:
            f[j] = i
        j += 1
    return f

inf = open("events.csv", "r", encoding="utf8")
outf = open("formated.csv", "w+", encoding="utf8")
games = []
previousSet = {}
inf.readline()

for line in inf:
    data = line.strip().split(",")
    if data[7] != "NA":
        side = int(data[7]) - 1
        currentSet = {data[8], data[9]}
        if not currentSet == previousSet:
            games.append([[0 for i in range(9 + 15)],[0 for i in range(9 + 15)]])
            previousSet = currentSet
        data = clense(data)
        #print(data)
        size = len(games) - 1
        games[size][side][0] += 1
        games[size][side][1] += (int(data[14]))
        games[size][side][2] += (int(data[15]))
        games[size][side][3] += (int(data[16]))
        games[size][side][4] += (int(data[17]))
        games[size][side][5] += (int(data[18]))
        games[size][side][6] += (int(data[19]))
        games[size][side][7] += (int(data[20]))
        games[size][side][8] += (int(data[21]))
        games[size][side][int(data[5]) + 8] += 1

print("events, shot place, shot_outcome, is goal, location, body part, assist meathod, situation, fast break", file = outf, end = ",")
print("e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12,e13,e14,e15", file=outf, end = ",") 
print("event type 2, shot place 2, shot outcome 2, is goal 2, location 2, body part 2, assist meathod 2, situation 2, fast break 2", file=outf, end = ",")
print("e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12,e13,e14,e15", file=outf) 

for game in games:
    for t in game:
        for elem in t:
            print("{}".format(elem), file=outf, end=",")
    print("",file=outf)

inf.close()
outf.close()

