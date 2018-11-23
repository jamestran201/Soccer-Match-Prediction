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
outft = open("formatedTesting.csv", "w+", encoding="utf-8")
games = []
scores = [0,0]
previousSet = {}
inf.readline()

def whichWin(d):
    if d[0] > d[1]:
        return 1
    elif d[0] == d[1]:
        return 2
    return 0

for line in inf:
    data = line.strip().split(",")
    if data[7] != "NA":
        side = int(data[7]) - 1
        currentSet = {data[8], data[9]}
        size = len(games) - 1
        if not currentSet == previousSet:
            if len(games) != 0:
                games[size].append(whichWin(scores))
            games.append([[0 for i in range(5 + 15 + 4 + 13)],[0 for i in range(5 + 15 + 4 + 13)]])
            scores = [0,0]
            previousSet = currentSet
        if len(games) < (9075 * (3/4)) or int(data[3]) < 45:
            data = clense(data)
            #print(data)
            size = len(games) - 1
            games[size][side][0] += 1
            if int(data[15]) != 0:
                games[size][side][1] += 1
            games[size][side][2] += (int(data[16]))
            if int(data[19]) != 0:
                games[size][side][3] += 1
            games[size][side][5] += (int(data[21]))
            games[size][side][int(data[5]) + 5 - 1] += 1
            games[size][side][int(data[20]) + 5 + 15 - 1] += 1
            games[size][side][int(data[14]) + 5 + 15 + 4 - 1] += 1
        if int(data[19]) != 0:
            scores[side] += 1

print("events, shots made, is goal, assist count, fast break", file = outf, end = ",")
print("e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12,e13,e14,e15", file=outf, end = ",")
print("s1,s2,s3,s4", file=outf, end = ",")
print("sp1,sp2,sp3,sp4,sp5,sp6,sp7,sp8,sp9,sp10,sp11,sp12,sp13", file=outf, end = ",")
print("events 2, shots made 2, is goal 2, assist count 2, fast break 2", file=outf, end = ",")
print("e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12,e13,e14,e15", file=outf, end=",") 
print("s1,s2,s3,s4", file=outf, end = ",")
print("sp1,sp2,sp3,sp4,sp5,sp6,sp7,sp8,sp9,sp10,sp11,sp12,sp13", file=outf, end=",")
print("target Feature", file=outf)

print("events, shots made, is goal, assist count, fast break", file = outft, end = ",")
print("e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12,e13,e14,e15", file=outft, end = ",")
print("s1,s2,s3,s4", file=outft, end = ",")
print("sp1,sp2,sp3,sp4,sp5,sp6,sp7,sp8,sp9,sp10,sp11,sp12,sp13", file=outft, end = ",")
print("events 2, shots made 2, is goal 2, assist count 2, fast break 2", file=outft, end = ",")
print("e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12,e13,e14,e15", file=outft, end=",") 
print("s1,s2,s3,s4", file=outft, end = ",")
print("sp1,sp2,sp3,sp4,sp5,sp6,sp7,sp8,sp9,sp10,sp11,sp12,sp13", file=outft, end=",")
print("target Feature", file=outft)

for i in range(len(games)):
    game = games[i]
    for t in game:
        try:
            for elem in t:
                if i < (9075 * (3/4)):
                    print("{}".format(elem), file=outf, end=",")
                else:
                    print("{}".format(elem), file=outft, end=",")
        except:
            if i < (9075 * (3/4)):
                print("{}".format(t), file=outf, end=",")
            else:
                print("{}".format(t), file=outft, end=",")
    if i < (9075 * (3/4)):               
        print("",file=outf)
    else:
        print("",file=outft)

inf.close()
outf.close()
outft.close()

