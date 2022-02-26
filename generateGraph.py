import random
n = int(sys.arg[1]) #number of nodes
name = "garph" + n + ".txt"
f = open(name, "a")
for i in range(n):
    f.write(str(i)) #node name
    f.write(" ")
    f.write(str(random.randrange(1, 70, 1))) #x value of node
    f.write(" ")
    f.write(str(random.randrange(1, 50, 1)))# y value of node
    f.write("\n")
f.close()
