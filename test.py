import control

s = control.Structure("data/olcao.skl")
print(s.rlm)

print(s.atomCoors[1])
s.toFrac()
print(s.atomCoors[1])
s.toCart()
print(s.atomCoors[1])
s.toFrac()
print(s.atomCoors[1])

print(s.minDistMat())

s.atomCoors[1][2] = 23.0

print(s.atomCoors)

s.applyPBC()
print(s.atomCoors)
