"""Este módulo é de teste"""

file = "../img/knnplow.jpg"
print file
for weights in ['uniform', 'distance']:
    print weights
    file = ("../img/knnplow%s.jpg"%weights)
    print file

