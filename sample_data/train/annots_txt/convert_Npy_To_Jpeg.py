import pickle


infile = open("7000.pickle",'rb')
videoAnnot = pickle.load(infile)

with open("7000.txt", 'w') as filehandle:
    for listitem in videoAnnot:
        filehandle.write('%s\n' % listitem)