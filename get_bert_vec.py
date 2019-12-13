
from bert_serving.client import BertClient

w = "new_chars.txt"
words = []
nw = "chars.txt"
v = "selected_vec.txt"

bc = BertClient(ip='localhost')

with open(w,"r",encoding="utf-8") as wd:
    #i = 0
    for line in wd:
        #print(line.strip("\n"))
        if line.strip("\n") is not "":
            words.append(line.strip("\n"))
        #if i > 10:
        #    break
        #i += 1

#print(words)

with open(nw,"w",encoding="utf-8") as nwd:
    nwd.write("\n".join(words))
vecs= bc.encode(words)
#print(len(vecs))
vecs_str = []

with open(v,"w",encoding="utf-8") as vd:
    for line in vecs:
        #print(line)
        vec = [value for value in map(str,line)]
        #print(len(vec))
        #print(vec)
        vec = " ".join(vec)
        #print(vec)
        vecs_str.append(vec)
        print(len(vecs_str))
    vd.write("\n".join(vecs_str))

vec = bc.encode(['我','你'])
print(vec)
print("维度：",len(vec[0]))
bad_save = ['1','2','3']
with open('bad.txt','w') as bad_file:
    bad_file.write("\n".join(bad_save))
