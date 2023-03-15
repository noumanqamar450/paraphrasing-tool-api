from flask import Flask, request
from main2 import *
import os, re

app = Flask(__name__)
@app.route('/api/healthcheck',methods = ["GET"])
def hello():
    return {"msg": 'Hello, World!'}

@app.route('/best-en',methods = ["POST"])
def query():
    data = request.json
    data = data["data"]
    num_beams = 2
    num_return_sequences = 2
    phrases = [s.strip() for s in data.split('.') if s.strip() != '']
    x = []
    for phrase in phrases:
        # print(phrase)
        z = get_response(phrase,num_return_sequences,num_beams,temperature=1,do_sample=False)
        x = x + z
    list1 = x[0::2]
    list2 = x[1::2]
    # list3 = x[2::3]
    # print(" ".join(list1))
    # print(" ".join(list2))
    x = list1 + ["\n"] + list2
    answer = ' '.join(x)
    print(answer)
    return {"Answer": answer}   


@app.route('/avg-en',methods = ["POST"])
def query1():
    data = request.json
    data = data["data"]
    num_beams = 5
    num_return_sequences = 2
    phrases = [s.strip() for s in data.split('.') if s.strip() != '']
    x = []
    for phrase in phrases:
        # print(phrase)
        z = get_response(phrase,num_return_sequences,num_beams,temperature=1,do_sample=True)
        x = x + z
    list1 = x[0::2]
    list2 = x[1::2]
    # list3 = x[2::3]
    # print(" ".join(list1))
    # print(" ".join(list2))
    x = list1 + ["\n"] + list2
    answer = ' '.join(x)
    print(answer)
    return {"Answer": answer} 


@app.route('/low-en',methods = ["POST"])
def query2():
    data = request.json
    data = data["data"]
    num_beams = 10
    num_return_sequences = 2
    phrases = [s.strip() for s in data.split('.') if s.strip() != '']
    x = []
    for phrase in phrases:
        # print(phrase)
        z = get_response(phrase,num_return_sequences,num_beams,temperature=0.5,do_sample=True)
        x = x + z
    list1 = x[0::2]
    list2 = x[1::2]
    # list3 = x[2::3]
    # print(" ".join(list1))
    # print(" ".join(list2))
    x = list1 + ["\n"] + list2
    answer = ' '.join(x)
    print(answer)
    return {"Answer": answer}



@app.route('/best-sp',methods = ["POST"])
def query3():
    data = request.json
    data = data["data"]
    num_beams = 2
    num_return_sequences = 2
    phrases = [s.strip() for s in data.split('.') if s.strip() != '']
    x = []
    for phrase in phrases:
        # print(phrase)
        z = get_response1(phrase,num_return_sequences,num_beams,temperature=1,do_sample=True)
        x = x + z
    list1 = x[0::2]
    list2 = x[1::2]
    # list3 = x[2::3]
    # print(" ".join(list1))
    # print(" ".join(list2))
    x = list1 + ["\n"] + list2
    answer = ' '.join(x)
    print(answer)
    return {"Answer": answer}


@app.route('/low-sp',methods = ["POST"])
def query4():
    data = request.json
    data = data["data"]
    num_beams = 2
    num_return_sequences = 5
    phrases = [s.strip() for s in data.split('.') if s.strip() != '']
    x = []
    for phrase in phrases:
        # print(phrase)
        z = get_response1(phrase,num_return_sequences,num_beams,temperature=1,do_sample=False)
        x = x + z
    list1 = x[0::2]
    list2 = x[1::2]
    # list3 = x[2::3]
    # print(" ".join(list1))
    # print(" ".join(list2))
    x = list1 + ["\n"] + list2
    answer = ' '.join(x)
    print(answer)
    return {"Answer": answer}





if __name__ == '__main__':
    app.run()
