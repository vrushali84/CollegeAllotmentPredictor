from flask import Flask, render_template, escape, request, redirect
from CollegeAllotment import algorithms

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predictCollege')
def predictCollege():
    return render_template("predictCollege.html")

@app.route('/results')
def results():
    a=""
    marks=request.args.get('marks')
    algorithm=request.args.get('algo')
    caste=request.args.get('caste')
    c=algorithms()
    if algorithm=="KNN":
        colleges=c.predictKNN(marks,caste)
        a="multi"
        return render_template("results.html",colleges=colleges,a=a)
    elif algorithm=="SVM":
        college=c.predictSVM(marks,caste)
        colleges=[]
        colleges.append(college)
        a="single"
        return render_template("results.html",colleges=colleges,a=a)
        
@app.route('/colleges')
def colleges():
    return render_template("colleges.html")
   
@app.route('/sortedResults')
def sortedResults():
    start=float(request.args.get('minpercentage'))
    end=float(request.args.get('maxpercentage'))
    results=int(request.args.get('results'))
    caste=str(request.args.get('caste'))
    c=algorithms()
    my_list=c.get_by_range(start,end,results,caste)
    return render_template("sortedResults.html",my_list=my_list,size=len(my_list))

if __name__ == '__main__':
    app.run()
