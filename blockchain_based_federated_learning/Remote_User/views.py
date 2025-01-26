from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404
import datetime

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.ensemble import VotingClassifier
#model selection
from sklearn.metrics import confusion_matrix, accuracy_score, plot_confusion_matrix, classification_report
# Create your views here.
from Remote_User.models import ClientRegister_Model,detect_poisoning_attack,detection_ratio,detection_accuracy

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def Add_DataSet_Details(request):

    return render(request, 'RUser/Add_DataSet_Details.html', {"excel_data": ''})


def Register1(request):

    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city)

        return render(request, 'RUser/Register1.html')
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def predict_poisoning_attack(request):
    if request.method == "POST":

        if request.method == "POST":

            Fid= request.POST.get('Fid')
            age= request.POST.get('age')
            anaemia= request.POST.get('anaemia')
            creatinine_phosphokinase= request.POST.get('creatinine_phosphokinase')
            diabetes= request.POST.get('diabetes')
            ejection_fraction= request.POST.get('ejection_fraction')
            high_blood_pressure= request.POST.get('high_blood_pressure')
            platelets= request.POST.get('platelets')
            serum_creatinine= request.POST.get('serum_creatinine')
            serum_sodium= request.POST.get('serum_sodium')
            sex= request.POST.get('sex')
            smoking_history= request.POST.get('smoking_history')
            bmi= request.POST.get('bmi')
            HbA1c_level= request.POST.get('HbA1c_level')
            blood_glucose_level= request.POST.get('blood_glucose_level')
            blockchain_code_sha= request.POST.get('blockchain_code_sha')

        dataset = pd.read_csv('Healthcare_Datasets.csv', encoding='latin-1')

        def apply_results(label):
            if (label == 0):
                return 0  # No Poisoning Attack Found
            elif (label == 1):
                return 1  # Poisoning Attack Found

        dataset['results'] = dataset['Label'].apply(apply_results)

        cv = CountVectorizer()

        x = dataset["Fid"]
        y = dataset["results"]

        x = cv.fit_transform(x)

        models = []
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
        X_train.shape, X_test.shape, y_train.shape

        print("Convolutional Neural Network (CNN)")
        from sklearn.neural_network import MLPClassifier
        mlpc = MLPClassifier().fit(X_train, y_train)
        y_pred = mlpc.predict(X_test)
        testscore_mlpc = accuracy_score(y_test, y_pred)
        accuracy_score(y_test, y_pred)
        print(accuracy_score(y_test, y_pred))
        print(accuracy_score(y_test, y_pred) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, y_pred))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, y_pred))
        models.append(('MLPClassifier', mlpc))

        print("Naive Bayes")

        from sklearn.naive_bayes import MultinomialNB
        NB = MultinomialNB()
        NB.fit(X_train, y_train)
        predict_nb = NB.predict(X_test)
        naivebayes = accuracy_score(y_test, predict_nb) * 100
        print(naivebayes)
        print(confusion_matrix(y_test, predict_nb))
        print(classification_report(y_test, predict_nb))
        models.append(('naive_bayes', NB))

        # SVM Model
        print("SVM")
        from sklearn import svm
        lin_clf = svm.LinearSVC()
        lin_clf.fit(X_train, y_train)
        predict_svm = lin_clf.predict(X_test)
        svm_acc = accuracy_score(y_test, predict_svm) * 100
        print(svm_acc)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, predict_svm))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, predict_svm))
        models.append(('svm', lin_clf))

        print("Logistic Regression")

        from sklearn.linear_model import LogisticRegression
        reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, y_pred) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, y_pred))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, y_pred))
        models.append(('logistic', reg))


        classifier = VotingClassifier(models)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        Fid1 = [Fid]
        vector1 = cv.transform(Fid1).toarray()
        predict_text = classifier.predict(vector1)

        pred = str(predict_text).replace("[", "")
        pred1 = pred.replace("]", "")

        prediction = int(pred1)

        if prediction == 0:
            val = 'No Poisoning Attack Found'
        elif prediction == 1:
            val = 'Poisoning Attack Found'

            print(val)

        detect_poisoning_attack.objects.create(
        Fid=Fid,
        age=age,
        anaemia=anaemia,
        creatinine_phosphokinase=creatinine_phosphokinase,
        diabetes=diabetes,
        ejection_fraction=ejection_fraction,
        high_blood_pressure=high_blood_pressure,
        platelets=platelets,
        serum_creatinine=serum_creatinine,
        serum_sodium=serum_sodium,
        sex=sex,
        smoking_history=smoking_history,
        bmi=bmi,
        HbA1c_level=HbA1c_level,
        blood_glucose_level=blood_glucose_level,
        blockchain_code_sha=blockchain_code_sha,
        Prediction=val)

        return render(request, 'RUser/predict_poisoning_attack.html',{'objs': val})
    return render(request, 'RUser/predict_poisoning_attack.html')



