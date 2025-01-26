from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)


class detect_poisoning_attack(models.Model):

    Fid= models.CharField(max_length=300)
    age= models.CharField(max_length=300)
    anaemia= models.CharField(max_length=300)
    creatinine_phosphokinase= models.CharField(max_length=300)
    diabetes= models.CharField(max_length=300)
    ejection_fraction= models.CharField(max_length=300)
    high_blood_pressure= models.CharField(max_length=300)
    platelets= models.CharField(max_length=300)
    serum_creatinine= models.CharField(max_length=300)
    serum_sodium= models.CharField(max_length=300)
    sex= models.CharField(max_length=300)
    smoking_history= models.CharField(max_length=300)
    bmi= models.CharField(max_length=300)
    HbA1c_level= models.CharField(max_length=300)
    blood_glucose_level= models.CharField(max_length=300)
    blockchain_code_sha= models.CharField(max_length=300)
    Prediction= models.CharField(max_length=3000)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)



