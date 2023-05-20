from flask import Flask,render_template,request
import joblib
import numpy as np

# load all models
model=joblib.load('heart_risk_regression.sav')
model_poly=joblib.load('model_poly.sav')
model_qntl_data=joblib.load('model_qntl_data.sav')
model_qntl_target=joblib.load('model_qntl_target.sav')

# app created
app=Flask(__name__)

# end point creation
# home page
@app.route('/')
def index():

	return render_template('patient_details.html')

# route http://127.0.0.1:5000/getresults
@app.route('/getresults',methods=['POST'])
def getresults():

	# categorical feature handling
	gender_dict={'female':0,'male':1}
	smoke_dict={'no':0,'yes':1}
	bmp_dict={'no':0,'yes':1}
	diab_dict={'no':0,'yes':1}

	# get form details
	result=request.form 

	# get dictionary keys in the form & its type is text (string) so it is converted to float
	name=result['name']
	gender=result['gender']
	age=float(result['age'])
	tc=float(result['tc'])
	hdl=float(result['hdl'])
	# sbp=float(result['sbp']) # check
	smoke=result['smoke']
	bpm=result['bpm']
	diab=result['diab']

	# the features to be added to the model are created as an array & convert to 2d array
	test_data=np.array([gender_dict[gender],age,tc,hdl,smoke_dict[smoke],bmp_dict[bpm],diab_dict[diab]]).reshape(1,-1)

	# quantile transformation of test_data
	test_data = model_qntl_data.transform(test_data)
	# polynomial transformation of test_data
	test_data = model_poly.transform(test_data)

	# the test_data is put into the loaded model
	prediction=model.predict(test_data)

	# the prediction is a quantile transformation between -5 and +5
	# so put it in model_qntl_target and do inverse_transform (-5 to +5)
	prediction = model_qntl_target.inverse_transform(prediction)

	# the received prediction is put into a dictionary
	# we get the prediction as a 2d array. Therefore, the value is rounded to two decimal places
	resultDict={"name":name,"risk":round(prediction[0][0],2)}
	
	# the prediction is sent to the patient_results.html file
	# return resultDict
	return render_template('patient_results.html',results=resultDict)

app.run(debug=True)