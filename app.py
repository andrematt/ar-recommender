# web app packages
from audioop import reverse
from cmath import nan
from webbrowser import get
#from tkinter import N
#from turtle import distance
from numpy import nan_to_num, square, string_
import requests
from flask import Flask, render_template, redirect, url_for, request,jsonify
from werkzeug.wrappers import Request, Response
from werkzeug.serving import run_simple

# Doc2Vec (and NL processing) packages & log settings
import os
import json
import re
import string
import collections
import random
import gensim
import logging
import csv
import random
from collections import defaultdict
from collections import OrderedDict
from gensim import corpora
from gensim import similarities
from gensim import models
from gensim.parsing.porter import PorterStemmer
import gensim.models as g

p = PorterStemmer()
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Greedy Algo packages
import pandas as pd
import numpy as np
from operator import itemgetter
from collections import defaultdict

MODELS_DIR = "./models/"
DATA_DIR = "./data/"

# Building the dictionary that holds user objects and distances
# Add:
# entrance-door
# bedroom-bed occupancy?
# bedroom lamp
# bedroom hue
# living room hue
# date-time

ACTIVABLE_RULES_LIST_STR = ['3', '4', '6', '7', '8', '25', '26', '28', '36', '40', '44', '47', '48', '49', '50', '54', '55', '56', '57', '61', '63', '64', '68', '70', '71', '82', '83', '84', '90', '98', '103', '108', '111', '112', '116', '121', '126', '130', '136', '139', '141', '146', '148', '153', '159', '161', '165', '166', '173', '176', '177', '178', '179', '180', '181', '183', '186', '188', '190', '194', '196', '200', '201', '205', '211', '215', '222', '225', '226', '227', '232', '235', '237', '240', '241', '242', '244', '247', '249', '252', '258', '261', '265', '267', '268', '270', '272', '274', '276', '277', '278', '287', '288', '292', '293', '294', '296', '298', '302', '303', '306', '308', '310', '315', '316', '317', '318', '322', '327', '330', '331', '332', '334', '335', '336', '342', '343', '350', '352', '358', '382', '389', '401', '404', '410', '425', '427']
ACTIVABLE_RULES_LIST = [eval(i) for i in ACTIVABLE_RULES_LIST_STR]
#ACTIVABLE_RULES_LIST = []
# ACTIVABLE_RULES_LIST.extend(range(60,300))

USER_INSTALLATION_LIST = [ #!!!! MANCA BEDROOM ALLLIGHT!!!
        # LOCAL TIME DAYOF WEEK!
        # LOCALDAY WEEKDAY!!
        # INVOKEFUNCTION CHANGEWINDOWSSTATE!! OK c'�!!
						  ('reminders-reminders', "n", [0,0,0]), 
						  ('alarms-alarms', "n", [0,0,0]), 
						  ('cognitive-trainingtime', "n", [0,0,0]), 
						  ('physiological-steps', "n", [0,0,0]), 
						  ('currentweather-rain', "n", [0,0,0]), 
						  ('currentweather-snow', "n", [0,0,0]), 
						  ('currentweather-outdoorcondition', "n", [0,0,0]), 
						  ('currentweather-outdoortemperature', "n", [0,0,0]), 
						  ('twentyfourhoursweatherforecast-rain', "n", [0,0,0]), 
						  ('twentyfourhoursweatherforecast-snow', "n", [0,0,0]), 
						  ('twentyfourhoursweatherforecast-outdoorcondition', "n", [0,0,0]), 
						  ('twentyfourhoursweatherforecast-outdoortemperature', "n", [0,0,0]), 
						  ('datetime-localtime', "n", [0,0,0]), 
						  ('relativeposition-typeofproximity', "n", [0,0,0]), 
						  #('fridge-door',"yes", [-2,0,2]), 
						  ('microwave-door',"yes", [-1.5,0,2]), 
						  ('kitchen-motion', "y", [0,0,0]), 
						  ('kitchen-windowsensor',"yes", [-2,0,2]), 
						  ('kitchen-window',"yes", [-2,0,2]), 
						  ('kitchen-smokesensor', "yes", [0.5,0,0.5]),
						  ('kitchen-gassensor', "yes", [0.5,0,0.5]),
						  ('kitchenlight-state', "yes", [3,0,3]),
						  ('kitchen-hue color light kitchen', "yes", [3,0,3]),
						  ('kitchen-alllight', "yes", [3,0,3]),
						  ('kitchen-lightlevel', "yes", [0.5, 0, 0.5]),
						  ('kitchen-temperaturelevel', "yes", [0.5,0,0.5]),
						  ('kitchen-humiditylevel', "yes", [0.5,0,0.5]),
						  ('sleep-bedoccupancy', "yes", [0.5, 0, 0.5]),
						  ('sleep-sleepduration', "yes", [0.5, 0, 0.5]),
						  ('bedroom-window', "y", [0,0,0]), 
						  ('bedroom-windowsensor', "y", [0,0,0]), 
						  ('bedroom-motion', "y", [0,0,0]), 
						  ('bedroom-alllight', "y", [0,0,0]), 
						  ('bedroomlight-state', "y", [0,0,0]), 
						  ('bedroom-lightlevel', "yes", [0.5, 0, 0.5]),
						  ('bedroom-temperaturelevel', "yes", [0.5, 0, 0.5]),
						  ('bedroom-humiditylevel', "yes", [0.5, 0, 0.5]),
						  ('livingroom-windowsensor',"yes", [-2,0,2]), 
						  ('livingroom-motion', "y", [0,0,0]), 
						  ('livingroom-window',"yes", [-2,0,2]), 
						  ('livingroom-hue color light living room', "yes", [5.5,0,6]),
						  ('livingroom-alllight', "yes", [5.5,0,6]),
						  ('livingroomlight-state', "yes", [5.5,0,6]),
						  ('livingroom-lightlevel', "yes", [0.5, 0, 0.5]),
						  ('livingroom-temperaturelevel', "yes", [0.5, 0, 0.5]),
						  ('livingroom-humiditylevel', "yes", [0.5, 0, 0.5]),
						  ('entrance-doorsensor',"yes", [-2,0,2]), 
						  ('entrance-door',"yes", [-2,0,2]), 
						  ('entrance-motion', "y", [0,0,0]), 
						  ('entrance-hue color light living room', "yes", [5.5,0,6]),
						  ('entrance-alllight', "yes", [5.5,0,6]),
						  ('entrancelight-state', "yes", [5.5,0,6]),
						  ('entrance-lightlevel', "yes", [0.5, 0, 0.5]),
						  ('entrance-temperaturelevel', "yes", [0.5, 0, 0.5]),
						  ('entrance-humiditylevel', "yes", [0.5, 0, 0.5])]
USER_INSTALLATION_DICT = defaultdict(list)
for k, v1, v2 in USER_INSTALLATION_LIST:
    USER_INSTALLATION_DICT[k].append(v1)
    USER_INSTALLATION_DICT[k].append(v2)

print(USER_INSTALLATION_DICT)
#USER_POSITION = np.array([0,0,0])

model_old = g.Doc2Vec.load(os.path.join(MODELS_DIR, "model_19_07.pkl"))
#model = g.Doc2Vec.load(os.path.join(MODELS_DIR, "model_11_06.pkl"))
data_old= pd.read_csv(os.path.join(DATA_DIR, "element_att_21_07.csv"))

model = g.Doc2Vec.load(os.path.join(MODELS_DIR, "model_full.pkl"))
#model = g.Doc2Vec.load(os.path.join(MODELS_DIR, "model_11_06.pkl"))
#data= pd.read_csv(os.path.join(DATA_DIR, "full_element_att.csv"))
element_att_cols = ['id', 'rule_id', 'original_rule_id', 'xPath', 'parent', 'realName', 'ECA', 'operator', 'nextOperator', 'value', 'completeName']
data= pd.read_csv(os.path.join(DATA_DIR, "full_element_att.csv"), names=element_att_cols, header=None) # train element att table WITH HEADERS


synthetic_nl_cols = ['text', 'rule_id']
id_lookup = pd.read_csv(os.path.join(DATA_DIR, "full_synthetic_nl_no_index.csv"), names=synthetic_nl_cols, header=None) # The same used in step 3 for quick evaluation

app=Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
	

'''
print("IM BUILDING LSI INDEX")
documents = []
#data= pd.read_csv("data/element_att_08_07.csv")
for index, row in data.iterrows():
	my_string = row["parent"] + " " + row["realName"] + " " + row["ECA"] + " " + row["operator"] + " " + row["value"] + " " + row["nextOperator"] 
	documents.append(my_string)

texts = [#TODO use our tokenizer instead of this??
	[word for word in document.lower().split()] 
	for document in documents
]

# remove words that appear only once
frequency = defaultdict(int)
for text in texts:
	for token in text:
		frequency[token] += 1

texts = [ #TODO use our tokenizer instead of this??
	[token for token in text if frequency[token] > 1]
	for text in texts
]

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=4)
lsi.save("models/my_lsi_model")  # save model
'''		
def get_parent(parent):
    if parent == "livingroom":
        return "living room"
    elif parent == "livingroomlight":
        return "living room light"
    elif parent == "kitchenlight":
        return "kitchen light"
    elif parent == "bathroomlight":
        return "bathroom light"
    elif parent == "bedroomlight":
        return "bedroom light"
    elif parent == "entrancelight":
        return "entrance light"
    elif parent == "corridorlight":
        return "corridor light"
    return parent

def get_eca(eca):
	if eca == "action":
		return " then"
	elif eca == "event":
		return " when"
	else:
		return " if"

def get_operator(operator):
    if operator == "OPERATOR":
        return ""
    elif operator == "inside":
        operator = "is inside"
    elif operator == "outside":
        operator = "is outside"
    elif operator == "equal":
        operator = "is"
    elif operator == "lessthen" or operator == "less" or operator == "lt":
        operator = "is less then"
    elif operator == "morethen" or operator == "more" or operator =="gt":
        operator = "is more than"
    stripped_operator = operator.translate(str.maketrans(string.punctuation, '                                '))
    return stripped_operator.lower()

def get_value(value):
	if value == "VALUE":
		value = "true"
	elif value == "custom:greatLuminare":
		value = "activate great luminare"
	elif value == "update:lightColor" or value == "custom:lightColor":
		value = "change light color"
	elif value == "invokeFunctions:lightScene":
		value = "start light scene"
	elif value =="invokeFunctions:changeApplianceState":
		value = "turn on or off"
	elif value =="invokeFunctions:changeDoorState":
		value = "open or close"
	elif value =="invokeFunctions:startRoutine":
		value = "start alexa routine"
	elif value == "custom:notificationMode":
		value = "send an alert notification"
	stripped_value = value.translate(str.maketrans(string.punctuation, '                                '))
	return stripped_value.lower()


def tokenizer(line):
    # when ", is matched replace with ""
    match = re.sub('(",).*$', "", line)
    if match:
        text = match.translate(str.maketrans('','',string.punctuation)).lower()
        stemmed = p.stem_sentence(text)
        result = stemmed.split()
        return result
    text = line.translate(str.maketrans('','',string.punctuation)).lower() 
    stemmed = p.stem_sentence(result)
    result = stemmed.split()
    return result

# instantiate index page
@app.route("/")
def index():
   	return render_template("index.html")

@app.route("/simple-response")
def simple_response():
	return "received!"

@app.route('/query-example')
def query_example():
	#http://127.0.0.1:5000/query-example?nl=reminder to close the fridge&capability=reminders-reminders
	#http://127.0.0.1:5000/query-example?nl=if smoke sensor is on&capability=kitchen-smokesensor
	#http://127.0.0.1:5000/query-example?nl=gas sensor in the kitchen&capability=kitchen-gassensor
	#http://127.0.0.1:5000/query-example?nl=when light level in kitchen is low light &capability=kitchen-lightlevel
	#http://127.0.0.1:5000/query-example?nl=when door is open&capability=microwave-door
	nl = request.args.get('nl')
	capability = request.args.get('capability')
	x = float(request.args.get('x').replace(",","."))
	y = float(request.args.get('y').replace(",","."))
	z = float(request.args.get('z').replace(",","."))
	user_position = np.array([x,y,z])
	#use_context = request.args.get('context')
	use_context = "y"
	#print(user_position)
	#print("capability")
	#print(capability)
	#string_vector = nl.split(' ') # ARGH!!!!!!!!!!!!!!!!!!!!!!
	string_vector = tokenizer(nl)
	print("STRING_VECTOR!!!!!!!!!!")
	print(string_vector)
	inferred_vector = model.infer_vector(string_vector)
	sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
	print("MOST SIMILAR FROM DOC2VEC")
	print(sims[0])
	print(sims[1])
	print(sims[2])
	print(sims[3])
	print(sims[4])
	#results = greedy_opt(nl, capability, sims)
	initial_rules_to_use = 10 #useless
	min_candidates = 15
	results = greedy_opt(capability, sims, initial_rules_to_use, min_candidates, user_position, use_context) 
	#print(results)
	#final_rec_list = from_ids_to_rec_list(results)
	final_rec_dict = from_ids_to_rec_dict(results) 
	#rec_list_json = pd.read_json(final_rec_list, typ='list', orient='records')
	return {"data" : final_rec_dict}
	#return '''<h1>The received string is: {}</h1>'''.format(sims[0])

def from_ids_to_rec_list(recs):
	final_recs = []
	for rec in recs:
		candidate_indexes = np.where(data['id'] == rec)
		candidate_element = data.iloc[candidate_indexes[0]]
		#candidate_dict = candidate_element.to_dict()
		print("----------------------- ADDING TO FINAL REC LIST --------------------------")
		#print(candidate_dict)
		final_recs.append(candidate_element.values.tolist())
	print(final_recs)
	return final_recs

def from_ids_to_rec_dict(recs):
	new_data = data.rename(columns={'operator': 'myOperator', 'original_rule_id': 'originalRuleId', 'rule_id': 'ruleId'})
	final_recs = []
	for rec in recs:
		candidate_indexes = np.where(new_data['id'] == rec)
		candidate_element = new_data.iloc[candidate_indexes[0]]
		#candidate_dict = candidate_element.to_dict()
		print("----------------------- ADDING TO FINAL REC LIST --------------------------")
		print(candidate_element.to_dict('records')[0])
		final_recs.append(candidate_element.to_dict('records')[0])
	print("final_recs: dict to series")
	print(final_recs)
	return final_recs

def getTopNNew(sims, use_context, capability, target_recs_number):
	results = []
	results_no_capability = []
	i = 0
	while len(results_no_capability) < target_recs_number and i < len(sims):
		# Doc2vec returns the row in training set
		my_id= str(id_lookup.iloc[sims[i][0]]['rule_id'])
		'''
		print("ID LOOKUP TEXT!!")
		print (id_lookup.iloc[sims[i][0]]['text'])
		print("MY ID: " + str(my_id))
		print("ID LOOKUP: " + str(id_lookup.iloc[sims[i][0]]['rule_id']))
		'''
		matching_rows = data.loc[data['rule_id'] == my_id]
		#test_id = str(sims[i][0])
		#matching_rows = data.loc[data['rule_id'] == test_id]

		#print(matching_rows)
		if use_context == "y":
			for index, entry in matching_rows.iterrows():
				if entry["completeName"] in USER_INSTALLATION_DICT and int(entry["rule_id"]) in ACTIVABLE_RULES_LIST:
					results.append(entry)
					if entry["completeName"] != capability:
						#results_no_capability.append(data.iloc[index])
						results_no_capability.append(entry)
		else:
			for index, entry in matching_rows.iterrows():
				results.append(entry)
				if entry["completeName"] != capability:
					results_no_capability.append(entry)
		i = i+1
	return (results, results_no_capability)
	'''
	results = []
	results_no_capability = []
	i = 0
	while len(results_no_capability) < target_recs_number and i < len(sims):
		#print("TOP RULES FROM NL MATCHING")
		#print(sims[i][0])
		#rows_with_this_rule_id = data[data['rule_id'].isin([sims[i][0]])]
		for index, row in data.iterrows(): # Let's make this more efficient!!
		#for index, row in rows_with_this_rule_id.iteritems():  TODO
			if row["rule_id"] == sims[i][0]:
				#print(row)
					if use_context == "y":
						if row["completeName"] in USER_INSTALLATION_DICT:
							#print("EXISTS IN USER INSTALLATION!!")
							#print(row)
							results.append(data.iloc[index])
							if row["completeName"] != capability:
								results_no_capability.append(data.iloc[index])
					else: 
						results.append(data.iloc[index])
						results_no_capability.append(data.iloc[index])
		i = i+1
	return (results, results_no_capability)
	'''

def getTopNWithFilter(sims, recs_num, use_context, capability):
	topNsims = []
	for i in range(recs_num):
		topNsims.append(sims[i])
	results = []
	results_no_capability = []
	for i in range(len(topNsims)):
		#print("TOP RULES FROM NL MATCHING")
		#print(sims[i][0])
		for index, row in data.iterrows():
			if row["rule_id"] == topNsims[i][0]:
				#print(row)
					if use_context == "y":
						if row["completeName"] in USER_INSTALLATION_DICT:
							#print("EXISTS IN USER INSTALLATION!!")
							#print(row)
							results.append(data.iloc[index])
							if row["completeName"] != capability:
								results_no_capability.append(data.iloc[index])
						else:
							#print("IS NOT HERE!!")
							continue
					else: 
						results.append(data.iloc[index])
					#print(data.iloc[index]["rule_id"])
					continue
	return (results, results_no_capability)

def getTopN(sims, recs_num):
	topNsims = []
	for i in range(recs_num):
		topNsims.append(sims[i])
	results = []
	for i in range(len(topNsims)):
		#print(sims[i][0])
		for index, row in data.iterrows():
			if row["rule_id"] == topNsims[i][0]:
				#print(data.iloc[index]["rule_id"])
				results.append(data.iloc[index])
				continue
	#print(results[0])
	#print("ALL RESULTS:")
	#print(len(results))
	return results

# create an index rule element(xPath) - rules in which it is present
def create_inverted_index(topN):
	ii = dict()
	for line in topN:
		thisRuleElement = ii.get(line["completeName"])
		if(thisRuleElement):
			ii[line["completeName"]].append(line["rule_id"])
		else:
			ii[line["completeName"]] = [line["rule_id"]]
	#print("------------------------")
	#print("inverted Index!")
	#print(ii)
	return ii

# create an index rule element(xPath) - rules in which it is present
def create_candidate_inverted_index(topN):
	ii = dict()
	for line in topN:
		thisRuleElement = ii.get(line["completeName"])
		if(thisRuleElement):
			ii[line["completeName"]].append(line["id"])
		else:
			ii[line["completeName"]] = [line["id"]]
	#print("------------------------")
	#print("candidate inverted Index!")
	#print(ii)
	return ii

def sort_dictionary(my_dict):
	my_list=sorted(((value, key) for (key,value) in my_dict.items()), reverse=True)
	my_sorted_dict=dict([(k,v) for v,k in my_list])
	return(my_sorted_dict)

def get_diversity(key, suggestions):
	#print("------------------------")
	#print("SUGGESTIONS")
	#print(suggestions)
	#print(len(suggestions))
	suggestions_corpus = []
	candidate_indexes = np.where(data['id'] == key)
	candidate_element = data.iloc[candidate_indexes[0]]
	#print("CANDIDATE ELEMENT")
	my_eca = get_eca(candidate_element["ECA"].values[0])
	my_operator = get_operator(candidate_element["operator"].values[0])
	my_value = get_value(candidate_element["value"].values[0])
	my_text = my_eca + " " + candidate_element["parent"].values[0] + " " + candidate_element["realName"].values[0] + " " + my_operator + " " + my_value + " " 
	candidate_string = my_text
	vec_bow = dictionary.doc2bow(tokenizer(candidate_string)) # TODO TEST IF THE NEW TOKENIZER IS OK!!!
	vec_lsi = lsi[vec_bow]  # convert the query to LSI space
	for element in suggestions:
		suggestion_indexes = np.where(data['id'] == element)
		suggestion_element = data.iloc[suggestion_indexes[0]]
		my_eca = get_eca(suggestion_element["ECA"].values[0])
		my_operator = get_operator(candidate_element["operator"].values[0])
		my_value = get_value(candidate_element["value"].values[0])
		my_text = my_eca + " " + candidate_element["parent"].values[0] + " " + candidate_element["realName"].values[0] + " " + my_operator + " " + my_value + " " 
		suggestion_string = my_text
		suggestions_corpus.append(suggestion_string.lower().split()) # non importa tokenizzare?
		'''
		suggestion_indexes = np.where(data['id'] == element)
		suggestion_element = data.iloc[suggestion_indexes[0]]
		suggestion_string = suggestion_element["realName"].values[0] + " " + suggestion_element["ECA"].values[0] + " " + suggestion_element["operator"].values[0] + " " + suggestion_element["value"].values[0] + " " + suggestion_element["nextOperator"].values[0] 
		suggestions_corpus.append(suggestion_string.lower().split())
		'''
	corpus = [dictionary.doc2bow(text) for text in suggestions_corpus]
	index = similarities.MatrixSimilarity(lsi[corpus]) 
	sims = index[vec_lsi]  # perform a similarity query against the corpus
	#print(sims)  
	my_sum = sum(sims)/len(suggestions)
	#print(my_sum)
	return 1 - my_sum


def calculate_distance(capability, user_position):
	print("USER POSITION")
	print(user_position)
	#print(capability)
	#print(USER_INSTALLATION_DICT[capability])
	if USER_INSTALLATION_DICT[capability][0] == "yes":
		position = np.array(USER_INSTALLATION_DICT[capability][1])
		squared_distance = np.sum((user_position - position)**2, axis=0)
		dist = np.sqrt(squared_distance)
		return dist
	return -1 #standard neutral value 

def create_distance_scores(inverted_index, capability, top20, user_position):
	distance_scores = {}
	for i in range(len(top20)):
		distance_scores[top20[i]["completeName"]] = calculate_distance(top20[i]["completeName"], user_position) 
	return distance_scores

#https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range
def normalize_scores(distance_scores):
	#print(distance_scores)
	normalized_scores = {}
	min = 999
	max = 0 
	for (key,value) in distance_scores.items():
		if value < min and value != -1:
			min = value
		if value > max:
			max = value

	if max == min:
		max = max+1
	#print("---------------- MIN ---------------")
	#print(min)
	#print("---------------- MAX ---------------")
	#print(max)
	for (key,value) in distance_scores.items():
		normalized = {}
		if value == -1:
			normalized = 0.5
		else:
			normalized = (value-min)/(max-min)
			if normalized == 1:
				normalized = 0.99999 
		normalized_scores[key] = normalized
	print("---------------- NORMALIZED DISTANCE SCORES ---------------")
	print(normalized_scores)
	return(normalized_scores)
	


# get support, cofidence and lift foreach element in invered index. 
# returns the lift
def create_support_confidence_lift_scores(inverted_index, capability, topN, topN_with_capability):
	frequency_count = {}
	support_scores = {}
	confidence_scores = {}
	lift_scores = {}
	
	# initialize the tables
	for i in range(len(topN_with_capability)):
		frequency_count[topN_with_capability[i]["completeName"]] =  0
	for i in range(len(topN_with_capability)):
		support_scores[topN_with_capability[i]["completeName"]] =  0
	for i in range(len(topN_with_capability)):
		confidence_scores[topN_with_capability[i]["completeName"]] =  0
	for i in range(len(topN_with_capability)):
		lift_scores[topN_with_capability[i]["completeName"]] =  0
	
	# support: my instances / all rule element instances
	# NOOO!! it is how many times I am toghether with the candidate (what I called confidence before) / total number of transactions (the topN)
	for key, value in inverted_index.items():
		#print("I AM: ")
		#print(key)
		for i in range(len(value)):
			#print("RULE WHERE I AM: ")
			#print(value[i])
			frequency_count[key] = frequency_count[key] + 1
		#support_scores[key] = support_scores[key] / len(topN)
	print("FREQUENCY COUNTS")
	print(frequency_count)
	#return support_scores
	
	#SUPPORT	
	number_of_rules_in_topN = get_number_of_rules_in_topN(topN_with_capability)
	for key, value in inverted_index.items():
		#print("I AM: ")
		#print(key)
		for i in range(len(value)):
			#print("RULE WHERE I AM: ")
			#print(value[i])
			current_rule = create_current_rule(value[i], topN_with_capability)
			if is_in_rule_new(current_rule, capability):
				support_scores[key] = support_scores[key] + 1
				#print("my score: " + str(support_scores[key]))
		#if confidence_scores[key] == 0:
			#confidence_scores[key] = 0.00001
		#print("MY OCCURRENCES: " + str(support_scores[key]))
		support_scores[key] = support_scores[key] / number_of_rules_in_topN  # 
	print("SUPPORT SCORES")
	print(support_scores)
	return support_scores

	# confidence: ratio of rules where I am that also contains the user inserted 
	# capability / all rules where I am
	for key, value in inverted_index.items():
		#print("I AM: ")
		#print(key)
		for i in range(len(value)):
			#print("RULE WHERE I AM: ")
			#print(value[i])
			current_rule = create_current_rule(value[i], topN_with_capability)
			if is_in_rule_new(current_rule, capability):
				confidence_scores[key] = confidence_scores[key] + 1
				#print("my score: " + str(support_scores[key]))
		#if confidence_scores[key] == 0:
			#confidence_scores[key] = 0.00001
		#print("MY OCCURRENCES: " + str(support_scores[key]))
		confidence_scores[key] = confidence_scores[key] / len(value) # this is the len of the list in my inverted index position ( = the list of rules where I am) 
	print("CONFIDENCE SCORES")
	print(confidence_scores)
	return confidence_scores

	# lift: Ratio between my confidence and support of user inserted capabiltiy
	if capability in support_scores:
		B_support = support_scores[capability] # TODO: TEST if now is fixed - if the user inserted capability is not in support_scores it crashes. It is unlikely, but to fix OK Fixed
		for key, value in lift_scores.items():
			if support_scores[key] != 0 and confidence_scores[key] != 0:
				lift_scores[key] = confidence_scores[key] / support_scores[key] * B_support
			else:
				lift_scores[key] = 0
	
	print("LIFT SCORES")
	print(lift_scores)
	return lift_scores

# We are just multipling here because we are using lift which is a metric not in the 0-1 range
#def quality(similarity, distance, diversity):
def quality(similarity, diversity):
	print("MY SCORE:")
	#quality = (similarity * (1- distance)) * diversity
	#quality = similarity * diversity
	#print(quality)
	#return quality
	top = 0.75
	mid = 0.5
	low = 0.25
	print((low * similarity) + ((1-low) * diversity))
	return (low * similarity) + ((1-low) * diversity)


# total diversity > threshold ? return results : bounded_greedy(b * 2 * k)
# Può avere senso aumentare b se c'è troppa poca diversity tra i risultati!
# NL, CAPABILITY, SIMS
#def greedy_opt(nl, capability, sims, recs = 20):
def greedy_opt(capability, sims, used_rules = 1, min_recs_candidates = 15, user_position = np.array([0,0,0]), use_context = "y"):
	#print(data.head())
	#print("LEN SIMS")
	#print(len(sims))
	
	# first: filter rules elements that can not be used in current user installation
	topN = []
	topN_with_capability = []

	results = getTopNNew(sims, use_context, capability, min_recs_candidates)  # ...
	topN_with_capability = results[0]
	topN = results[1]

	if len(topN) < min_recs_candidates:
		print("NOT ENOUGH CANDIDATES WITHIN ALL RULES: TERMINATING")
		raise Exception('NOT ENOUGHT CANDIDATE RULE ELEMENTS COMPATIBLE WITH USER INSTALLATION')
		return
	
	print("---------------------------------------------")
	print(" TOP N")
	print(topN_with_capability)
	print("---------------------------------------------")
	#print(topN_with_capability)
	# Now we got a topN (len = min_recs_candidats) list of xPaths coarsely sorted basing on query similarity

	# create an inverted index xpath: rules id (sorted)
	inverted_index = create_inverted_index(topN) 
	# create another inverted index xpath: rule element id (sorted)
	candidate_inverted_index = create_candidate_inverted_index(topN) 
	
	# create a support dict (xpath: accurate score) & sort it
	lift_scores = create_support_confidence_lift_scores(inverted_index, capability, topN, topN_with_capability) # WARNING now it just return the support
	sorted_lift_scores = sort_dictionary(lift_scores)	

	# distance_scores = create_distance_scores(inverted_index, capability, topN, user_position)
	# normalized_distance_scores = normalize_scores(distance_scores)
	# sorted_distance_scores = sort_dictionary(normalized_distance_scores)
	
	#remove the user inserted element from the candidates and the lift scores dictionaries
	if capability in sorted_lift_scores:
		del sorted_lift_scores[capability]
	#if capability in sorted_distance_scores:
		#del sorted_distance_scores[capability]
	if capability in candidate_inverted_index:
		del candidate_inverted_index[capability]
	print("FINAL SCORES!")
	print(sorted_lift_scores)	
	#print("distance scores!")
	#print(sorted_distance_scores)	
	print("candidates!")
	print(candidate_inverted_index)	

	# create the "real"	suggestions list (considering also the quality)
	suggestions = []
	# for the first element we do not need to check the diversity
	# take element with most support (first of the sorted_support_scores)
	# from that xPath, get the first rule element id in the candidates inverted index
	suggestions.append(candidate_inverted_index[next(iter(sorted_lift_scores))][0])
	#suggestions.append(candidate_inverted_index[next(iter(sorted_support_scores))][0])
	#print("ADDED THE FIRST ELEMENT TO SUGGESTIONS:")
	#print(suggestions)
	#remove the suggestion from the candidates inveted index
	candidate_inverted_index[next(iter(sorted_lift_scores))].pop(0)
	# loop to identify the remaining candidates considering also the diversity
	while(len(suggestions)<5):
		candidates = []
		for key, value in sorted_lift_scores.items():
			if capability != key:
				# get the first rule element id from the candidate inverted index
				if candidate_inverted_index[key] and candidate_inverted_index[key][0] != None:
					my_rule_element_id = candidate_inverted_index[key][0] 
					#print("LIFT: FOR RULE ELEMENT ID " + str(my_rule_element_id))
					#print(value)l
					if my_rule_element_id not in suggestions:
						my_diversity = get_diversity(my_rule_element_id, suggestions)
						#print("MY DIVERSITY")
						#print(my_diversity)
						#my_quality = quality(value, sorted_distance_scores[key], my_diversity)  
						my_quality = quality(value, my_diversity)  
						candidates.append([my_rule_element_id, my_quality, key])
		sorted_candidates = sorted(candidates, key=itemgetter(1), reverse=True)
		#print("SORTED CANDIDATES")
		#print(sorted_candidates)
		suggestions.append(sorted_candidates[0][0])
		my_xpath = sorted_candidates[0][2]
		#print("POPPING FIRST VALUE OF " + my_xpath)
		#print(candidate_inverted_index[my_xpath])
		candidate_inverted_index[my_xpath].pop(0)
		#print(candidate_inverted_index[my_xpath])
	#print(suggestions)
	return suggestions

# update the rule_counter if previous rule id is different from actual
def get_number_of_rules_in_topN(topN_with_capability):
	count = 1
	last = ""
	for index, element in enumerate(topN_with_capability):
		if index == 0:
			last = element["rule_id"]
		else:
			if element["rule_id"] != last:
				count = count + 1
				last = element["rule_id"]
	return count

# NO! it must work with all the recs data, not only with topN. 
# Otherwise good candidates rule elements in rules with a rule element not in 
# the user installation are penalized
def create_current_rule(index, topN):
	current_rule = []
	for i in range(len(topN)):
		if topN[i]["rule_id"] == index:
			current_rule.append(topN[i])
	return current_rule


def is_in_rule_new(current_rule, capability):
	for i in range(len(current_rule)):
		if current_rule[i]["completeName"] == capability:  
			return True
	return False

'''
def getRulesInElementAttTable(elementAttTable):
	print(len(elementAttTable))
	print(elementAttTable)
	count = 0
	index = 0
	initial_id = elementAttTable["original_rule_id"].iloc[0]
	for element in elementAttTable["original_rule_id"]:
		if index == 0:
			index = index +1
			continue
		if(elementAttTable["original_rule_id"].iloc[index -1] != element):
			count = count + 1
		index = index +1
	#print("count: ")
	#print(count)
	return count

def getAllRulesInElementAttTable():
	count = 0
	index = 0
	initial_id = data["original_rule_id"].iloc[0]
	for element in data["original_rule_id"]:
		if index == 0:
			index = index +1
			continue
		if(data["original_rule_id"].iloc[index -1] != element):
			count = count + 1
		index = index +1
	#print("count: ")
	#print(count)
	return count
'''
print("IM BUILDING LSI INDEX")
documents = []
for index, row in data.iterrows():
	my_eca = get_eca(row["ECA"])
	my_operator = get_operator(row["operator"])
	my_value = get_value(row["value"])
	my_string = my_eca + " " + row["parent"] + " " + row["realName"] + " " + my_operator + " " + my_value + " " 
	documents.append(my_string)

texts = []
for document in documents:
	text = tokenizer(document)
	texts.append(text)

print("TEXTS!!") # It works! change it also on the original RecSystem on Heroku
print(texts)

# remove words that appear only once
frequency = defaultdict(int)
for text in texts:
	for token in text:
		frequency[token] += 1

texts = [ 
	[token for token in text if frequency[token] > 1]
	for text in texts
]


dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=4)
lsi.save("models/my_lsi_model")  # save model
print("OK!")

if __name__ == "__main_":
	app.debug = True
	#print(app.before_first_request_funcs)
	run_simple("localhost", 5000, app)

