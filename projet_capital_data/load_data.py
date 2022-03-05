import urllib3
import requests
import module_mysql 
from numba import jit
import progressbar

def check_url_exist(url):
	try:
		request = requests.get(url)
		return request.status_code == 200
	except:
		return False

@jit
def load_data_from_url(url):
	if check_url_exist(url):
		http = urllib3.PoolManager()
		r = http.request('GET', url)
#		print(r.status)		
		data = r.data.decode("utf-8")
	else:
		print("URL unreachable")
		exit()	
	return data

def save_data_to_file(data,path):
	file = open(path, "w")
	file.write(data)
	file.close()
	return True

def load_data_from_file(path):	
	file = open(path, "r")
	data = file.read()
	file.close()
	return data

@jit
def check_balise(line):
	line_size = len(line)
	output = []
	if line_size>1:
		if line[0]=="@":
			output.append(1)
			output.append(line[1:])
	if line_size>3:
		if line[:3]=="uid":
			output.append(2)
			output.append(line[3:])
	if line_size>6:
		if line[:6]=="cookie":
			output.append(3)
			output.append(line[6:])
	return output


def segment_split(line,info):
	output = []
#return remainder
	position=len(line)
	for i,character in enumerate(line):
		if character=="@":
			position=i
			break
		if len(line[i:])>3:
			if line[i:i+3]=="uid":
				position=i
				break
		if len(line[i:])>6:
			if line[i:i+6]=="cookie":
				position=i
				break
	output.append(info+line[:position])
	output.append(line[position:])
	return output


def segment_email(line):
#find second partition
	info=str()
	partition=str()
	for i,character in enumerate(line):
		if character=="@":
			partition = line[i+1:]
			info = line[:i]+"_AT_"#replace @ because SQL does not support the syntax
	if not partition:
		partition = line
	return segment_split(partition,info)
	
def segment_uid(line):
	return segment_split(line,"")

def segment_cookie(line):
	return segment_split(line,"")

segments = {1 : segment_email  ,
			2 : segment_uid    ,
			3 : segment_cookie }

			
def segmentation_data(data,database):
	lines = data.split('\n')
	i=0
	list_contact =[]
#	print("number of contacts",len(lines))
	bar = progressbar.ProgressBar(max_value=len(lines))
	for line in lines:
		if (len(line)==0):
			break
		contact = {"email":[],"uid":[],"cookie":[]}
		id_contact = {"email":[],"uid":[],"cookie":[]}
# segment data in function of their type and associate it to the good key
		while True:
			check = check_balise(line)
			if len(check)==0:
				return False#first character is not correct : return an error
# commented for optimisation
			output = segments[check[0]](check[1])
			if check[0]==1:
#				output = segment_email(check[1])
				contact["email"].append(output[0])
			if check[0]==2:
#				output = segment_uid(check[1])
				contact["uid"].append(output[0])
			if check[0]==3:
#				output = segment_cookie(check[1])
				contact["cookie"].append(output[0])
			if len(output[1])>0:
				line = output[1]
			else:
				break
# once data of the line is recorded in the dictionnary contact: put it in the database "capital_data_test"
		for uid in contact["uid"]:
			id_uid = database.insert_mysql("uid",uid,)
			id_contact["uid"].append(id_uid[0])
		for email in contact["email"]:
			id_email = database.insert_mysql("email",email)
			id_contact["email"].append(id_email[0])
		for cookie in contact["cookie"]:
			id_cookie = database.insert_mysql("cookie",cookie)	
			id_contact["cookie"].append(id_cookie[0])
		for id_uid in id_contact["uid"]:	
			for id_email in id_contact["email"]:
				database.insert_link("uid","email",id_uid,id_email)
			for id_cookie in id_contact["cookie"]:
				database.insert_link("uid","cookie",id_uid,id_cookie)
		for id_email in id_contact["email"]:
			for id_cookie in id_contact["cookie"]:
				database.insert_link("email","cookie",id_email,id_cookie)		
		bar.update(i)
		i=i+1
	return True

