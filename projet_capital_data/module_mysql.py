import pymysql
import pymysql.cursors
from numba import jit

class mySQLdb:

	def __init__(self,user,host,password):
# connection to mysql 
		self.connection = pymysql.connect(user=user,host=host,password=password)
		self.cursor=self.connection.cursor()
#avoid warnings		
		self.cursor.execute("SET sql_notes = 0; ")	

	def __del__(self):
		self.connection.close()

	def create_database(self,database_name):
# create database if not exists
		sql = 'CREATE DATABASE IF NOT EXISTS '+database_name
		self.cursor.execute(sql)			
# use capital_data_test as DATABASE
		sql = 'USE '+database_name
		self.cursor.execute(sql)

	def create_tables(self):
# create nodes of the graph
		sql = "create table IF NOT EXISTS list_email ("\
		"id int auto_increment,"\
		"email  varchar(255) not null,"\
		"primary key (id)) ENGINE=InnoDB DEFAULT CHARSET=latin1"
		self.cursor.execute(sql)
	
		sql = "create table IF NOT EXISTS list_uid ("\
		"id int auto_increment,"\
		"uid  varchar(255) not null,"\
		"primary key (id)) ENGINE=InnoDB DEFAULT CHARSET=latin1"
		self.cursor.execute(sql)
	
		sql = "create table IF NOT EXISTS list_cookie ("\
		"id int auto_increment,"\
		"cookie  varchar(255) not null,"\
		"primary key (id)) ENGINE=InnoDB DEFAULT CHARSET=latin1"
		self.cursor.execute(sql)
# create edges of the graph
		sql = "CREATE TABLE IF NOT EXISTS link_uid_email ("\
		"id int auto_increment,"\
		"uid_id int,"\
		"email_id int,"\
		"foreign key (uid_id  )  references list_uid(id),"\
		"foreign key (email_id)  references list_email(id),"\
		"primary key (id)) ENGINE=InnoDB DEFAULT CHARSET=latin1"
		self.cursor.execute(sql)

		sql = "CREATE TABLE IF NOT EXISTS link_uid_cookie ("\
		"id int auto_increment,"\
		"uid_id int,"\
		"cookie_id int,"\
		"foreign key (uid_id  )  references list_uid(id),"\
		"foreign key (cookie_id) references list_cookie(id),"\
		"primary key (id)) ENGINE=InnoDB DEFAULT CHARSET=latin1"
		self.cursor.execute(sql)

		sql = "CREATE TABLE IF NOT EXISTS link_email_cookie ("\
		"id int auto_increment,"\
		"email_id int,"\
		"cookie_id int,"\
		"foreign key (email_id  )  references list_uid(id),"\
		"foreign key (cookie_id) references list_cookie(id),"\
		"primary key (id)) ENGINE=InnoDB DEFAULT CHARSET=latin1"
		self.cursor.execute(sql)
		self.connection.commit()

#insert node of the graph
	@jit
	def insert_mysql(self,type,type_name):
#		sql = "select id from list_"+type+" where "+type+" like '"+type_name+"'"
#		self.cursor.execute(sql)
#		id_output = self.cursor.fetchone()
#		if id_output is None:
		sql = "INSERT INTO list_"+type+"("+type+") value('"+type_name+"')"
		self.cursor.execute(sql)
		self.cursor.execute('SELECT last_insert_id()')
		id_output = self.cursor.fetchone()
		self.connection.commit()
		return id_output	

#insert edge of the graph	
	@jit
	def insert_link(self,type1,type2,id1,id2):
#		sql = "select id from link_"+type1+"_"+type2+" where "+type1+"_id = "+str(id1)+" and "+type2+"_id ="+str(id2)
#		self.cursor.execute(sql)
#		id_output = self.cursor.fetchone()
#		if id_output is None:
		sql = "INSERT INTO link_"+type1+"_"+type2+"("+type1+"_id,"+type2+"_id) values("+str(id1)+","+str(id2)+")"
		self.cursor.execute(sql)
		self.connection.commit()

