#!/usr/bin/env python
import sys, traceback
from module_mysql import mySQLdb
import load_data as ld

config = {
  'user'    : 'root',
  'host'    : 'localhost',
  'password': 'password'
}

def main():
	try:
		if (len(sys.argv)>=2):
			database = mySQLdb(**config)
# create database
			database.create_database("capital_data_test")
# create tables
			database.create_tables()
			url = sys.argv[1]
			print("You are loading data from URL :",url)
			data = ld.load_data_from_url(url)
#			path = sys.argv[1]
#			data = ld.load_data_from_file(path)
			if ld.segmentation_data(data,database):
				print("Data loaded in database!")
			else:
				print("Data loading failed!")
				exit()
# close connection to mysql server
		else:
			print("Enter as argument an URL, please")
	except KeyboardInterrupt:
		print("Shutdown requested...exiting")
	except Exception:
		traceback.print_exc(file=sys.stdout)
	sys.exit(0)

if __name__ == "__main__":
    main()