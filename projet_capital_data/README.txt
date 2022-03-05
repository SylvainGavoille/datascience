
For this project, i used python 3.6 and MySQL for the database.
I chose python3.6 due the limited time for development of this project. 
Python is really efficient to develop fastly. In other circonstance, i will have used C++, or even Golang to improve the performance.

To launch the program you need to specify as argument the URL.
example on Windows :
python main.py http://tech.kdata.fr:8080/vincent.neto/3

PRE-REQUIREMENT
In term of packages, i installed:
- urllib3 : to get access to URL API
- pymysql : API from python to MySQL
- progressbar2 : it allows to show advancement in the loading of data
- numba : to use decorator @jit in python to optimize computational performance

CONFIG:
The program is composed of main program "main.py" and two modules:
- load_data.py : it concerns loading of data and segmentation to form the database.
- module_mysql.py : inside this module is define a class object "mySQLdb" that allows to connect to the MySQL server and launch MySQL queries.

At the beginning of main.py, it is necessary to enter the configuration of your MySQL server.

ORGANISATION OF DATAS:

The whole dataset named "capital_data_test" is composed of 6 tables:
+-----------------------------+
| Tables_in_capital_data_test |
+-----------------------------+
| link_email_cookie           |
| link_uid_cookie             |
| link_uid_email              |
| list_cookie                 |
| list_email                  |
| list_uid                    |
+-----------------------------+
The organisation of datas has been designed as a graph where nodes are uid,email and cookie informations are respectively in list_uid, list_email and list_cookie tables,  and where edges
are the link between the different entities. 

This design allow to have an access to informations from uid or an email in a linear time with SQL queries.
Another matter, data could be duplicated in the program given. To make it not duplicated, uncomment the concerning check in module_mysql.py. But, if you uncomment this part of the code, insertion is not in linear time anymore.
DETAILS ABOUT TABLE:

For example, here we give the descriptioin of list_uid. list_email and list_cookie are designed with similarity.

+-------+--------------+------+-----+---------+----------------+
| Field | Type         | Null | Key | Default | Extra          |
+-------+--------------+------+-----+---------+----------------+
| id    | int(11)      | NO   | PRI | null    | auto_increment |
| uid   | varchar(255) | NO   |     | null    |                |
+-------+--------------+------+-----+---------+----------------+

There is 2 features:
- id is the primary key.
- uid contains the uid's data

Concerning link, which allow to have acces to connection between datas. We present below "link_uid_cookie" datastructure. The same philosophy is used for link_email_cookie and link_uid_cookie.

+----------+---------+------+-----+---------+----------------+
| Field    | Type    | Null | Key | Default | Extra          |
+----------+---------+------+-----+---------+----------------+
| id       | int(11) | NO   | PRI | null    | auto_increment |
| uid_id   | int(11) | YES  | MUL | null    |                |
| email_id | int(11) | YES  | MUL | null    |                |
+----------+---------+------+-----+---------+----------------+

There is 2 features:
- id is the primary key.
- uid_id is a foreign key refencing to list_uid id.
- email_id is a foreign key refencing to list_email id.




