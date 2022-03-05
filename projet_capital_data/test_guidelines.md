# CapitalData Application Test

This test is meant to both give the applicant a good overview of typical projects he will have to work on, and to give CapitalData a good overview of the applicant skills and "way of doing things"
The applicant is free to choose whatever language he wants 
(Golang being the primary language used at CapitalData and being well-suited for the task, using it might be a good point)

The core goals can be done in less than 180 minutes, stretch goals are optional but appreciated. This test is guideline-less on purpose to let the applicant expose all skills he wants. 
The submission must consist of a compilable source tree with a README file explaining how to compile, configure and 
run it from scratch

## Description of the Task :

* Goal : develop an application to save contact's information loaded from an API into a database (MySQL or SQLite). The application has to
    * handle the stream format.
         * one line per contact
         * each channel of this contact begins with a prefix that defines the value (@ = email, cookie = cookie, uid = uid)
    * handle the fact that one contact can have several emails, uids (user identification key) and cookies. 

* In this test, the internal software can :
    * Call an API 
    * Load and read the API response
    * Test if the response respect criteria
    * If it does, save all information in a database
    * Deals with potential errors
    * 4 levels are available: 
        * One information per contact, few contacts
        * Two or less information per contact, few contacts
        * Two or less information per contact, several contacts
        * five or less information per contact, several contacts

* API call:
    * The application will call this url : http://tech.kdata.fr:8080/your_firstname.your_lastname/level
    * Example http://tech.kdata.fr:8080/vincent.neto/3

## Rating

The rating of the submission will be based on multiple criteria, such as :

* simplicity: how simple is the code 
* readability: is the code understandable, readable, and specificated
* flexibility : is the application easily reconfigurable
* rapidity : how much time does the application take
* prediction : is it possible to predict this time
* optimization : is the application usable for Big Data 
