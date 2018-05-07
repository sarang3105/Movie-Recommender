			Project: Collaborative filtering		

SUBMITTED BY: SARANGDEEP SINGH and Ishan Agarwal
EMAIL: ssingh53@uncc.edu, iagarwa1@uncc.edu

Overview:
We ran our code on local cloudera machine because we could not execute the code on Amazon and DSBA cluster. SO we chose a smaller version
of our dataset.

STEPS TO EXECUTE linreg.py IN CLOUDERA:

1. Copy the input files to hdfs, the files to use are ratings.csv and movies.csv from the dataset provided:
   hadoop fs -put <source folder> <destination folder>
ex.hadoop fs -put /home/cloudera/rating.csv /user/cloudera/

2.Execute the following command:
   spark-submit <path of CF.py> <path of ratings.csv file on hdfs> <path of movies.csv file on hdfs> <user ID> 
ex.spark-submit /home/cloudera/Item-Item-CF.py /user/cloudera/rating.csv /user/cloudera/rating.csv 3


