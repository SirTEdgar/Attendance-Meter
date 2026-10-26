# Attendance-Meter
A Machine learning model that capture class attendance through live cctv footage

A beginner-friendly data science project that answers one question:

Can a camera record student attendance automatically, and can we learn useful things from that data?

The project has two parts:

Collect the data: a camera watches a classroom, recognises faces, and records who is present.
Analyse the data: we clean it, explore it, and use machine learning to spot students who may be falling behind.

What You Will Learn
Cleaning and combining data with pandas and SQL
Exploring data and calculating useful numbers (attendance rate, lateness)
Building and comparing simple machine learning models
Measuring model quality with precision, recall, and F1
Showing results in a simple dashboard

How it works:
Enrol students: save a few photos of each student.
Turn each face into numbers: a pretrained model converts a face into a list of 512 numbers (called an embedding). Photos of the same person give similar numbers.
Watch the camera: for each face seen, find which enrolled student has the most similar numbers.
Save the result: store who was seen, when, and how confident the match was.
Analyse: calculate attendance rates, find patterns, and predict who is at risk.


The Data

There are five simple tables.


students	Student ID, name, class, gender
sessions	Each lesson: class, subject, date, start time
detections	Raw camera results: who was seen, when, and confidence
attendance	Cleaned results: one row per student per lesson with a status
grades	Each student's scores


Machine Learning Used:

Predict at-risk students	Logistic Regression	Simple and easy to explain. This is our starting point (baseline).
Predict at-risk students	Random Forest	Handles more complex patterns. Shows which features matter most.
Group similar students	K-Means	Finds groups like "reliable", "often late", "often absent" without needing labels.
Find students with unusual changes	Isolation Forest	Flags unusual behaviour, such as a sudden drop in attendance.
Link attendance to grades	Correlation and Linear Regression	Shows whether students who attend more tend to score higher.

Convolution Neural networks:

Face detection: RetinaFace and SCRFD are CNN-based.
Face embeddings: ArcFace is a CNN (usually a ResNet backbone) that turns a face into 512 numbers.


Baseline: ArcFace embeddings + KNN.
CNN version: fine-tune MobileNetV2 (or ResNet18) on face crops of your volunteers.
Compare precision, recall, F1, training time, and what happens when a new student is added.

Expect KNN on ArcFace embeddings to be competitive or better, especially with few photos per student. That is still a good result, because it shows you tested an idea and chose based on evidence.
