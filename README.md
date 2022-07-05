# Park EZ: An Automated Parking Management System for Easy Parking

Deep Learning Final Project by Marvee dela Resma, Patrick Guillano La Rosa, Miguel Carlo Pingol, Christian Alfred Soriano - March 21, 2022. The code, analysis, and the full report are included in the [Technical Report](https://github.com/pgplarosa/Automated-Parking-Management-System-for-Easy-Parking/blob/main/md/Technical_Report.md). If you have any questions regarding this study, please send me a message via  [LinkedIn](https://www.linkedin.com/in/patricklarosa/).

![header](https://user-images.githubusercontent.com/67182415/177184474-c5193c16-54c6-4856-a0cc-3a6acf99221f.PNG)

## Executive Summary

<p align="justify">Parking management systems tend to inconvenience customers and serve as cost centers to business owners. Park EZ remedies these pains by serving as a parking management system alternative which automates the entire process for the customer while reducing the costs needed from the owner.</p>

<p align="justify">Current automated parking management systems (PMS) involve sensors and computer systems which entail high costs. This project aims to be a jump off point in creating a PMS that relies only on CCTV camera feeds, which are commonly available in these establishments, and methods in computer vision.</p>

<p align="justify">The team sampled 156 frames from a CCTV footage of a parking lot, then train a custom YOLO model to detect cars with these manually labeled sampled images as training and validation set, The team created a rule based on the manually labeled irregular quadrilaterals for each of the parking spaces and the detected cars based on its Intersection Over Union (IoU) and its coordinates to decide if the parking space is taken or vacant. In order to attach a timer to each car, we located the centroid of each car and assigned a car id into it then time for each frame that the car id is in the view of the camera.</p>

<p align="justify">The application of transfer learning on YOLO has allowed for high precision of 100% on the validation set with fast detection. This allows for robust and real-time detection when deployed as a real-time car tracker. Thresholding for values for rules is dependent on the camera angle and distinguishes between parked cars and edge cases. Together with the timer and unique ID, Park EZ presents a proof of concept for a real time parking management system optimized custom for that parking lot.</p>

<p align="justify">Recommendations for this project involve additional features outside the current scope of the study but builds on the aspect of a fully developed parking management system for end-to-end deployment. To further improve this implementation, the authors suggest exploring security monitoring, deployment within a camera network, and a user app. Additional recommendations involve alternative use cases for the system. With modifications to the implementation, this system may be redeployed for employee monitoring, security systems, and traffic management.</p>

https://user-images.githubusercontent.com/67182415/177197134-81bd0fe8-1a6d-463a-9c0d-63ce7cb73ff8.mov

High quality output can be seen in this [link](https://github.com/pgplarosa/Automated-Parking-Management-System-for-Easy-Parking/blob/main/img/parking_cctv_timer.mp4).



