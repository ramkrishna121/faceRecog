# Automatic attendance system

Functionalities:
 
 1)Create dataset:
    This method is very helpful as this very option provides the registration part to every individual
    student. For every new registration at first it asks to provide the enrollment number and then
    an unique ID is created the database for that enrollment number.
    
 2)Training the classifier:
    Every image from the created dataset for every unique ID encoded in the form of vector and it
    is linked with the name of the student under the same dictionary. And when this is done every
    encoding along with it’s link is saved in a dot pickle (’.pickle’) file so that there is no need of
    training the data again.
    
 3)Take Attendance:
    This part plays a vital role as this marks the attendance of every student by comparing their faces
    from their respective stored dataset. At first when an image is captured, every face in that image
    is detected and then all the faces are matched by the series of encoded images and whichever
    is matched with any encoded image, the respective attendance for that student is marked under
    his/her name in the list.
    
 4)View Attendance:
    The view attendance option in the interface shows the marked attendances for all the recognized
    faces. The recognized faces get their marking of attendance as present or absent under their
    respective namesalong with the time of their entry.
    
 5)Personal Information:
    This parts plays a vital role in extracting the student’s information if we want to know the details
    of a student who is unknown. Whenever there is a need to know the details of any student then
    his/her face will be placed in front of camera and after detecting the face of the student, the
    system starts matching the it with all the encodings in the database. If there is any match found
    then the all the details of that student is shown.
    
