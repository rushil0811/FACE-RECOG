import cv2,os,pickle
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
data_file="faces_data.pkl"
if os.path.exists(data_file):
    with open(data_file,"rb") as file:
        known_faces=pickle.load(file)
else:
    known_faces={}
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    if not ret:break
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5,minSize=(30,30))
    for(x,y,w,h)in faces:
        matched_name="Unknown"
        detected_face=cv2.resize(gray[y:y+h,x:x+w],(100,100))
        detected_face=cv2.equalizeHist(detected_face)
        for name,stored_face in known_faces.items():
            stored_face=cv2.resize(stored_face,(100,100))
            stored_face=cv2.equalizeHist(stored_face)
            diff=sum(abs(detected_face[i][j]-stored_face[i][j])for i in range(100)for j in range(100))
            avg_diff=diff/(100*100)
            print(f"Comparing with {name}: Avg Diff={avg_diff}")
            if avg_diff<50:
                matched_name=name
                break
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,matched_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        if matched_name=="Unknown":
            cv2.putText(frame,"Press 's' to save new face",(50,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
    cv2.imshow("Face Recognition",frame)
    key=cv2.waitKey(1)&0xFF
    if key==ord("s"):
        name=input("Enter name for new face: ")
        if name:
            known_faces[name]=detected_face
            with open(data_file,"wb")as file:
                pickle.dump(known_faces,file)
            print(f"Saved face for {name}!")
    elif key==ord("q"):break
cap.release()
cv2.destroyAllWindows()