# 1. Import Dependencies
# Import opencv
 import cv2 
# Import uuid
 import uuid
 # Import Operating System
 import os
 # Import time

# 2. Define Images to Collect
 labels = ['ሀ', 
'ለ',
 'መ',
 'ሰ',
 'በ',
 'መደሰት',
 'ቋንቋ',
 'እናት',
 'የምልክት',
 'ይቅርታ አድርግልኝ']
 number_imgs = 15
 Setup Folders
 IMAGES_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'collectedimage
 print(IMAGES_PATH)

                            
 # 4. Capture Images
if not os.path.exists(IMAGES_PATH):
 if os.name == 'posix':
 !mkdir -p {IMAGES_PATH}
 if os.name == 'nt':
 !mkdir {IMAGES_PATH}
 for label in labels:
 path = os.path.join(IMAGES_PATH, label)
 if not os.path.exists(path):
 !mkdir {path}

   
 # 5. Image Labelling
 for label in labels:
 cap = cv2.VideoCapture(0)
 print('Collecting images for {}'.format(label))
 time.sleep(5)
 for imgnum in range(number_imgs):
 print('Collecting image {}'.format(imgnum))
 ret, frame = cap.read()
 imgname = os.path.join(IMAGES_PATH,label,label+'.'+'{}.jpg'.format(str
 cv2.imwrite(imgname, frame)
 cv2.imshow('frame', frame)
 time.sleep(2)
 if cv2.waitKey(1) & 0xFF == ord('q'):
 break
 cap.release()
 cv2.destroyAllWindows()















                            



                            
