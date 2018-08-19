import os
from xml.etree import ElementTree as ET

f = []
count = 0
for path, subdirs, files in os.walk("Matches"):
    for name in files:
        file_path = os.path.join(path, name)
        print(file_path)
        try:
            xml = ET.parse(file_path)    
            root_element = xml.getroot()    
        #for child in root_element:
            #print(child)
        except:
            count+=1
print(count)