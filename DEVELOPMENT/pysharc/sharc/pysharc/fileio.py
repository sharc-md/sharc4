import sys

def readFile(fileName,option="rb+"):
    """Read File and return text in file
       
       fileName = str, Name of the file to be read (or path+fileName)
       return   = str, Information in fileName
    """

    try:
        data = open(fileName,option)
    except:
        print(" Error reading file: " + fileName)
        sys.exit()

    text = data.read()
    data.close()  
    return text

def writeOutput(fileName,content):
    """ write content to file [fileName]

        fileName = str, Name of the file to be read 
        content  = str, content written to the file
        return   = None
    """
    try:
      OUT = open(fileName,"w")
    except:
      print("Error writing to file: "+fileName)
      sys.exit()
    
    OUT.write(content)
    OUT.close()
    return;
