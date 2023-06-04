import json

config = "config.json"

def getkey(file_name,key_string):
    with open(file_name,"r")as file:
        data = json.load(file)
        keys = key_string.split('.')
        
        for index, key in enumerate(keys):
            if key.isdigit() and isinstance(data,list):
                list_index = int(key)
                if list_index < len(data):
                    data = data[list_index]
                else:
                    print("Search failed wrong index: ",list_index,"in key string: ",end=" ")
                    for i in range(index + 1):
                        print(keys[i],end=" ")
                    return 
            elif isinstance(data,dict) and key in data:
                data = data[key]
            else:
                print("Search failed wrong key: ",end="")
                for i in range(index + 1):
                        print(keys[i])
                return
    return data

