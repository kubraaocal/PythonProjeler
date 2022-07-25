import csv
import json

def json2CSV():
    # Opening JSON file
    f = open('data.json')

    # returns JSON object as
    # a dictionary
    data = json.load(f)

    # Iterating through the json
    # list
    for i in data['Camp']:
        satir = data['Camp'][i]

        dizi = ""
        dizi += "0,"

        for j in satir:
            sonuc = satir[j]
            if(str(sonuc) == "True"):

                dizi += "1,"
            elif(str(sonuc) == "False"):
                dizi += "0,"
        dizi = dizi[:-1]

        filename = 'data.csv'

        with open(filename, 'a', newline="") as file:
            file.write(dizi)
            file.write("\n")


    # Closing file
    f.close()