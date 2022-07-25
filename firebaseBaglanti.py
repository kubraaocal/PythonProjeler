import firebase_admin
from firebase_admin import credentials
from firebase_admin import  db


database_url = "https://toprakkokusu-c3451-default-rtdb.firebaseio.com"


def baglan_ve_json_Olustur():

    cred = credentials.Certificate("key.json")
    firebase_admin.initialize_app(cred,{
        'databaseURL' : database_url
    })

    ref = db.reference('/')
    best = ref.get()

    import json

    with open('data.json', 'w') as fp:
        json.dump(best, fp)






