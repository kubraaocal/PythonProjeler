import firebaseBaglanti, jsonToCsv, kumeleme, webAPI
import flask
from flask import url_for,request,render_template, jsonify
import os
app = flask.Flask(__name__)



#
@app.route('/', methods=['POST', 'GET'])
def get_json():
    if request.method == 'GET':
        return jsonify({'data': data })


def ServerBaslat():
    app.run()


ozellikler = ["facility","fire","network","paid","park","sea","transport","trekking","water","wc","wildAnimal"]

if __name__ == "__main__":
    file = "data.csv"
    os.remove(file)
    firebaseBaglanti.baglan_ve_json_Olustur()
    jsonToCsv.json2CSV()
    enCokKullanilanIndisler = kumeleme.kumele()

    data = [ozellikler[enCokKullanilanIndisler[0]], ozellikler[enCokKullanilanIndisler[1]]]

    ServerBaslat()





