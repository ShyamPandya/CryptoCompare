import json
import requests
import pymongo

coins=['BTC','ETH','USD','CND','IOTA','XRP','LTC','DASH','EOS','GAS','COE','POE','ADA','DEEP','AIR']
api_url_base = 'https://min-api.cryptocompare.com/data/histominute?fsym='
headers = {'Content-Type': 'application/json'}
resultb=[]
resulte=[]
resultu=[]

def get_coin_list(coin_name,choice):
    if choice=='u':
        response = requests.get(api_url_base+coin_name+'&tsym=USD', headers=headers)
    elif choice=='e':
        response = requests.get(api_url_base+coin_name+'&tysm=ETH', headers=headers)
    elif choice=='b':
        response = requests.get(api_url_base+coin_name+'&tysm=BTC', headers=headers)

    if response.status_code == 200:
        return response.content.decode('utf-8')
    else:
        return None

for i in range(15):
    if coins[i]=='BTC':
        resultu.append(get_coin_list(coins[i],'u'))
        resulte.append(get_coin_list(coins[i], 'e'))
    elif coins[i]=='ETH':
        resultu.append(get_coin_list(coins[i], 'u'))
        resultb.append(get_coin_list(coins[i], 'b'))
    elif coins[i]=='USD':
        resultb.append(get_coin_list(coins[i], 'b'))
        resulte.append(get_coin_list(coins[i], 'e'))
    else:
        resultu.append(get_coin_list(coins[i], 'u'))
        resulte.append(get_coin_list(coins[i], 'e'))
        resultb.append(get_coin_list(coins[i], 'b'))

print(len(resultu))
print(len(resultb))
print(len(resulte))

