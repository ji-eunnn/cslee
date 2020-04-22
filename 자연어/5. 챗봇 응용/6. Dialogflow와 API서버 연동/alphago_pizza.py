from flask import Flask, request, jsonify



app = Flask(__name__)



#----------------------------------------------------
# 피자 정보 처리
#----------------------------------------------------
def process_pizza_info(pizza_name):

    if pizza_name == u'불고기피자':
        answer = '한국의 맛 불고기를 부드러운 치즈와 함께!'
    elif pizza_name == u'페퍼로니피자':
        answer = '고소한 페파로니햄이 쫀득한 치즈위로 뜸뿍!'
    elif pizza_name == u'포테이토피자':
        answer = '저칼로리 감자의 담백한 맛!'

    return answer



#----------------------------------------------------
# 피자 주문 처리
#----------------------------------------------------
def process_pizza_order(pizza_name, address):
	
    answer = pizza_name + '를 주문하셨습니다.'
    answer += " '" + address + "'의 주소로 지금 배달하도록 하겠습니다."
    answer += ' 이용해주셔서 감사합니다.' 

    return answer



#----------------------------------------------------
# Dialogflow fullfillment 처리
#----------------------------------------------------
@app.route('/', methods=['POST'])
def webhook():

    #--------------------------------
    # 의도 구함
    #--------------------------------
    req = request.get_json(force=True)
    intent = req['queryResult']['intent']['displayName']

    #--------------------------------
    # 의도 처리
    #--------------------------------
    if intent == 'pizza_info':
        pizza_name = req['queryResult']['parameters']['pizza_type']
        answer = process_pizza_info(pizza_name)
    elif intent == 'pizza_order - custom':
        pizza_name = req['queryResult']['parameters']['pizza_type']
        address = req['queryResult']['parameters']['address']
        answer = process_pizza_order(pizza_name, address)
    else:
        answer = 'error'

    res = {'fulfillmentText': answer}
        
    return jsonify(res)



#----------------------------------------------------
# 웹서버 테스트
#----------------------------------------------------
@app.route('/test')
def test():
    
    return "<h1>This is test page!</h1>"



#----------------------------------------------------
# 메인 함수
#----------------------------------------------------
if __name__ == '__main__':

    app.run(host='0.0.0.0', threaded=True)    
    

