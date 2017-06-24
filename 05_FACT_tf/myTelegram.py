import telegram

token = '220076659:AAEM8D4zapcZ5pJJruJeAgRQY2q4ZcKlgFY'

def sendTelegram(message):
    bot = telegram.Bot(token=token)
    bot.send_message(chat_id=377868589, text=message)