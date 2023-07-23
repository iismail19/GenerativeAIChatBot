# Creating Chatbot for PDF Files

## How to run

1. Clone repo
2. Setting us Python virtual env
    1. $ `cd GenerativeAIChatBot`
    2. $ `python3 -m venv env`
    3. $ `source env/bin/activate`
3. Run requirements.txt with python:
    1. `pip install -r requirements.txt` -> (Comming soon)
4. Update os.environ["HUGGINGFACEHUB_API_TOKEN"] = "<ENTER_KEY_HERE>" using your personal api key from https://huggingface.co/

### Application porgression

| Version | Name                       | Description                                                               |
|---------|----------------------------|---------------------------------------------------------------------------|
| 1       | terminalBasedChatbot.py    | Understanding a very basic version of creating a chatbot                  |
| 2       | LocalChatbot.py            | Creating a chatbot to perform  queries on a single pdf (not using openai) |
| 3       | LocalChatbotMultiplePdf.py | Creating a chatbot to perform queries on multiple PDFs                    | 