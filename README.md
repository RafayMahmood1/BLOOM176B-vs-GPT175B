# Email Response Generation using GPT-3 and BLOOM
This code generates email responses using two state-of-the-art language models, GPT-3 and BLOOM, with the inference API of both models. The code takes an email as input and generates two responses, one from each model, which can be used to quickly reply to the email.

## Installation
To use this code, you need to have an OpenAI API key for GPT-3 and a BLOOM API key. 
For GPT-3 API key you need to sign up for Open AI
For BLOOM API key you need to sign up on github and use its inference

Once you have the API keys, you need to clone this repository to your local machine and install the required Python packages:

## Usage
To generate email responses using GPT-3 and BLOOM, run the BLOOM-GPT.ipynb script


## What to Achieve
To compare both GPT3 and BLOOM for response generation of Emails

## Code

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom")
```

    /Users/rafay/anaconda3/envs/torch/lib/python3.10/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm
    Downloading: 100%|█████████████████████████████| 222/222 [00:00<00:00, 67.9kB/s]
    Downloading: 100%|█████████████████████████| 14.5M/14.5M [00:05<00:00, 2.54MB/s]
    Downloading: 100%|███████████████████████████| 85.0/85.0 [00:00<00:00, 29.7kB/s]



```python
import re
def preProcessText(text):
    # remove links
    text = re.sub(
        "(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])",
        " ",
        text,
    )

    text = re.sub(r"\n", " ", text)  # remove /r /n etc
    text = re.sub(r"\r", " ", text)  # remove /r /n etc
    text = re.sub(r" +", " ", text)  # remove multiple spaces
    text = re.split("^\s*(On\s([a-zA-Z]){3}.*)", text, flags=re.M)[0] 
    text = re.split("^\s*(From:\s([a-zA-Z]){3}.*)", text, flags=re.M)[0]
    text = text.split("wrote:")[0]  # remove thread part

    return text
```


```python
state='''Hey James,

I just wanted to touch base with you about a new project I am working on.

I've got some research data on this topic that I'd like to share with you. Let's talk about it over coffee/beer/wine/whatever.

I've attached the PDF of our whitepaper.

Regards,
Paul Walker

'''
```


```python
# prompt = preProcessText(prompt)
# prompt = f"Write a response for me to the email.\nEmail:\n{state}\n\n###\n\nResponse:\n"
# prompt = f"Answer the following question.\nQuestion: How do I make chicken masala\n\n###\n\nAnswer:\n"
state = preProcessText(state)
prompt = f"Response to the following email.\nEmail:\n{state}.\n\n###\nResponse:\n"
```


```python
print(prompt)
```

    Response to the following email.
    Email:
    Hey James, I just wanted to touch base with you about a new project I am working on. I've got some research data on this topic that I'd like to share with you. Let's talk about it over coffee/beer/wine/whatever. I've attached the PDF of our whitepaper. Regards, Paul Walker .
    
    ###
    Response:
    



```python
def bloom(state):
    end_sequence='###'
    state = state.replace('"','')

    prompt = f"Response to the following email.\nEmail:\n{state}\n\n###\nResponse:\n"
    import requests

    API_URL = "################################"
    headers = {"Authorization": "Bearer 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"}

    result_length = 250

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    output = query({
        "inputs": prompt,
        "parameters" : {
                         "temperature": 0.5,
                          "max_new_tokens":200,
            "eos_token_id": int(tokenizer.convert_tokens_to_ids(end_sequence)),
        "return_full_text": False,


                        },
                "options": 
            {  
                "use_cache": False,
            }
        })

    new = output[0]['generated_text']

    new = new.split("###")[0]
    check = new
    output = check
    
    ls_state = state.split('.')
    out2 = output.split('.')
    for i in ls_state:
        if i in out2:
            out2.remove(i)
            
    out2 = list(dict.fromkeys(out2))
    output = '.'.join(out2)
    
    out2 = output.split('\n')
    
    for i in ls_state:
        if i in out2:
            out2.remove(i)
            
    out2 = list(dict.fromkeys(out2))
    output = ' '.join(out2)
    
    
    output = output.replace(state,'')
    
    out3 = output.split("\n\n")
    output = out3[0]    
    return output
```


```python
def gpt3(state):
    import requests
    text = state
    selectedTone = "Professional"

    API_URL = "###################################"
    payload = {"email": f'"""{text}"""', "tone": selectedTone}
    files = []
    headers = {}
    response = requests.request(
        "POST", API_URL, headers=headers, data=payload, files=files
    )
    json_response = response.json()
    lead_response = json_response["output"]["choices"][0]["text"]
    final_response = f"GPT:{lead_response}"
    
    return final_response

```


```python
final_response = gpt3(state)
output = bloom(state)

print(prompt+"\n\n"+"-"*100+"\n"+final_response+"\n"+"-"*100+"\n" +"\n"+"BLOOM:\n" +output)
```

    Response to the following email.
    Email:
    Hey James, I just wanted to touch base with you about a new project I am working on. I've got some research data on this topic that I'd like to share with you. Let's talk about it over coffee/beer/wine/whatever. I've attached the PDF of our whitepaper. Regards, Paul Walker .
    
    ###
    Response:
    
    
    ----------------------------------------------------------------------------------------------------
    GPT:
    Hi Paul,
    
    Thank you for reaching out. I appreciate the opportunity to discuss your project with you. Let's schedule a meeting to discuss your data further. Thank you for providing the PDF of the white paper. I look forward to hearing more about your project.
    ----------------------------------------------------------------------------------------------------
    
    BLOOM:
    Hi Paul, Thanks for the email. I would love to have a chat with you about this project. I am in town this week, so if you are free on Tuesday afternoon, let me know and we can meet up. Regards, James 



```python

```
