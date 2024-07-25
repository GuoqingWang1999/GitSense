import json
import pandas as pd
from tqdm import tqdm
import sys
import time
def read_data(file_name):
    items = []
    for i in open(file_name,'r').readlines():
        items.append(json.loads(i))
    return pd.DataFrame(items)

data =read_data("./data.json")
feature = read_data("./feature_data.json")
# If you want to run it by yourself, you can change the ChatGPT API-Key tokens here.
tokens = open("keys.txt").readlines()
tokens = [t[t.rindex('----')+len("----"):].strip() for t in tokens]

import openai
import datetime
#describe the CI scenario
prompt_program = 'In continuous integration, there are some builds can be skipped. Now I will give you a build with some texts and you need to determine if this build can be skipped. You cannot give vague answers. Your answer must be either skip or not skip, it cannot be anything else. The texts are: {}.'

def get_response(query_list, index_list, token, token_id):
    print(f"Processing on {token_id}, length of data is {len(query_list)}")
    time4threeQuery = 0
    loop_index = -1
    for index, query in tqdm(zip(index_list, query_list)):
        loop_index+=1
        i = 0 
        # .loc[feature.id==index]
        
        query_program = prompt_program.format(query)
        # cut the query_program under the restriction of GPT tokens
        if len(query_program) > 16000:
            query_program = query_program[:16000]
        with open(f'./result_{token_id}.json',"a+") as f:
            try:
                if (loop_index)%3==0 and loop_index>0:
                    sleepTime = 65-time4threeQuery
                    if sleepTime>0:
                        time.sleep(sleepTime)
                    time4threeQuery = 0
                starttime = datetime.datetime.now()
                openai.api_key = token
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-1106",
                    messages=[
                        {"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible."},
                        {"role": "user", "content": query_program}
                    ],
                    max_tokens = 500,
                    temperature = 0.2
                )
                endtime = datetime.datetime.now()
                time4threeQuery += (endtime - starttime).seconds
                f.write(json.dumps({'id':index,'response':dict(response)})+'\n')
            except Exception as e:
                f.write(json.dumps({'id':index,'response':"Failed"})+'\n')
                print(index, ": ", query_program)
                print("error: ", e)


# Multi threading
import threading

numsplit = 30
pernum = len(data)//numsplit
print(f"Per thread num: {pernum}")
for i in range(numsplit):
    if i+1 == numsplit:
        query_list = data.text.tolist()[i*pernum:]
        index_list = data.id.tolist()[i*pernum:]
    else:
        query_list = data.text.tolist()[i*pernum:(i+1)*pernum]
        index_list = data.id.tolist()[i*pernum:(i+1)*pernum]
    t = threading.Thread(target=get_response, args=(query_list,index_list,tokens[i],i,))
    t.start()