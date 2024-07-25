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
prompt_program = 'In continuous integration, there are some builds can be skipped. Now I will give you a build with some texts and features and you need to determine if this build can be skipped. You cannot give vague answers. Your answer must be either skip or not skip, it cannot be anything else. The features are: {}. The texts are: {}.'

def get_response(query_list, index_list, token, token_id):
    print(f"Processing on {token_id}, length of data is {len(query_list)}")
    time4threeQuery = 0
    loop_index = -1
    for index, query in tqdm(zip(index_list, query_list)):
        loop_index+=1
        i = 0 
        # .loc[feature.id==index]
        feature_text = 'last build result is ' + "pass" if feature.loc[feature.id==index].iloc[i, 1] == 0 else "fail" \
        +', the broken builds in all the previous builds is ' + str(feature.loc[feature.id==index].iloc[i,2])+ '%' \
        + ', size of the intersection of log_src_files and src_files is ' + str(feature.loc[feature.id==index].iloc[i,3]) \
        + ', whether tests throw exceptions is ' + "no" if feature.loc[feature.id==index].iloc[i, 4] == 0 else "yes" \
        + ', production files changed between the latest passed build and the previous build is ' + str(feature.loc[feature.id==index].iloc[i, 5]) \
        + ', fields modified, added or deleted is ' + str(feature.loc[feature.id==index].iloc[i, 6]) \
        + ', builds since the last broken build is ' + str(feature.loc[feature.id==index].iloc[i, 7]) \
        + ', lines of production code changed of previous build is ' + str(feature.loc[feature.id==index].iloc[i, 8]) \
        + ', commits on the files in last 3 months is ' + str(feature.loc[feature.id==index].iloc[i, 9]) \
        + ', the number of tests passed is ' + str(feature.loc[feature.id==index].iloc[i, 10]) \
        + ', sum of the probability of each changed file involved in previous broken builds is ' + str(feature.loc[feature.id==index].iloc[i, 11]) \
        + ',  broken builds in recent 5 builds that were triggered by the current committer is ' + str(feature.loc[feature.id==index].iloc[i, 12]+'%') \
        + ', the sum of consecutive broken builds is ' + str(feature.loc[feature.id==index].iloc[i, 13]) \
        + ', the maximum of the probability of each changed file involved in previous broken builds is ' + str(feature.loc[feature.id==index].iloc[i, 14]) \
        + ', overall time duration of the build is ' + str(feature.loc[feature.id==index].iloc[i, 15]) \
        + ', whether a core member triggers the build is ' + "no" if feature.loc[feature.id==index].iloc[i, 16] == 0 else "yes" \
        + ', the production files changed is ' + str(feature.loc[feature.id==index].iloc[i, 17]) \
        + ', the number of method bodies modified is ' + str(feature.loc[feature.id==index].iloc[i, 18]) \
        + ', the number of production files reported in the build log of the previous build is ' + str(feature.loc[feature.id==index].iloc[i, 19]) \
        + ', the number of import statements added is ' + str(feature.loc[feature.id==index].iloc[i, 20]) \
        + ', the size of team contributing in last 3 months is ' + str(feature.loc[feature.id==index].iloc[i, 22]) \
        + ', the number of files added is ' + str(feature.loc[feature.id==index].iloc[i, 23]) \
        + ', the number of classes modified, added or deleted is ' + str(feature.loc[feature.id==index].iloc[i, 24]) \
        + ', the number of methods deleted is ' + str(feature.loc[feature.id==index].iloc[i, 25]) \
        + ', whether test code is changed in AST is ' + "no" if feature.loc[feature.id==index].iloc[i, 26] == 0 else "yes" \
        + ', the number of fields deleted is ' + str(feature.loc[feature.id==index].iloc[i, 27]) \
        + ', the number of merge commits included is ' + str(feature.loc[feature.id==index].iloc[i, 28]) \
        + ', the number of lines of production code changed is ' + str(feature.loc[feature.id==index].iloc[i, 29]) \
        + ', the number of deleted lines in all files is ' + str(feature.loc[feature.id==index].iloc[i, 30]) \
        + ', the number of commits included is ' + str(feature.loc[feature.id==index].iloc[i, 31]) \
        + ', whether the build occurs on master branch is ' + "no" if feature.loc[feature.id==index].iloc[i, 32] == 0 else "yes" \
        + ', the number of added lines in all files is ' + str(feature.loc[feature.id==index].iloc[i, 33])
        query_program = prompt_program.format(feature_text, query)
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