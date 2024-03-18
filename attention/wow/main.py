from flask import Flask, request
from attention import get_completion, get_head_and_layer_nums
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

prompt = """
The 2008 Summer Olympics torch relay was run from March 24 until August 8, 2008, prior to the 2008 Summer Olympics, with the theme of "one world, one dream". Plans for the relay were announced on April 26, 2007, in Beijing, China. The relay, also called by the organizers as the "Journey of Harmony", lasted 129 days and carried the torch 137,000 km (85,000 mi) – the longest distance of any Olympic torch relay since the tradition was started ahead of the 1936 Summer Olympics.

After being lit at the birthplace of the Olympic Games in Olympia, Greece on March 24, the torch trav- eled to the Panathinaiko Stadium in Athens, and then to Beijing, arriving on March 31. From Beijing, the torch was following a route passing through six continents. The torch has visited cities along the Silk Road, symbolizing ancient links between China and the rest of the world. The relay also included an ascent with the flame to the top of Mount Everest on the border of Nepal and Tibet, China from the Chinese side, which was closed specially for the event.

Q: what is the main point of this text?
A:
""".strip()

#/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/BERT_TRAINING_SERVICE/platform/model/gpt2-medium
path = ""
device = "cuda:7"
tokenizer = None
model = None
trunks = None
attention = None
prompt_len = None
ave_layer_head_attention = None
num_head = None
num_layer = None
    

@app.route('/model_path', methods=['POST'])
def submit_text():
    global path, trunks, attention, prompt_len, ave_layer_head_attention, num_head, num_layer, tokenizer, model
   # from form get text input
    text_input = request.form['model_path']
    path = text_input
    print(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path).to(device)
    trunks, attention, prompt_len, ave_layer_head_attention = get_completion(prompt, model, tokenizer, device)
    num_head, num_layer = get_head_and_layer_nums(model)
    return text_input
print(trunks)
@app.route("/attention")
def attention_view():
    return json.dumps({
        'tokens_text': trunks,
        'attention_values': attention,
        'prompt_len': prompt_len,
        'num_layer': num_layer,
        'num_head': num_head,
        'average_attention_value': ave_layer_head_attention
    })

# atten_m
# tuple,tuple,
# seq_len, layer_num, tensor(this tensor represents this token's attention to other tokens in different head)
# attn_m[0] is the attention martix of prompt, so begin from 1


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

#127.0.0.1:5000/static/index.html
