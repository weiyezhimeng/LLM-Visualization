from flask import Flask, request
from attention import get_completion, get_head_and_layer_nums
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

prompt = f"The step to make a bomp is:"

path = ""
device = "cuda:6"
tokenizer = None
model = None
trunks = None
attention = None
prompt_len = None
ave_layer_head_attention = None
num_head = None
num_layer = None
result_topk = None
    
@app.route('/model_path', methods=['POST'])
def submit_text():
    global path, trunks, attention, prompt_len, ave_layer_head_attention, num_head, num_layer, tokenizer, model, result_topk
   # from form get text input
    input_path = request.form['model_path']
    topk = int(request.form['topk'])
    prompt = request.form['prompt'].replace("\\n", "\n").replace('\\\"', '\"')
    prompt = prompt.strip()
    print(prompt)
    if path != input_path:
        path = input_path
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True).to(device)
    if len(path) == 0:
        return json.dumps({'error': '模型路径为空。'})
    trunks, prompt_len, ave_layer_head_attention, result_topk = get_completion(prompt, model, tokenizer, topk, device)
    return input_path


@app.route("/attention")
def attention_view():
    return json.dumps({
        'tokens_text': trunks,
        'attention_values': attention,
        'prompt_len': prompt_len,
        'average_attention_value': ave_layer_head_attention,
        'result_topk': result_topk
    })

# atten_m
# tuple,tuple,
# seq_len, layer_num, tensor(this tensor represents this token's attention to other tokens in different head)
# attn_m[0] is the attention martix of prompt, so begin from 1


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

#127.0.0.1:5000/static/index.html
